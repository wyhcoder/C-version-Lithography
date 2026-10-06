#include "full_chip_lsm.h"
#include "gradient.h"
#include "level_set_utils.h"
#include "loss.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace litho;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

ImagingCache make_cache(int n, bool socs = true) {
    ImagingCache cache{};
    cache.N = n;
    cache.source_ws = Eigen::VectorXd::Constant(2, 0.5);
    if (socs) cache.socs_eigenvalues = {0.7, 0.3};
    for (int k = 0; k < 2; ++k) {
        Eigen::MatrixXcd kernel(n, n);
        for (int y = 0; y < n; ++y) {
            for (int x = 0; x < n; ++x) {
                const double fy = static_cast<double>(y <= n / 2 ? y : y - n) / n;
                const double fx = static_cast<double>(x <= n / 2 ? x : x - n) / n;
                kernel(y, x) = std::polar(std::exp(-(12.0 + 8.0 * k) * (fx * fx + fy * fy)), k * (0.35 * fx - 0.2 * fy));
            }
        }
        cache.optical_kernels_frequency.push_back(std::move(kernel));
    }
    return cache;
}

Eigen::MatrixXd make_target() {
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(32, 32);
    target.block(8, 8, 16, 16).setOnes();
    target.block(14, 14, 4, 4).setZero();
    return target;
}

Eigen::MatrixXd make_initial() {
    Eigen::MatrixXd mask = Eigen::MatrixXd::Zero(32, 32);
    mask.block(6, 7, 20, 18).setOnes();
    mask.block(14, 14, 4, 4).setZero();
    return mask;
}

void check_consistency(const PartitionMeta& meta, const ImagingCache& cache, const Eigen::MatrixXd& initial, const Eigen::MatrixXd& target,
                       const FullChipLSMOptions& options, const FullChipLSMResult& result) {
    require(result.history.size() == options.iterations + 1, "missing initial or final evaluation");
    const auto best = std::min_element(result.history.begin(), result.history.end(),
                                     [](const auto& a, const auto& b) { return a.loss < b.loss; });
    require(result.best_iteration == best->iteration && result.best_loss == best->loss, "wrong best-state selection");
    require(result.global_phi.allFinite() && result.global_mask.allFinite(), "non-finite returned state");
    require((result.global_mask - (result.global_phi.array() >= 0.0).cast<double>().matrix()).norm() == 0.0,
            "global phi and mask belong to different states");
    require((result.global_mask - Partition::assemble_global(result.local_masks, meta)).norm() == 0.0, "wrong Core assembly");
    for (const auto& tile : meta.tiles) {
        const auto& phi = result.local_phi[tile.id];
        require((result.local_masks[tile.id] - (phi.array() >= 0.0).cast<double>().matrix()).norm() == 0.0,
                "local phi and mask do not match");
        // 独立重建本块初始 SDF，验证窗口外保留的是本地初值，而不是最新 Core 回拼后的上下文。
        const auto local_initial = Partition::extract_local_tile(initial, tile);
        const auto initial_phi = LevelSetUtils::reinit_sdf((local_initial.array() - 0.5).matrix(), options.dx, options.dy);
        const auto& w = tile.update_local;
        for (int y = 0; y < phi.rows(); ++y) {
            for (int x = 0; x < phi.cols(); ++x) {
                if (y >= w.y0 && y < w.y1 && x >= w.x0 && x < w.x1) continue;
                require(phi(y, x) == initial_phi(y, x), "phi outside Core + Guard changed from its local initial value");
            }
        }
    }
    MicroGradientExchange exchange(meta, cache, 0.25, 8.0);
    auto gradient_options = options.gradient_options;
    gradient_options.loss_region = MicroLossRegion::CoreGuard;
    gradient_options.cept_guard_only = true;
    const auto reevaluated = exchange.compute(result.local_masks, target, gradient_options, false);
    require(std::abs(reevaluated.loss - result.best_loss) < 1e-9, "returned best state does not reproduce best loss");
}

void check_single_micro(bool socs, int reinit_interval) {
    const auto meta = Partition::build(32, 32, 1, 1, 0, 0);
    const auto cache = make_cache(32, socs);
    const auto initial = make_initial(), target = make_target();
    FullChipLSMOptions options;
    options.iterations = 12;
    options.reinit_interval = reinit_interval;
    options.verbose = false;
    FullChipLSMOptimizer optimizer(meta, cache, 0.25, 8.0);
    const auto result = optimizer.optimize(initial, target, options);

    // 独立用原单块 Imaging/Gradient/LevelSetUtils 执行完整轨迹，核对 RK2、CFL、重初始化和最佳轮 phi。
    Imaging imaging(cache);
    Gradient gradient(cache, 0.25, 8.0);
    Eigen::MatrixXd phi = LevelSetUtils::reinit_sdf((initial.array() - 0.5).matrix(), 1.0, 1.0);
    Eigen::MatrixXd best_phi;
    double best_loss = std::numeric_limits<double>::infinity(), dt = 0.0;
    int best_iteration = 0;
    for (int iteration = 0; iteration <= options.iterations; ++iteration) {
        const Eigen::MatrixXd mask = (phi.array() >= 0.0).cast<double>().matrix();
        const auto image = imaging.compute(mask, 0.25, 8.0);
        const double loss = Loss::pe_loss(image.wafer_image, target);
        require(std::abs(loss - result.history[iteration].loss) < 1e-10, "single Micro loss differs from original primitives");
        require(std::abs(dt - result.history[iteration].min_dt) < 1e-12, "wrong per-Micro CFL step");
        if (loss < best_loss) {
            best_loss = loss;
            best_iteration = iteration;
            best_phi = phi;
        }
        if (iteration == options.iterations) break;
        const auto gm = gradient.pe_gradient(image.wafer_image, target, imaging.get_electric_field());
        const auto first = LevelSetUtils::evolve_normal_WENO_godunov(phi, gm, 1.0, 1.0);
        dt = LevelSetUtils::get_dt_normal_kappa(options.cfl, 1.0, 1.0, first.H1_abs, first.H2_abs, 0.01);
        const Eigen::MatrixXd phi1 = phi + dt * (LevelSetUtils::evolve_kappa(phi, 1.0, 1.0, 0.01) - first.delta);
        const auto second = LevelSetUtils::evolve_normal_WENO_godunov(phi1, gm, 1.0, 1.0);
        phi = 0.5 * phi + 0.5 * (phi1 + dt * (LevelSetUtils::evolve_kappa(phi1, 1.0, 1.0, 0.01) - second.delta));
        if (reinit_interval > 0 && (iteration + 1) % reinit_interval == 0) {
            const Eigen::MatrixXd signs = ((phi.array() >= 0.0).cast<double>() - 0.5).matrix();
            phi = LevelSetUtils::reinit_sdf(signs, 1.0, 1.0);
        }
    }
    require(result.best_iteration == best_iteration, "single Micro did not return historical minimum");
    require((result.global_phi - best_phi).cwiseAbs().maxCoeff() < 1e-11, "best phi does not match independent trajectory");
    check_consistency(meta, cache, initial, target, options, result);
}

void check_guard_is_independent() {
    const auto meta = Partition::build(32, 32, 2, 2, 8, 3);
    const auto cache = make_cache(meta.tile_size);
    Eigen::MatrixXd initial = Eigen::MatrixXd::Ones(32, 32);
    initial.block(0, 0, 2, 2).setZero();
    FullChipLSMOptions options;
    options.iterations = 0;
    options.verbose = false;
    options.gradient_options.enable_cept = true;
    FullChipLSMOptimizer optimizer(meta, cache, 0.25, 8.0);
    const auto result = optimizer.optimize(initial, initial, options);
    bool different_guard = false;
    for (const auto& tile : meta.tiles) {
        const auto local = Partition::extract_local_tile(initial, tile);
        const Eigen::MatrixXd expected = LevelSetUtils::reinit_sdf((local.array() - 0.5).matrix(), 1.0, 1.0);
        const auto context = Partition::extract_local_tile(result.global_phi, tile);
        const auto& w = tile.update_local;
        const auto actual = result.local_phi[tile.id].block(w.y0, w.x0, w.rows(), w.cols());
        require((result.local_phi[tile.id] - expected).norm() == 0.0, "initial local phi was overwritten by Core context");
        different_guard |= (actual - context.block(w.y0, w.x0, w.rows(), w.cols())).norm() > 1e-8;
    }
    require(different_guard, "fixture must distinguish independent Guard phi from owner Core phi");
    require(result.history[0].cept_loss == 0.0, "CEPT must compare mask values, not initial phi differences");
    check_consistency(meta, cache, initial, initial, options, result);
}

void check_multiple_micros() {
    const auto meta = Partition::build(32, 32, 2, 2, 8, 3);
    const auto cache = make_cache(meta.tile_size);
    const auto initial = make_initial(), target = make_target();
    FullChipLSMOptimizer optimizer(meta, cache, 0.25, 8.0);
    for (const auto type : {MicroLossType::PE, MicroLossType::PEPenalty, MicroLossType::EPE, MicroLossType::PE_EPE}) {
        for (const auto mode : {GradientFusionMode::Average, GradientFusionMode::Linear}) {
            FullChipLSMOptions options;
            options.iterations = 8;
            options.reinit_interval = 3;
            options.verbose = false;
            options.gradient_options.loss_type = type;
            options.gradient_options.epe_interval_line = 2;
            options.gradient_options.epe_interval_corner = 2;
            options.gradient_options.enable_cept = true;
            options.gradient_options.enable_grad_fusion = true;
            options.gradient_options.fusion_mode = mode;
            options.parallel_micro = false;
            const auto serial = optimizer.optimize(initial, target, options);
            options.parallel_micro = true;
            const auto parallel = optimizer.optimize(initial, target, options);
            require(serial.best_iteration == parallel.best_iteration, "serial and parallel best iterations differ");
            require((serial.global_phi - parallel.global_phi).cwiseAbs().maxCoeff() < 1e-11, "parallel PDE changed phi");
            for (int i = 0; i <= options.iterations; ++i) {
                require(std::abs(serial.history[i].loss - parallel.history[i].loss) < 1e-9, "parallel loss trajectory changed");
                require(std::isfinite(parallel.history[i].loss), "non-finite loss");
                if (i > 0) {
                    require(parallel.history[i].min_dt > 0.0 && parallel.history[i].max_dt >= parallel.history[i].min_dt,
                            "invalid local CFL step range");
                }
            }
            check_consistency(meta, cache, initial, target, options, parallel);
        }
    }
}

int main() {
#ifdef _OPENMP
    omp_set_num_threads(2);
    omp_set_max_active_levels(2);
#endif
    check_single_micro(false, 0);
    check_single_micro(true, 0);
    check_single_micro(true, 3);
    check_guard_is_independent();
    check_multiple_micros();
    std::cout << "Full-chip LSM: RK2/CFL, independent Guard, mask CEPT, best-state and parallel consistency passed.\n";
}
