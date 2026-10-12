#include "full_chip_ctm.h"
#include "CTM_Optimizer.h"
#include "loss.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using namespace litho;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

ImagingCache make_cache(int n, bool socs) {
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
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(24, 24);
    target.block(7, 5, 10, 14).setOnes();
    target.block(10, 10, 4, 4).setZero();
    return target;
}

void check_single_micro(bool socs, double learning_rate = 0.01, int iterations = 4, bool expect_earlier_best = false) {
    const auto meta = Partition::build(24, 24, 1, 1, 0, 0);
    const auto cache = make_cache(meta.tile_size, socs);
    const auto target = make_target();
    FullChipCTMOptimizer optimizer(meta, cache, 0.25, 8.0);
    FullChipCTMOptions options;
    options.iterations = iterations;
    options.learning_rate = learning_rate;
    options.verbose = false;
    options.retain_final_masks = true;
    const auto result = optimizer.optimize(target, options);

    // 独立用原单块 Imaging/Gradient 执行完整轨迹，核对各轮 loss 和返回的历史最佳 theta/mask。
    Imaging imaging(cache);
    Gradient gradient(cache, 0.25, 8.0);
    Eigen::MatrixXd theta = CTM_Optimizer::initialize_theta(target);
    Eigen::MatrixXd best_theta, best_mask;
    double best_loss = std::numeric_limits<double>::infinity();
    int best_iteration = 0;
    for (int iteration = 0; iteration <= options.iterations; ++iteration) {
        const Eigen::MatrixXd mask = (0.5 * (1.0 + theta.array().cos())).matrix();
        const auto image = imaging.compute(mask, 0.25, 8.0);
        const double loss = Loss::pe_loss(image.wafer_image, target);
        require(std::abs(loss - result.history[iteration].loss) < 1e-11, "single Micro history differs from standalone GD");
        if (loss < best_loss) {
            best_loss = loss;
            best_iteration = iteration;
            best_theta = theta;
            best_mask = mask;
        }
        if (iteration < options.iterations) {
            const auto mask_gradient = gradient.pe_gradient(image.wafer_image, target, imaging.get_electric_field());
            theta += 0.5 * options.learning_rate * (mask_gradient.array() * theta.array().sin()).matrix();
        }
    }
    require(best_mask.isApprox(result.global_mask, 1e-12), "returned global mask is not the minimum-loss state");
    // 大步长样例的末轮与最佳轮不同；单独核对固定次数快照，避免对照实验误用最佳轮。
    const Eigen::MatrixXd final_mask = (0.5 * (1.0 + theta.array().cos())).matrix();
    require(final_mask.isApprox(result.final_local_masks[0], 1e-12), "retained final mask differs from standalone trajectory");
    require(best_mask.isApprox(result.local_masks[0], 1e-12), "returned local mask is not the minimum-loss state");
    require(best_theta.isApprox(result.local_theta[0], 1e-12), "returned theta does not belong to best mask");
    require(result.best_iteration == best_iteration && std::abs(result.best_loss - best_loss) < 1e-11, "best metadata differs");
    if (expect_earlier_best) {
        require(best_iteration < options.iterations && result.history.back().loss > best_loss, "regression case has no late loss increase");
    } else {
        require(best_loss < result.history.front().loss, "fixed-step single Micro GD did not reduce PE");
    }
}

void check_guard_only_update(bool socs) {
    const auto meta = Partition::build(24, 24, 2, 2, 4, 2);
    const auto cache = make_cache(meta.tile_size, socs);
    const auto target = make_target();
    FullChipCTMOptimizer optimizer(meta, cache, 0.25, 8.0);
    FullChipCTMOptions options;
    options.iterations = 2;
    options.learning_rate = 0.01;
    options.verbose = false;
    options.gradient_options.enable_cept = true;
    options.gradient_options.enable_grad_fusion = true;
    // 故意传 false，确认 full-chip CTM 内部强制 Guard-only，而不是仅修改默认配置。
    options.gradient_options.cept_guard_only = false;
    auto expected_options = options.gradient_options;
    expected_options.cept_guard_only = true;
    auto expected_theta = Partition::extract_all(CTM_Optimizer::initialize_theta(target), meta);
    const auto initial_theta = expected_theta;
    MicroGradientExchange exchange(meta, cache, 0.25, 8.0);
    const auto result = optimizer.optimize(target, options);
    double best_loss = std::numeric_limits<double>::infinity(), core_dual_difference = 0.0;
    int best_iteration = 0;
    std::vector<Eigen::MatrixXd> best_theta;
    bool active_cept = false;
    // 第一轮初值的 CEPT 为零；至少推进两轮，才能核对非零 CEPT 的驱动确实不包含反向 Core 端。
    for (int iteration = 0; iteration <= options.iterations; ++iteration) {
        std::vector<Eigen::MatrixXd> masks;
        for (const auto& theta : expected_theta) masks.push_back((0.5 * (1.0 + theta.array().cos())).matrix());
        const auto evaluation = exchange.compute(masks, target, expected_options);
        require(std::abs(evaluation.loss - result.history[iteration].loss) < 1e-10, "CTM differs from Guard-only trajectory");
        if (evaluation.loss < best_loss) {
            best_loss = evaluation.loss;
            best_iteration = iteration;
            best_theta = expected_theta;
        }
        for (const auto& tile : meta.tiles) {
            const auto& core = tile.core_local;
            require(evaluation.locals[tile.id].cept_mask_gradient.block(core.y0, core.x0, core.rows(), core.cols()).norm() == 0.0,
                    "Guard-only CEPT directly drives Core pixels");
        }
        if (evaluation.cept_loss > 0.0 && iteration < options.iterations) {
            active_cept = true;
            auto dual_options = expected_options;
            dual_options.cept_guard_only = false;
            const auto dual = exchange.compute(masks, target, dual_options);
            for (const auto& tile : meta.tiles) {
                const auto& core = tile.core_local;
                core_dual_difference += dual.locals[tile.id].cept_mask_gradient.block(
                    core.y0, core.x0, core.rows(), core.cols()).norm();
            }
        }
        if (iteration == options.iterations) break;
        // 逐像素执行链式更新，不复用 optimizer 的 Eigen block 表达式；窗口外从不赋新值。
        for (const auto& tile : meta.tiles) {
            const auto& window = tile.update_local;
            for (int y = window.y0; y < window.y1; ++y) {
                for (int x = window.x0; x < window.x1; ++x) {
                    double& theta = expected_theta[tile.id](y, x);
                    theta += options.learning_rate * 0.5 * evaluation.locals[tile.id].mask_gradient(y, x) * std::sin(theta);
                }
            }
        }
    }
    require(active_cept && core_dual_difference > 1e-12, "fixture does not distinguish Guard-only from dual-end CEPT");
    require(result.best_iteration == best_iteration && best_iteration > 0, "Guard-only update selected the wrong best state");
    double cept_loss = 0.0;
    for (const auto& tile : meta.tiles) {
        require(result.local_theta[tile.id].isApprox(best_theta[tile.id], 1e-12), "Core/Guard theta update is incorrect");
        const auto& window = tile.update_local;
        for (int y = 0; y < meta.tile_size; ++y) {
            for (int x = 0; x < meta.tile_size; ++x) {
                if (y >= window.y0 && y < window.y1 && x >= window.x0 && x < window.x1) continue;
                require(result.local_theta[tile.id](y, x) == initial_theta[tile.id](y, x), "theta outside Core + Guard changed");
            }
        }
        require(result.local_masks[tile.id].isApprox((0.5 * (1.0 + result.local_theta[tile.id].array().cos())).matrix(), 1e-12),
                "theta and mask belong to different states");
        for (const auto& neighbor : tile.neighbors) {
            const auto& a = neighbor.my_guard_local;
            const auto& b = neighbor.neighbor_overlap_local;
            cept_loss += (result.local_masks[tile.id].block(a.y0, a.x0, a.rows(), a.cols()) -
                          result.local_masks[neighbor.id].block(b.y0, b.x0, b.rows(), b.cols())).squaredNorm();
        }
    }
    require(cept_loss > 0.0, "updated independent Guards produced no CEPT mismatch");
    require(std::abs(cept_loss - result.history[best_iteration].cept_loss) < 1e-12, "CEPT does not belong to best-state masks");
    require(result.global_mask.isApprox(Partition::assemble_global(result.local_masks, meta), 1e-12), "Core assembly differs");
    const auto selected = exchange.compute(result.local_masks, target, expected_options);
    require(std::abs(selected.loss - result.best_loss) < 1e-10, "best history loss is stale");
}

void check_parallel_workflow(bool socs) {
    const auto meta = Partition::build(24, 24, 2, 2, 4, 2);
    const auto cache = make_cache(meta.tile_size, socs);
    const auto target = make_target();
    FullChipCTMOptimizer optimizer(meta, cache, 0.25, 8.0);
    MicroGradientExchange verification(meta, cache, 0.25, 8.0);
    FullChipCTMOptions options;
    options.iterations = 3;
    options.learning_rate = 0.005;
    options.verbose = false;
    options.gradient_options.enable_cept = true;
    options.gradient_options.epe_interval_line = 3;
    options.gradient_options.epe_interval_corner = 1;
    for (const auto loss : {MicroLossType::PE, MicroLossType::PEPenalty, MicroLossType::EPE, MicroLossType::PE_EPE}) {
        options.gradient_options.loss_type = loss;
        for (const bool fusion : {false, true}) {
            options.gradient_options.enable_grad_fusion = fusion;
            options.gradient_options.fusion_mode = loss == MicroLossType::PE ? GradientFusionMode::Average : GradientFusionMode::Linear;
            options.parallel_micro = false;
            const auto serial = optimizer.optimize(target, options);
            options.parallel_micro = true;
            const auto parallel = optimizer.optimize(target, options);
            require(parallel.history.size() == 4 && parallel.history.back().iteration == 3, "incorrect number of updates");
            require(parallel.global_mask.isApprox(serial.global_mask, 1e-12), "serial and parallel final masks differ");
            require(parallel.best_iteration == serial.best_iteration && parallel.best_loss == serial.best_loss,
                    "serial and parallel best selections differ");
            const auto best_row = std::min_element(parallel.history.begin(), parallel.history.end(),
                [](const auto& a, const auto& b) { return a.loss < b.loss; });
            require(parallel.best_iteration == best_row->iteration && parallel.best_loss == best_row->loss,
                    "selection did not minimize total loss including CEPT");
            const auto selected = verification.compute(parallel.local_masks, target, options.gradient_options);
            require(std::abs(selected.loss - parallel.best_loss) < 1e-10, "returned local state does not reproduce best loss");
            require(parallel.global_mask.isApprox(Partition::assemble_global(parallel.local_masks, meta), 1e-12),
                    "returned global mask does not match best local Cores");
            require(parallel.global_mask.minCoeff() >= 0.0 && parallel.global_mask.maxCoeff() <= 1.0, "mask is out of range");
            for (int k = 0; k <= options.iterations; ++k) {
                require(std::abs(parallel.history[k].loss - serial.history[k].loss) < 1e-11, "serial and parallel histories differ");
            }
            for (int id = 0; id < static_cast<int>(meta.tiles.size()); ++id) {
                require(parallel.local_theta[id].isApprox(serial.local_theta[id], 1e-12), "serial and parallel local theta differ");
            }
        }
    }
    options.iterations = 0;
    const auto initialized = optimizer.optimize(target, options);
    require(initialized.history.size() == 1 && initialized.history.front().iteration == 0, "zero updates did not retain initial state");
    require(initialized.best_iteration == 0 && initialized.best_loss == initialized.history.front().loss,
            "initial state was excluded from best selection");
}

} // namespace

int main() {
    try {
#ifdef _OPENMP
        omp_set_num_threads(4);
#endif
        for (const bool socs : {false, true}) {
            check_single_micro(socs);
            check_single_micro(socs, 100.0, 8, true);
            check_guard_only_update(socs);
            check_parallel_workflow(socs);
            std::cout << (socs ? "SOCS" : "Abbe") << ": fixed-step CTM, Guard-only CEPT, frozen Halo and parallel workflow passed\n";
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
