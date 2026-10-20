#include "micro_gradient_exchange.h"
#include "epe_utils.h"
#include "lithography_simulator.h"
#include "loss.h"
#include "level_set_utils.h"

#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

// 小尺寸平滑光学核用于独立的有限差分检查，频域 DC 位于角点，符合 Imaging 的缓存约定。
litho::ImagingCache make_cache(int n, bool socs) {
    litho::ImagingCache cache{};
    cache.N = n;
    cache.source_ws = Eigen::VectorXd::Constant(2, 0.5);
    if (socs) cache.socs_eigenvalues = {0.7, 0.3};
    for (int k = 0; k < 2; ++k) {
        Eigen::MatrixXcd kernel(n, n);
        for (int y = 0; y < n; ++y) {
            for (int x = 0; x < n; ++x) {
                const double fy = static_cast<double>(y <= n / 2 ? y : y - n) / n;
                const double fx = static_cast<double>(x <= n / 2 ? x : x - n) / n;
                const double amplitude = std::exp(-(12.0 + 8.0 * k) * (fx * fx + fy * fy));
                kernel(y, x) = std::polar(amplitude, k * (0.35 * fx - 0.2 * fy));
            }
        }
        cache.optical_kernels_frequency.push_back(std::move(kernel));
    }
    return cache;
}

// 用冻结基准光强斜率的平方 EPE 做 mask 有限差分；
// EP 的 AI 取样和误差信号散射必须使用同一组双线性权重。
// 覆盖 Abbe/SOCS、非单位光源权重和、亚像素/重复 EP、零/非整数 EP 权重及 EP 之外的 mask 像素。
void check_squared_epe_gradient(bool socs) {
    using namespace litho;
    constexpr int n = 12;
    constexpr double threshold = 0.18, alpha = 8.0, step = 1e-6;
    auto cache = make_cache(n, socs);
    cache.source_ws << 0.3, 0.9;
    Imaging imaging(cache);
    Gradient gradient(cache, threshold, alpha);
    Eigen::MatrixXd mask(n, n), direction(n, n);
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            mask(y, x) = 0.43 + 0.12 * std::sin(0.31 * x + 0.27 * y) + 0.05 * std::cos(0.52 * x - 0.23 * y);
            direction(y, x) = std::sin(0.71 * x - 0.43 * y) + 0.4 * std::cos(0.37 * x + 0.61 * y);
        }
    }
    direction /= direction.norm();
    EpsResult eps;
    eps.eps.resize(6, 2);
    eps.eps << 0.25, 1.5, 3.2, 5.7, 7.4, 9.1, 10.5, 10.25, 3.2, 5.7, 8.75, 2.4;
    eps.weight_epe.resize(6);
    eps.weight_epe << 1.0, 0.0, 0.4, 2.0, 0.7, 1.3;
    eps.weight_meef = Eigen::RowVectorXd::Ones(6);
    const auto baseline = imaging.compute(mask, threshold, alpha);
    const auto electric_field = imaging.get_electric_field(); // 后续扰动成像会覆盖内部缓存，保存基准电场副本。
    const Eigen::MatrixXd gx = LevelSetUtils::_gradient_1d(baseline.aerial_image, 1.0, 1);
    const Eigen::MatrixXd gy = LevelSetUtils::_gradient_1d(baseline.aerial_image, 1.0, 0);
    const Eigen::MatrixXd frozen_slope = (gx.array().square() + gy.array().square()).sqrt().max(1e-6).matrix();
    const auto analytic = gradient.epe_gradient(baseline.aerial_image, electric_field, eps, threshold);
    require(analytic.norm() > 1e-8, "squared EPE test gradient is zero");

    // pixel_size_nm 只换算绝对 EPE 统计，不能改变用于优化的 pixel^2 目标。
    const auto evaluated = Loss::evaluate_sparse_epe(baseline.aerial_image, eps, threshold, 8.0);
    const auto unit_scale = Loss::evaluate_sparse_epe(baseline.aerial_image, eps, threshold, 1.0);
    double expected_loss = 0.0, expected_abs_nm = 0.0;
    int expected_count = 0;
    for (int i = 0; i < eps.eps.rows(); ++i) {
        if (eps.weight_epe(i) <= 0.0) {
            require(std::isnan(evaluated.signed_epe_pixel(i)) && evaluated.point_losses(i) == 0.0,
                    "inactive EP entered sparse EPE statistics");
            continue;
        }
        const auto stencil = make_bilinear_stencil(n, n, eps.eps(i, 0), eps.eps(i, 1));
        const double sample = bilinear_sample(baseline.aerial_image, stencil);
        const double magnitude = std::max(bilinear_sample(frozen_slope, stencil), 1e-6);
        const double signed_epe = (sample - threshold) / magnitude;
        expected_loss += eps.weight_epe(i) * signed_epe * signed_epe;
        expected_abs_nm += std::abs(signed_epe) * 8.0;
        ++expected_count;
        require(std::abs(evaluated.aerial_samples(i) - sample) < 1e-14 &&
                std::abs(evaluated.gradient_samples(i) - magnitude) < 1e-14,
                "sparse EPE did not bilinearly sample AI and gradient magnitude");
    }
    require(std::abs(evaluated.weighted_squared_loss - expected_loss) < 1e-12 &&
            std::abs(evaluated.absolute_sum_nm - expected_abs_nm) < 1e-12 && evaluated.active_count == expected_count,
            "sparse EPE loss or absolute-nm statistics are wrong");
    require(evaluated.weighted_squared_loss == unit_scale.weighted_squared_loss &&
            std::abs(evaluated.absolute_sum_nm - 8.0 * unit_scale.absolute_sum_nm) < 1e-12,
            "pixel size changed the pixel-squared objective");

    Eigen::MatrixXd pixel_direction = Eigen::MatrixXd::Zero(n, n);
    pixel_direction(6, 6) = 1.0; // 此 mask 像素不是 EP，但仍能经光学系统改变 EP 的光强。
    auto frozen_loss = [&](const Eigen::MatrixXd& trial) {
        const auto image = imaging.compute(trial, threshold, alpha);
        double loss = 0.0;
        for (int i = 0; i < eps.eps.rows(); ++i) {
            if (eps.weight_epe(i) <= 0.0) continue;
            const auto stencil = make_bilinear_stencil(n, n, eps.eps(i, 0), eps.eps(i, 1));
            const double sample = bilinear_sample(image.aerial_image, stencil);
            const double magnitude = std::max(bilinear_sample(frozen_slope, stencil), 1e-6);
            const double epe = (sample - threshold) / magnitude;
            loss += eps.weight_epe(i) * epe * epe;
        }
        return loss;
    };
    const Eigen::MatrixXd aligned = analytic / analytic.norm();
    for (const auto& test_direction : std::vector<Eigen::MatrixXd>{direction, aligned, pixel_direction}) {
        const double numeric =
            (frozen_loss(mask + step * test_direction) - frozen_loss(mask - step * test_direction)) / (2.0 * step);
        const double expected = (analytic.array() * test_direction.array()).sum();
        require(std::abs(numeric - expected) < 5e-6 * std::max(1.0, std::abs(expected)),
                "bilinear frozen-denominator EPE finite difference failed");
    }
    std::cout << (socs ? "SOCS" : "Abbe") << ": subpixel bilinear EPE statistics and frozen-denominator finite differences passed\n";
}

void check_exchange(const litho::PartitionMeta& meta, const litho::MicroGradientExchangeResult& result) {
    require(result.locals.size() == meta.tiles.size(), "Micro count mismatch");
    double loss = 0.0;
    for (const litho::MicroTile& tile : meta.tiles) {
        const auto& local = result.locals[tile.id];
        require(local.raw_mask_gradient.rows() == meta.tile_size && local.raw_mask_gradient.cols() == meta.tile_size,
                "raw gradient size mismatch");
        require(local.raw_mask_gradient.allFinite(), "nonfinite raw gradient");
        require(local.received_neighbors.size() == tile.neighbors.size(), "neighbor count mismatch");
        loss += local.loss;
        for (int j = 0; j < static_cast<int>(tile.neighbors.size()); ++j) {
            const auto& edge = tile.neighbors[j];
            const auto& received = local.received_neighbors[j];
            const auto& gradient = result.locals[edge.id].raw_mask_gradient;
            require(received.neighbor_id == edge.id && received.direction == edge.direction, "neighbor identity mismatch");
            const auto& guard = edge.neighbor_overlap_local;
            const auto& overlap = edge.neighbor_guard_local;
            require(received.guard_gradient.rows() == edge.my_guard_local.rows() &&
                    received.guard_gradient.cols() == edge.my_guard_local.cols(), "Guard slice dimensions mismatch");
            require(received.overlap_gradient.rows() == edge.my_overlap_local.rows() &&
                    received.overlap_gradient.cols() == edge.my_overlap_local.cols(), "Core slice dimensions mismatch");
            require((received.guard_gradient - gradient.block(guard.y0, guard.x0, guard.rows(), guard.cols())).squaredNorm() == 0.0,
                    "Guard received wrong neighbor data");
            const Eigen::MatrixXd overlap_difference = received.overlap_gradient -
                gradient.block(overlap.y0, overlap.x0, overlap.rows(), overlap.cols());
            require(overlap_difference.squaredNorm() == 0.0, "Core received wrong neighbor data");
        }
    }
    require(loss == result.loss, "total loss mismatch");
}

void check_case(const litho::PartitionMeta& meta, bool socs) {
    const auto cache = make_cache(meta.tile_size, socs);
    Eigen::MatrixXd mask(meta.height, meta.width), target(meta.height, meta.width);
    for (int y = 0; y < meta.height; ++y) {
        for (int x = 0; x < meta.width; ++x) {
            mask(y, x) = 0.45 + 0.15 * std::sin(0.31 * x + 0.17 * y);
            target(y, x) = ((x / 4 + y / 3) % 3 == 0) ? 1.0 : 0.0;
        }
    }
    const Eigen::MatrixXd original_mask = mask, original_target = target;
    constexpr double threshold = 0.25, alpha = 8.0;
#ifdef _OPENMP
    // 先创建最多四个工作对象，再切换运行时线程数，确保多块案例实际覆盖 Micro 并行路径。
    omp_set_num_threads(4);
#endif
    litho::MicroGradientExchange exchange(meta, cache, threshold, alpha);
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    const auto serial = exchange.compute(mask, target, false);
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    const auto parallel = exchange.compute(mask, target, true);
    check_exchange(meta, serial);
    check_exchange(meta, parallel);
    require((mask - original_mask).squaredNorm() == 0.0 && (target - original_target).squaredNorm() == 0.0, "inputs changed");
    require(serial.loss == parallel.loss, "serial and Micro-parallel losses differ");
    for (int id = 0; id < static_cast<int>(meta.tiles.size()); ++id) {
        require((serial.locals[id].raw_mask_gradient - parallel.locals[id].raw_mask_gradient).squaredNorm() == 0.0,
                "serial and Micro-parallel gradients differ");
    }

    // 有限差分从实际成像的 Core loss 出发，不通过待测梯度公式计算数值导数。
    const auto& tile = meta.tiles.front();
    const auto& core = tile.core_local;
    const Eigen::MatrixXd local_mask = litho::Partition::extract_local_tile(mask, tile);
    const Eigen::MatrixXd local_target = litho::Partition::extract_local_tile(target, tile);
    const auto& gradient = parallel.locals.front().raw_mask_gradient;
    const Eigen::MatrixXd direction = gradient / gradient.norm();
    litho::Imaging imaging(cache);
    auto loss_at = [&](const Eigen::MatrixXd& trial_mask) {
        const auto image = imaging.compute(trial_mask, threshold, alpha);
        return (image.wafer_image.block(core.y0, core.x0, core.rows(), core.cols()) -
                local_target.block(core.y0, core.x0, core.rows(), core.cols())).squaredNorm();
    };
    require(std::abs(loss_at(local_mask) - parallel.locals.front().loss) < 1e-12, "loss is not restricted to Core");
    constexpr double step = 1e-5;
    const double numeric = (loss_at(local_mask + step * direction) - loss_at(local_mask - step * direction)) / (2.0 * step);
    require(std::abs(numeric - gradient.norm()) < 1e-6 * std::max(1.0, gradient.norm()), "Core PE gradient finite difference failed");

    // Core 误差会影响相邻 Halo 像素；确认返回的梯度没有被裁成 Core。
    if (tile.halo > 0) {
        Eigen::MatrixXd pixel = Eigen::MatrixXd::Zero(meta.tile_size, meta.tile_size);
        const int y = core.y0 - 1, x = core.x0 + 1;
        pixel(y, x) = 1.0;
        const double halo_numeric = (loss_at(local_mask + step * pixel) - loss_at(local_mask - step * pixel)) / (2.0 * step);
        require(std::abs(gradient(y, x)) > 1e-8, "Halo gradient was removed");
        require(std::abs(halo_numeric - gradient(y, x)) < 1e-6 * std::max(1.0, std::abs(gradient(y, x))),
                "Halo pixel gradient finite difference failed");
    }

    // 全 1 窗口须与原有无窗口 PE 梯度一致，保持原接口行为。
    const auto image = imaging.compute(local_mask, threshold, alpha);
    litho::Gradient full_gradient(cache, threshold, alpha);
    const auto plain = full_gradient.pe_gradient(image.wafer_image, local_target, imaging.get_electric_field());
    const auto weighted = full_gradient.pe_gradient(image.wafer_image, local_target, imaging.get_electric_field(),
                                                   Eigen::MatrixXd::Ones(meta.tile_size, meta.tile_size));
    require((plain - weighted).squaredNorm() == 0.0, "all-one window changed original PE gradient");
    std::cout << (socs ? "SOCS" : "Abbe") << " " << meta.grid_rows << 'x' << meta.grid_cols
              << ": exchange and finite differences passed, guard=" << meta.guard << std::endl;
}

// 总 loss 的有限差分同时扰动多个独立局部 mask，覆盖 CEPT 约束的 Guard 和邻居 Core 两端。
void check_joint_derivative(litho::MicroGradientExchange& exchange, const std::vector<Eigen::MatrixXd>& masks,
                            const Eigen::MatrixXd& target, const litho::MicroGradientOptions& options) {
    const auto result = exchange.compute(masks, target, options);
    double norm_squared = 0.0;
    for (const auto& local : result.locals) norm_squared += local.mask_gradient.squaredNorm();
    const double norm = std::sqrt(norm_squared);
    require(norm > 1e-8, "joint gradient is zero");
    auto plus = masks, minus = masks;
    constexpr double step = 1e-5;
    for (int id = 0; id < static_cast<int>(masks.size()); ++id) {
        const Eigen::MatrixXd delta = step * result.locals[id].mask_gradient / norm;
        plus[id] += delta;
        minus[id] -= delta;
    }
    const double numeric = (exchange.compute(plus, target, options).loss - exchange.compute(minus, target, options).loss) / (2.0 * step);
    require(std::abs(numeric - norm) < 2e-5 * std::max(1.0, norm), "selected loss / CEPT joint derivative failed");
}

void check_options(bool socs) {
    using namespace litho;
    const auto meta = Partition::build(24, 24, 2, 2, 4, 3);
    const auto cache = make_cache(meta.tile_size, socs);
    Eigen::MatrixXd mask(24, 24), target = Eigen::MatrixXd::Zero(24, 24);
    target.block(6, 6, 12, 12).setOnes();
    for (int y = 0; y < 24; ++y) {
        for (int x = 0; x < 24; ++x) mask(y, x) = 0.35 + 0.1 * std::sin(0.29 * x) + 0.08 * std::cos(0.19 * y);
    }
    auto masks = Partition::extract_all(mask, meta);
    for (int id = 0; id < static_cast<int>(masks.size()); ++id) masks[id].array() += 0.025 * (id + 1);
    MicroGradientExchange exchange(meta, cache, 0.25, 8.0);
    MicroGradientOptions options;
    options.enable_cept = true;
    options.cept_weight = 0.7;
    options.epe_interval_line = 3;
    options.epe_interval_corner = 1;
    options.pe_weight = 0.6;
    options.epe_weight = 0.2;
    options.epe_pixel_size = 2.0;

    const auto consistent = exchange.compute(mask, target, options);
    require(consistent.cept_loss == 0.0, "globally consistent masks have nonzero CEPT");
    for (const auto& local : consistent.locals) require(local.cept_mask_gradient.squaredNorm() == 0.0, "consistent CEPT gradient nonzero");

    const auto global_eps = select_micro_epe_points(target, options);
    Imaging direct_imaging(cache);
    Gradient direct_gradient(cache, 0.25, 8.0);
    for (MicroLossType type : {MicroLossType::PE, MicroLossType::PEPenalty, MicroLossType::EPE, MicroLossType::PE_EPE}) {
        options.loss_type = type;
        const auto result = exchange.compute(masks, target, options);
        require(result.cept_loss > 0.0, "independent masks did not activate CEPT");
        if (type == MicroLossType::PEPenalty) require(result.locals.front().bg_penalty_raw > 0.0, "background penalty did not activate");
        if (type == MicroLossType::EPE) require(result.data_loss > 0.0, "EPE selected no active points");
        for (const auto& local : result.locals) {
            const double expected = type == MicroLossType::EPE ? options.epe_weight * local.epe_loss :
                type == MicroLossType::PE_EPE ? options.pe_weight * local.pe_loss + options.epe_weight * local.epe_loss :
                options.pe_weight * local.pe_loss + local.penalty_loss;
            require(std::abs(local.data_loss - expected) < 1e-12, "loss selector used wrong coefficients");
        }
        if (type == MicroLossType::PE || type == MicroLossType::PEPenalty) {
            check_joint_derivative(exchange, masks, target, options);
        } else {
            // 按生产路径的浮点 Core 归属和双线性窗口筛选，独立复算平方 EPE 目标及其冻结分母梯度。
            std::vector<int> ownership(global_eps.eps.rows(), 0);
            for (const MicroTile& tile : meta.tiles) {
                EpsResult eps = global_eps;
                int count = 0;
                const auto goal = Partition::extract_local_tile(target, tile);
                const auto windows = build_micro_loss_windows(goal, tile.core_local, options);
                for (int i = 0; i < global_eps.eps.rows(); ++i) {
                    const double y = global_eps.eps(i, 0), x = global_eps.eps(i, 1);
                    const auto& core = tile.core_global;
                    if (y < core.y0 || y >= core.y1 || x < core.x0 || x >= core.x1) continue;
                    const double local_y = y - tile.tile_global.y0, local_x = x - tile.tile_global.x0;
                    const auto stencil = make_bilinear_stencil(meta.tile_size, meta.tile_size, local_y, local_x);
                    if (!bilinear_touches_window(windows.epe, stencil)) continue;
                    ++ownership[i];
                    eps.eps(count, 0) = local_y;
                    eps.eps(count, 1) = local_x;
                    eps.weight_epe(count) = global_eps.weight_epe(i);
                    eps.weight_meef(count) = global_eps.weight_meef(i);
                    ++count;
                }
                eps.eps.conservativeResize(count, 2);
                eps.weight_epe.conservativeResize(count);
                eps.weight_meef.conservativeResize(count);
                const auto image = direct_imaging.compute(masks[tile.id], 0.25, 8.0);
                const auto direct_epe = Loss::evaluate_sparse_epe(image.aerial_image, eps, 0.25, options.epe_pixel_size);
                const auto direct = direct_gradient.epe_gradient(image.aerial_image, direct_imaging.get_electric_field(), eps, 0.25);
                const auto& local = result.locals[tile.id];
                require(local.epe_loss == direct_epe.weighted_squared_loss && local.epe_count == direct_epe.active_count &&
                        local.epe_abs_sum_nm == direct_epe.absolute_sum_nm,
                        "EPE adapter changed sparse objective or absolute-nm statistics");
                Eigen::MatrixXd expected = options.epe_weight * direct;
                if (type == MicroLossType::PE_EPE) {
                    expected += options.pe_weight * direct_gradient.pe_gradient(
                        image.wafer_image, goal, direct_imaging.get_electric_field(), windows.pe);
                }
                require((local.raw_mask_gradient - expected).squaredNorm() == 0.0, "EPE adapter changed bilinear sparse gradient");
            }
            for (int count : ownership) require(count == 1, "global EP was lost or counted in multiple Cores");
        }
    }

    // 单独检查 CEPT，避免数据项梯度过大掩盖漏算邻居 Core 端导数的问题。
    options.loss_type = MicroLossType::PE_EPE;
    options.pe_weight = 0.0;
    options.epe_weight = 0.0;
    check_joint_derivative(exchange, masks, target, options);

    options = MicroGradientOptions{};
    const auto original = exchange.compute(masks, target, options);
    options.enable_grad_fusion = true;
    options.fusion_weight_owner = 0.3;
    options.fusion_weight_neighbor = 0.7;
    for (GradientFusionMode mode : {GradientFusionMode::Average, GradientFusionMode::Linear}) {
        options.fusion_mode = mode;
        const auto serial = exchange.compute(masks, target, options, false);
        const auto parallel = exchange.compute(masks, target, options, true);
        check_exchange(meta, parallel);
        require(parallel.loss == original.loss, "fusion changed scalar objective");
        double change = 0.0;
        for (const MicroTile& tile : meta.tiles) {
            const auto& local = parallel.locals[tile.id];
            const auto& core = tile.core_local;
            require((local.raw_mask_gradient - original.locals[tile.id].raw_mask_gradient).squaredNorm() == 0.0,
                    "fusion changed raw gradient");
            require((local.mask_gradient - serial.locals[tile.id].mask_gradient).squaredNorm() == 0.0,
                    "serial and parallel final gradients differ");
            const Eigen::MatrixXd core_difference =
                local.mask_gradient.block(core.y0, core.x0, core.rows(), core.cols()) -
                local.raw_mask_gradient.block(core.y0, core.x0, core.rows(), core.cols());
            require(core_difference.squaredNorm() == 0.0, "Guard fusion changed Core");
            change += (local.mask_gradient - local.raw_mask_gradient).squaredNorm();
            for (int j = 0; j < static_cast<int>(tile.neighbors.size()); ++j) {
                const auto& edge = tile.neighbors[j];
                const auto& guard = edge.my_guard_local;
                for (int y = 0; y < guard.rows(); ++y) {
                    for (int x = 0; x < guard.cols(); ++x) {
                        // 由像素距 Core 的距离独立计算线性权重，检查四种方向。
                        int distance = 0;
                        switch (edge.direction) {
                            case NeighborDirection::Left: distance = core.x0 - (guard.x0 + x); break;
                            case NeighborDirection::Right: distance = guard.x0 + x - core.x1 + 1; break;
                            case NeighborDirection::Top: distance = core.y0 - (guard.y0 + y); break;
                            case NeighborDirection::Bottom: distance = guard.y0 + y - core.y1 + 1; break;
                        }
                        const double strength = mode == GradientFusionMode::Average ? 1.0 :
                                                static_cast<double>(distance - 1) / (tile.guard - 1);
                        const double neighbor_weight = options.fusion_weight_neighbor * strength;
                        const double raw = local.raw_mask_gradient(guard.y0 + y, guard.x0 + x);
                        const double received = local.received_neighbors[j].guard_gradient(y, x);
                        const double expected = (options.fusion_weight_owner * raw + neighbor_weight * received) /
                                                (options.fusion_weight_owner + neighbor_weight);
                        require(std::abs(local.mask_gradient(guard.y0 + y, guard.x0 + x) - expected) < 1e-12,
                                "fusion weight / direction mismatch");
                    }
                }
            }
        }
        require(change > 1e-8, "fusion did not change any Guard gradient");
    }

    options.enable_cept = true;
    const auto serial = exchange.compute(masks, target, options, false);
    const auto parallel = exchange.compute(masks, target, options, true);
    require(serial.loss == parallel.loss, "Micro-parallel CEPT loss differs");
    for (int id = 0; id < static_cast<int>(masks.size()); ++id) {
        require((serial.locals[id].mask_gradient - parallel.locals[id].mask_gradient).squaredNorm() == 0.0,
                "Micro-parallel CEPT/fusion gradient differs");
    }
    std::cout << (socs ? "SOCS" : "Abbe") << ": PE/CEPT derivatives, subpixel EPE and fusion switches passed" << std::endl;
}

// 可选真实输入检查：从 build-release 运行 test_micro_gradient_exchange --real。
void check_real_patterns() {
    using namespace litho;
    const auto params = SimulationParameters::from_yaml("config.yaml");
    const auto meta = Partition::build(1024, 1024, 8, 8, 64, 30);
    const Grid grid(meta.tile_size, params.system.pixel_size_nm, params.optics.na, params.system.wavelength_nm);
    const auto& coords = grid.grid_coords();
    const Pupil pupil(LithographySimulator::_make_pupil_params(params), coords.Fx_2d, coords.Fy_2d);
    const Source source(LithographySimulator::_make_source_params(params, coords.Fx_1d));
    const LithoPrepare prepare(grid, pupil, source, true);
    MicroGradientExchange exchange(meta, prepare.cache(), params.resist.threshold, params.resist.alpha);
    for (const char* name : {"metal_3_0_only_1024", "contact_1_0_only_1024"}) {
        const cv::Mat input = cv::imread(std::string("target_pattern/") + name + ".bmp", cv::IMREAD_GRAYSCALE);
        require(input.rows == 1024 && input.cols == 1024, "real pattern dimensions mismatch");
        Eigen::MatrixXd target(1024, 1024);
        for (int y = 0; y < 1024; ++y) {
            for (int x = 0; x < 1024; ++x) target(y, x) = input.at<unsigned char>(y, x) / 255.0;
        }
        const Eigen::MatrixXd mask = (0.2 + 0.6 * target.array()).matrix();
        auto local_masks = Partition::extract_all(mask, meta);
        for (int id = 0; id < static_cast<int>(local_masks.size()); ++id) local_masks[id].array() += 0.005 * (id % 4);
        MicroGradientOptions options;
        options.loss_type = MicroLossType::PE_EPE;
        options.pe_weight = 1.0;
        options.epe_weight = 0.01;
        options.epe_pixel_size = params.system.pixel_size_nm;
        options.enable_cept = true;
        options.enable_grad_fusion = true;
        // 两种路径各预热一次，排除首次电场缓存分配；耗时包含 EP 准备、逐块计算和串行交换。
        exchange.compute(local_masks, target, options, false);
        exchange.compute(local_masks, target, options, true);
        const auto serial_start = std::chrono::steady_clock::now();
        const auto serial = exchange.compute(local_masks, target, options, false);
        const auto parallel_start = std::chrono::steady_clock::now();
        const auto result = exchange.compute(local_masks, target, options, true);
        const auto parallel_end = std::chrono::steady_clock::now();
        const double serial_seconds = std::chrono::duration<double>(parallel_start - serial_start).count();
        const double parallel_seconds = std::chrono::duration<double>(parallel_end - parallel_start).count();
        require(std::abs(serial.loss - result.loss) < 1e-12 * std::max(1.0, std::abs(serial.loss)),
                "real serial and Micro-parallel losses differ");
        for (int id = 0; id < static_cast<int>(local_masks.size()); ++id) {
            require(serial.locals[id].mask_gradient.isApprox(result.locals[id].mask_gradient, 1e-12),
                    "real serial and Micro-parallel final gradients differ");
        }
        check_exchange(meta, result);
        require(std::isfinite(result.loss) && result.loss > 0.0, "real Core PE is invalid");
        require(result.cept_loss > 0.0, "real CEPT was not active");
        double epe_loss = 0.0, fusion_change = 0.0;
        for (const auto& local : result.locals) {
            require(local.mask_gradient.allFinite(), "real final gradient nonfinite");
            epe_loss += local.epe_loss;
            const Eigen::MatrixXd fusion_delta = local.mask_gradient - local.raw_mask_gradient -
                                                options.cept_weight * local.cept_mask_gradient;
            fusion_change += fusion_delta.squaredNorm();
        }
        require(epe_loss > 0.0 && fusion_change > 0.0, "real EPE or fusion was not active");
        std::cout << name << ": 64 Micro PE+EPE/CEPT/fusion passed, data loss=" << result.data_loss
                  << ", CEPT=" << result.cept_loss << ", total loss=" << result.loss
                  << ", serial=" << serial_seconds << " s, Micro-parallel=" << parallel_seconds
                  << " s, speedup=" << serial_seconds / parallel_seconds << std::endl;
    }
}

// Python 全芯片选点生成的工字形基准：核对候选数、激活数、浮点坐标、顺序和端点权重。
void check_python_subpixel_fixture() {
    using namespace litho;
    const cv::Mat input = cv::imread("target_pattern/工字型.bmp", cv::IMREAD_GRAYSCALE);
    require(input.rows == 251 && input.cols == 251, "Python EP fixture dimensions mismatch");
    Eigen::MatrixXd target(input.rows, input.cols);
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) target(y, x) = input.at<unsigned char>(y, x) / 255.0;
    }
    const auto selected = EpSelect(target, 5.0, 1.0).select_eps_subpixel(10.0, 2.0);
    require(selected.eps.rows() == 60, "Python EP fixture candidate count changed");
    require((selected.weight_epe.array() > 0.0).count() == 52, "Python EP fixture active count changed");
    require(selected.eps(0, 0) == 166.5 && selected.eps(0, 1) == 163.0 && selected.weight_epe(0) == 1.0 &&
            selected.weight_meef(0) == 5.0,
            "Python EP fixture first point changed");
    require(selected.eps(59, 0) == 164.0 && selected.eps(59, 1) == 165.5 && selected.weight_epe(59) == 0.0 &&
            selected.weight_meef(59) == 1.0,
            "Python EP fixture final point changed");
}

// 独立参考：重复 3×3 膨胀等价于检查 L∞ 距离不超过 radius 的主图像素，不调用 OpenCV 或生产窗口函数。
litho::MicroLossWindows reference_loss_windows(const Eigen::MatrixXd& target, const litho::PartitionRect& region, int radius,
                                              litho::MicroLossType type = litho::MicroLossType::PE_EPE) {
    litho::MicroLossWindows windows{Eigen::MatrixXd::Zero(target.rows(), target.cols()),
                                    Eigen::MatrixXd::Zero(target.rows(), target.cols()),
                                    Eigen::MatrixXd::Zero(target.rows(), target.cols())};
    for (int y = region.y0; y < region.y1; ++y) {
        for (int x = region.x0; x < region.x1; ++x) {
            bool main = false;
            for (int sy = std::max(0, y - radius); sy <= std::min(static_cast<int>(target.rows()) - 1, y + radius); ++sy) {
                for (int sx = std::max(0, x - radius); sx <= std::min(static_cast<int>(target.cols()) - 1, x + radius); ++sx) {
                    main |= target(sy, sx) > 0.5;
                }
            }
            windows.epe(y, x) = type == litho::MicroLossType::PE_EPE ? main : 1.0;
            windows.pe(y, x) = type == litho::MicroLossType::PE_EPE ? !main :
                               type == litho::MicroLossType::PEPenalty ? target(y, x) > 0.5 : 1.0;
            windows.penalty(y, x) = target(y, x) < 0.5;
        }
    }
    return windows;
}

void check_loss_windows() {
    using namespace litho;
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(12, 15);
    target(0, 0) = 1.0; // 局部数组边缘按 0 填充，不从对侧周期膨胀。
    target(2, 2) = 0.5; // threshold 为严格大于 0.5。
    target.block(4, 5, 4, 4).setOnes();
    target(5, 6) = 0.0; // 膨胀会填充小孔。
    target(1, 8) = 1.0; // region 外的主图也能膨胀进入评价区域，验证先膨胀再裁窗口。
    const PartitionRect region{2, 11, 3, 14};
    MicroGradientOptions options;
    options.loss_type = MicroLossType::PE_EPE;
    for (int radius : {0, 1, 2, 4}) {
        options.main_region_dilation_pixels = radius;
        const auto actual = build_micro_loss_windows(target, region, options);
        const auto expected = reference_loss_windows(target, region, radius);
        require(actual.pe == expected.pe && actual.epe == expected.epe, "dilated main/background window mismatch");
        require((actual.pe.array() * actual.epe.array()).sum() == 0.0, "main and background windows overlap");
        require(actual.pe.sum() + actual.epe.sum() == region.rows() * region.cols(), "windows miss evaluation pixels");
    }
    for (const auto type : {MicroLossType::PE, MicroLossType::PEPenalty, MicroLossType::EPE}) {
        options.loss_type = type;
        const auto actual = build_micro_loss_windows(target, region, options);
        const auto expected = reference_loss_windows(target, region, 0, type);
        require(actual.pe == expected.pe && actual.epe == expected.epe && actual.penalty == expected.penalty,
                "single loss type used a dilated/background-only PE window");
    }
}

// 独立从膨胀邻域、成像输出和周期坐标验证 LSM 分区；CEPT 以冻结邻居 Core 的本块损失做有限差分。
void check_lsm_regions(bool socs) {
    using namespace litho;
    const auto meta = Partition::build(32, 32, 2, 2, 8, 3);
    const auto cache = make_cache(meta.tile_size, socs);
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(32, 32), mask(32, 32);
    target.block(1, 1, 8, 8).setOnes(); // 靠近芯片边缘，周期 EP 会落入对侧 Micro 的 Guard。
    target.block(12, 13, 10, 8).setOnes(); // 跨 Core 边界，保证 Guard 内也存在 EPE 采样点。
    for (int y = 0; y < 32; ++y) {
        for (int x = 0; x < 32; ++x) mask(y, x) = 0.35 + 0.1 * std::sin(0.29 * x) + 0.08 * std::cos(0.19 * y);
    }
    auto masks = Partition::extract_all(mask, meta);
    MicroGradientExchange exchange(meta, cache, 0.25, 8.0);
    Imaging imaging(cache);
    Gradient gradient(cache, 0.25, 8.0);
    MicroGradientOptions options;
    options.loss_region = MicroLossRegion::CoreGuard;
    options.cept_guard_only = true;
    options.collect_core_metrics = true; // 与 LSM 一致：未启用的 EPE/背景项也应报告损失区域内的原始值。
    options.pe_weight = 0.3;
    options.epe_weight = 0.2;
    options.penalty_weight = 0.7;
    for (const auto type : {MicroLossType::PE, MicroLossType::PEPenalty}) {
        options.loss_type = type;
        const auto result = exchange.compute(masks, target, options);
        auto core_options = options;
        core_options.loss_region = MicroLossRegion::Core;
        require(result.data_loss > exchange.compute(masks, target, core_options).data_loss, "Guard contributes no data loss");
        for (const auto& tile : meta.tiles) {
            const auto image = imaging.compute(masks[tile.id], 0.25, 8.0);
            const auto local_target = Partition::extract_local_tile(target, tile);
            const auto& w = tile.update_local;
            const auto windows = reference_loss_windows(local_target, w, options.main_region_dilation_pixels, type);
            double pe = 0.0, bg = 0.0, core_pe = 0.0, core_bg = 0.0;
            const auto& c = tile.core_local;
            for (int y = w.y0; y < w.y1; ++y) {
                for (int x = w.x0; x < w.x1; ++x) {
                    const double error = windows.pe(y, x) * std::pow(image.wafer_image(y, x) - local_target(y, x), 2);
                    const double penalty = windows.penalty(y, x) *
                        std::pow(std::max(image.wafer_image(y, x) - options.penalty_threshold, 0.0), 2);
                    pe += error;
                    bg += penalty;
                    if (y >= c.y0 && y < c.y1 && x >= c.x0 && x < c.x1) {
                        core_pe += error;
                        core_bg += penalty;
                    }
                }
            }
            const auto& local = result.locals[tile.id];
            const double expected = options.pe_weight * pe + (type == MicroLossType::PEPenalty ? options.penalty_weight * bg : 0.0);
            require(std::abs(expected - local.data_loss) < 1e-10, "selected PE/PEPenalty windows or weights are wrong");
            require(std::abs(local.core.pe_loss - core_pe) < 1e-10 && std::abs(local.core.bg_penalty_raw - core_bg) < 1e-10,
                    "Core metrics use the wrong selected loss window");
            const auto expected_gradient = type == MicroLossType::PE ?
                (options.pe_weight * gradient.pe_gradient(
                    image.wafer_image, local_target, imaging.get_electric_field(), windows.pe)).eval() :
                gradient.pe_gradient_with_penalty(image.wafer_image, local_target, imaging.get_electric_field(),
                                                  options.penalty_weight, options.penalty_threshold, windows.pe,
                                                  options.pe_weight, windows.penalty);
            require((expected_gradient - local.raw_mask_gradient).norm() < 1e-10, "masked output adjoint mismatch");
            // 输出评价窗口之外的 mask 像素仍能影响窗口内成像；光学反传后不能再裁成 PE 窗口。
            require((local.raw_mask_gradient.array() * (1.0 - windows.pe.array())).matrix().norm() > 1e-6,
                    "optical gradient was cropped to the background window");
            require((local.raw_mask_gradient.array() * (1.0 - Partition::update_window(tile).array())).matrix().norm() > 1e-6,
                    "fixture must exercise nonzero gradient outside Core + Guard");
        }
        check_joint_derivative(exchange, masks, target, options);
    }

    // 独立枚举全局亚像素 EP 的相邻周期副本，并按双线性支撑像素与 Core + Guard 窗口的交集筛选。
    const auto global_eps = select_micro_epe_points(target, options);
    options.loss_type = MicroLossType::EPE;
    const auto epe = exchange.compute(masks, target, options);
    int guard_points = 0, periodic_points = 0;
    for (const auto& tile : meta.tiles) {
        std::vector<Eigen::Vector2d> positions;
        std::vector<int> indices;
        const auto& w = tile.update_local;
        const auto& core = tile.core_local;
        const auto goal = Partition::extract_local_tile(target, tile);
        const auto windows = reference_loss_windows(goal, w, options.main_region_dilation_pixels, MicroLossType::EPE);
        const PartitionRect global_region = w.shifted(tile.tile_global.y0, tile.tile_global.x0);
        for (int i = 0; i < global_eps.eps.rows(); ++i) {
            for (int py = -1; py <= 1; ++py) {
                for (int px = -1; px <= 1; ++px) {
                    const double global_y = global_eps.eps(i, 0) + py * meta.height;
                    const double global_x = global_eps.eps(i, 1) + px * meta.width;
                    if (global_y < global_region.y0 || global_y >= global_region.y1 ||
                        global_x < global_region.x0 || global_x >= global_region.x1) continue;
                    const double local_y = global_y - tile.tile_global.y0, local_x = global_x - tile.tile_global.x0;
                    const auto stencil = make_bilinear_stencil(meta.tile_size, meta.tile_size, local_y, local_x);
                    if (!bilinear_touches_window(windows.epe, stencil)) continue;
                    positions.emplace_back(local_y, local_x);
                    indices.push_back(i);
                    if (global_eps.weight_epe(i) > 0.0) {
                        guard_points += local_y < core.y0 || local_y >= core.y1 || local_x < core.x0 || local_x >= core.x1;
                        periodic_points += py != 0 || px != 0;
                    }
                }
            }
        }
        const int count = static_cast<int>(positions.size());
        EpsResult eps{Eigen::MatrixXd(count, 2), Eigen::RowVectorXd(count), Eigen::RowVectorXd(count)};
        for (int j = 0; j < count; ++j) {
            eps.eps.row(j) = positions[j].transpose();
            eps.weight_epe(j) = global_eps.weight_epe(indices[j]);
            eps.weight_meef(j) = global_eps.weight_meef(indices[j]);
        }
        const auto image = imaging.compute(masks[tile.id], 0.25, 8.0);
        const auto expected = Loss::evaluate_sparse_epe(image.aerial_image, eps, 0.25, options.epe_pixel_size);
        const Eigen::MatrixXd expected_gradient = options.epe_weight * gradient.epe_gradient(
            image.aerial_image, imaging.get_electric_field(), eps, 0.25);
        require(std::abs(expected.weighted_squared_loss - epe.locals[tile.id].epe_loss) <
                    1e-11 * std::max(1.0, expected.weighted_squared_loss) &&
                expected.absolute_sum_nm == epe.locals[tile.id].epe_abs_sum_nm,
                "wrong Core + Guard sparse EPE objective or absolute-nm sum");
        require(epe.locals[tile.id].epe_count == expected.active_count,
                "EPE point count includes inactive points or excludes Guard/periodic active points");
        require((expected_gradient - epe.locals[tile.id].raw_mask_gradient).norm() < 1e-11 * std::max(1.0, expected_gradient.norm()),
                "wrong Core + Guard EPE gradient");
    }
    require(guard_points > 0 && periodic_points > 0, "EPE fixture missed Guard or periodic points");

    // 只优化 PE 时，EPE 与背景仍报告原始总值；统计不能额外加入它们的 loss 或梯度。
    options.loss_type = MicroLossType::PE;
    const auto pe_only = exchange.compute(masks, target, options);
    auto plain_options = options;
    plain_options.collect_core_metrics = false;
    const auto plain = exchange.compute(masks, target, plain_options);
    require(pe_only.loss == plain.loss, "reporting inactive terms changed PE objective");
    for (const auto& tile : meta.tiles) {
        const auto& local = pe_only.locals[tile.id];
        require(local.epe_count == epe.locals[tile.id].epe_count && local.epe_loss == epe.locals[tile.id].epe_loss,
                "inactive EPE statistics differ from the same-region EPE evaluation");
        require(local.raw_mask_gradient == plain.locals[tile.id].raw_mask_gradient, "reporting inactive terms changed PE gradient");
        const auto image = imaging.compute(masks[tile.id], 0.25, 8.0);
        const auto goal = Partition::extract_local_tile(target, tile);
        const auto& w = tile.update_local;
        const auto windows = reference_loss_windows(goal, w, options.main_region_dilation_pixels, MicroLossType::PE);
        double background = 0.0;
        for (int y = w.y0; y < w.y1; ++y) {
            for (int x = w.x0; x < w.x1; ++x) {
                if (windows.pe(y, x) == 0.0 || goal(y, x) >= 0.5) continue;
                const double excess = std::max(image.wafer_image(y, x) - options.penalty_threshold, 0.0);
                background += excess * excess;
            }
        }
        require(std::abs(local.bg_penalty_raw - background) < 1e-10, "inactive background metric uses the wrong region");
    }

    options.loss_type = MicroLossType::PE_EPE;
    options.pe_weight = options.epe_weight = 0.0; // 隔离 CEPT，验证它只驱动本块 Guard。
    options.enable_cept = true;
    for (int id = 0; id < masks.size(); ++id) masks[id].array() += 0.025 * id;
    const auto cept = exchange.compute(masks, target, options);
    for (const auto& tile : meta.tiles) {
        Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(meta.tile_size, meta.tile_size);
        double expected_loss = 0.0;
        for (const auto& edge : tile.neighbors) {
            const auto& g = edge.my_guard_local;
            const auto& b = edge.neighbor_overlap_local;
            const Eigen::MatrixXd diff = masks[tile.id].block(g.y0, g.x0, g.rows(), g.cols()) -
                masks[edge.id].block(b.y0, b.x0, b.rows(), b.cols());
            expected.block(g.y0, g.x0, g.rows(), g.cols()) += 2.0 * diff;
            expected_loss += diff.squaredNorm();
        }
        const auto& local = cept.locals[tile.id];
        require((local.cept_mask_gradient - expected).norm() < 1e-12, "CEPT changed Core, corners or outer Halo");
        require(std::abs(local.cept_loss - expected_loss) < 1e-12, "CEPT includes a reversed Core-to-Guard loss");
        auto plus = masks, minus = masks;
        const Eigen::MatrixXd direction = expected / expected.norm();
        constexpr double step = 1e-5;
        plus[tile.id] += step * direction;
        minus[tile.id] -= step * direction;
        const double numeric = (exchange.compute(plus, target, options).locals[tile.id].loss -
                                exchange.compute(minus, target, options).locals[tile.id].loss) / (2.0 * step);
        require(std::abs(numeric - options.cept_weight * expected.norm()) < 1e-8, "Guard-only CEPT local finite difference failed");
    }
    std::cout << (socs ? "SOCS" : "Abbe") << ": per-type PE/BG/EPE windows, full adjoint, periodic EP and Guard-only CEPT passed\n";
}

// 同一矩形跨越四个 Core，边上中点可落在分界线上；验证选点、唯一归属和不重叠 Core 的独立评价。
void check_core_metrics_and_sampling(bool socs) {
    using namespace litho;
    const auto meta = Partition::build(32, 32, 2, 2, 8, 3);
    const auto cache = make_cache(meta.tile_size, socs);
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(32, 32), mask(32, 32);
    target.block(8, 8, 16, 16).setOnes();
    for (int y = 0; y < 32; ++y) {
        for (int x = 0; x < 32; ++x) mask(y, x) = 0.4 + 0.1 * std::sin(0.29 * x) + 0.08 * std::cos(0.19 * y);
    }
    auto masks = Partition::extract_all(mask, meta);
    for (int id = 0; id < masks.size(); ++id) masks[id].array() += 0.02 * id;
    MicroGradientExchange exchange(meta, cache, 0.25, 8.0);
    Imaging imaging(cache);
    MicroGradientOptions options;
    options.loss_region = MicroLossRegion::CoreGuard;
    options.collect_core_metrics = true;
    options.epe_interval_line = 10; // 直线中点参与 EPE，靠近拐角的候选点保留为 inactive，覆盖 epe_all_eps。
    options.epe_interval_corner = 2;
    options.epe_pixel_size = 2.0;
    options.epe_sampling = MicroEpeSampling::Contact;
    const auto selected = select_micro_epe_points(target, options);
    const auto expected_selection = EpSelect(target, 5.0, 1.0).select_eps_subpixel(10.0, 2.0);
    require(selected.eps == expected_selection.eps && selected.weight_epe == expected_selection.weight_epe &&
            selected.weight_meef == expected_selection.weight_meef,
            "full-chip selector differs from the Python-style subpixel selector");
    for (const auto sampling : {MicroEpeSampling::Via, MicroEpeSampling::Contact, MicroEpeSampling::Metal}) {
        options.epe_sampling = sampling;
        const auto same = select_micro_epe_points(target, options);
        require(same.eps == selected.eps && same.weight_epe == selected.weight_epe,
                "layout label changed the unified full-chip EP selection");
    }
    check_loss_windows();

    auto all_options = options;
    all_options.epe_all_eps = true;
    all_options.loss_type = MicroLossType::EPE;
    const auto all_eps = select_micro_epe_points(target, all_options);
    require(all_eps.eps == selected.eps && all_eps.weight_meef == selected.weight_meef,
            "all-EP option changed point coordinates or MEEF weights");
    require(all_eps.weight_epe.minCoeff() == 1.0, "all-EP option left zero EPE weights");
    const int blue_count = static_cast<int>((selected.weight_epe.array() == 0.0).count());
    require(blue_count > 0, "fixture must contain inactive subpixel EPs");

    // 单块使全局/局部 EP 坐标相同；核对 epe_all_eps 新增点带来的 loss、统计量和 mask 梯度。
    const auto one_meta = Partition::build(32, 32, 1, 1, 0, 0);
    MicroGradientExchange one_exchange(one_meta, cache, 0.25, 8.0);
    auto active_options = all_options;
    active_options.epe_all_eps = false;
    const auto active_result = one_exchange.compute(mask, target, active_options, false);
    const auto all_result = one_exchange.compute(mask, target, all_options, false);
    const auto image = imaging.compute(mask, 0.25, 8.0);
    Gradient direct_gradient(cache, 0.25, 8.0);
    EpsResult blue_eps = all_eps;
    blue_eps.weight_epe = (selected.weight_epe.array() == 0.0).cast<double>().matrix();
    const auto blue_epe = Loss::evaluate_sparse_epe(image.aerial_image, blue_eps, 0.25, options.epe_pixel_size);
    const auto blue_gradient = direct_gradient.epe_gradient(
        image.aerial_image, imaging.get_electric_field(), blue_eps, 0.25);
    const Eigen::MatrixXd gradient_change = all_result.locals[0].raw_mask_gradient - active_result.locals[0].raw_mask_gradient;
    require(std::abs(all_result.loss - active_result.loss - blue_epe.weighted_squared_loss) < 1e-10,
            "all-EP mode added the wrong inactive-point loss");
    require(all_result.locals[0].core.ep_count - active_result.locals[0].core.ep_count == blue_count,
            "all-EP mode added the wrong active-point count");
    require((gradient_change - blue_gradient).norm() < 1e-10 * std::max(1.0, blue_gradient.norm()),
            "all-EP mode added the wrong inactive-point gradient");

    // 纯 PE 模式也必须统计 Core EPE；统计开关不能改变实际 loss 或梯度。
    options.epe_sampling = MicroEpeSampling::Contact;
    options.loss_type = MicroLossType::PE;
    const auto evaluation = exchange.compute(masks, target, options, false);
    auto no_metrics = options;
    no_metrics.collect_core_metrics = false;
    const auto plain = exchange.compute(masks, target, no_metrics, false);
    require(evaluation.loss == plain.loss, "Core diagnostics changed objective");
    std::vector<int> owners(selected.eps.rows(), 0);
    int active_core_count = 0;
    for (const auto& tile : meta.tiles) {
        const auto tile_image = imaging.compute(masks[tile.id], 0.25, 8.0);
        const auto& core = tile.core_global;
        double pe = 0.0, main_pe = 0.0, bg = 0.0;
        // 独立逐个全局 Core 像素取值，不复用 evaluator 的矩阵切片。
        for (int y = core.y0; y < core.y1; ++y) {
            for (int x = core.x0; x < core.x1; ++x) {
                const double wi = tile_image.wafer_image(y - tile.tile_global.y0, x - tile.tile_global.x0);
                const double error = std::pow(wi - target(y, x), 2);
                pe += error;
                if (target(y, x) > 0.5) main_pe += error;
                if (target(y, x) < 0.5) bg += std::pow(std::max(0.0, wi - options.penalty_threshold), 2);
            }
        }
        EpsResult eps = selected;
        int candidate_count = 0;
        for (int i = 0; i < selected.eps.rows(); ++i) {
            const double y = selected.eps(i, 0), x = selected.eps(i, 1);
            if (y < core.y0 || y >= core.y1 || x < core.x0 || x >= core.x1) continue;
            eps.eps.row(candidate_count) << y - tile.tile_global.y0, x - tile.tile_global.x0;
            eps.weight_epe(candidate_count) = selected.weight_epe(i);
            eps.weight_meef(candidate_count++) = selected.weight_meef(i);
            ++owners[i];
        }
        eps.eps.conservativeResize(candidate_count, 2);
        eps.weight_epe.conservativeResize(candidate_count);
        eps.weight_meef.conservativeResize(candidate_count);
        const auto epe = Loss::evaluate_sparse_epe(tile_image.aerial_image, eps, 0.25, options.epe_pixel_size);
        const auto& local = evaluation.locals[tile.id];
        require(std::abs(local.core.pe_loss - pe) < 1e-10 && std::abs(local.core.bg_penalty_raw - bg) < 1e-10,
                "Core pixel diagnostics include Guard or use wrong coefficients");
        require(std::abs(local.core.pe_penalty_loss - options.pe_weight * main_pe - options.penalty_weight * bg) < 1e-10,
                "Core PEPenalty includes ordinary background PE");
        require(local.core.ep_count == epe.active_count && std::abs(local.core.epe_loss - epe.weighted_squared_loss) < 1e-10 &&
                std::abs(local.core.epe_abs_sum_nm - epe.absolute_sum_nm) < 1e-10,
                "wrong Core sparse EPE ownership or value");
        const double expected_mean = epe.active_count == 0 ? 0.0 : epe.absolute_sum_nm / epe.active_count;
        require(local.core.epe_mean == expected_mean, "wrong Core absolute EPE mean");
        require(local.core.data_loss == local.core.pe_loss, "pure PE Core metric contains inactive penalties");
        require(local.raw_mask_gradient == plain.locals[tile.id].raw_mask_gradient, "Core diagnostics changed gradient");
        active_core_count += epe.active_count;
    }
    const int selected_active_count = static_cast<int>((selected.weight_epe.array() > 0.0).count());
    require(active_core_count == selected_active_count, "Core active EP count differs from selected global active EP count");
    for (int count : owners) require(count == 1, "boundary EP must belong to exactly one Core");

    // 先以非零背景惩罚验证 pe_weight=0/0.3 时的独立系数；零 PE 权重不能屏蔽背景驱动。
    options.loss_type = MicroLossType::PEPenalty;
    options.epe_sampling = MicroEpeSampling::Contact;
    options.penalty_weight = 0.7;
    options.epe_weight = 0.0;
    for (double pe_weight : {0.0, 0.3}) {
        options.pe_weight = pe_weight;
        check_joint_derivative(exchange, masks, target, options);
    }
    // 混合模式只含背景 PE 和主图 EPE；penalty_weight 非零也不能带入背景惩罚项。
    options.loss_type = MicroLossType::PE_EPE;
    check_joint_derivative(exchange, masks, target, options);
    const auto pixel = exchange.compute(masks, target, options, false);
    options.epe_weight = 0.2;
    const auto mixed = exchange.compute(masks, target, options, false);
    auto epe_options = options;
    epe_options.loss_type = MicroLossType::EPE;
    const auto epe = exchange.compute(masks, target, epe_options, false);
    for (const auto& tile : meta.tiles) {
        const auto& m = mixed.locals[tile.id];
        const auto& p = pixel.locals[tile.id];
        require(std::abs(m.data_loss - p.data_loss - epe.locals[tile.id].data_loss) < 1e-10,
                "mixed loss added background penalty or ignored EPE weight");
        const Eigen::MatrixXd expected = p.raw_mask_gradient + epe.locals[tile.id].raw_mask_gradient;
        require((m.raw_mask_gradient - expected).norm() < 1e-10, "mixed mode changed original EPE gradient");
        const auto& c = m.core;
        require(m.penalty_loss == 0.0, "PE_EPE must not contain a penalty term");
        require(std::abs(c.data_loss - options.pe_weight * c.pe_loss -
                         options.epe_weight * c.epe_loss) < 1e-10, "wrong weighted Core mixed objective");
    }
    std::cout << (socs ? "SOCS" : "Abbe")
              << ": unified subpixel EP, unique Core diagnostics and two-term mixture passed\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--real") {
            check_real_patterns();
        } else {
            check_python_subpixel_fixture();
            for (bool socs : {false, true}) {
                check_squared_epe_gradient(socs);
                check_case(litho::Partition::build(24, 28, 2, 3, 5, 2), socs);
                check_case(litho::Partition::build(25, 31, 3, 3, 5, 2), socs);
                check_case(litho::Partition::build(16, 16, 2, 2, 0, 0), socs);
                check_case(litho::Partition::build(8, 10, 1, 1, 12, 4), socs);
                check_options(socs);
                check_lsm_regions(socs);
                check_core_metrics_and_sampling(socs);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
