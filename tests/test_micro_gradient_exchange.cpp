#include "micro_gradient_exchange.h"
#include "lithography_simulator.h"
#include "loss.h"

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

    double plain_pe = 0.0;
    const auto pure_pe = exchange.compute(masks, target);
    const EpSelect selector(target, 1.0, 1.0);
    const auto global_eps = selector.select_eps_via(2); // 与当前生产代码的 EP 选择一致，不改变原 C++ 选点规则。
    Imaging direct_imaging(cache);
    Gradient direct_gradient(cache, 0.25, 8.0);
    for (MicroLossType type : {MicroLossType::PE, MicroLossType::PEPenalty, MicroLossType::EPE, MicroLossType::PE_EPE}) {
        options.loss_type = type;
        const auto result = exchange.compute(masks, target, options);
        require(result.cept_loss > 0.0, "independent masks did not activate CEPT");
        if (type == MicroLossType::PE) plain_pe = result.data_loss;
        if (type == MicroLossType::PEPenalty) require(result.data_loss > plain_pe, "background penalty did not activate");
        if (type == MicroLossType::EPE) require(result.data_loss > 0.0, "EPE selected no active points");
        for (const auto& local : result.locals) {
            const double expected = type == MicroLossType::EPE ? local.epe_loss : type == MicroLossType::PE_EPE ?
                options.pe_weight * local.pe_loss + options.epe_weight * local.epe_loss : local.pe_loss;
            require(std::abs(local.data_loss - expected) < 1e-12, "loss selector used wrong coefficients");
        }
        if (type == MicroLossType::PE || type == MicroLossType::PEPenalty) {
            check_joint_derivative(exchange, masks, target, options);
        } else {
            // EPE 按原 C++ loss/梯度接口验证复用一致性，不把原近似梯度当作 loss 的严格导数。
            std::vector<int> ownership(global_eps.eps.rows(), 0);
            for (const MicroTile& tile : meta.tiles) {
                EpsResult eps = global_eps;
                int count = 0;
                for (int i = 0; i < global_eps.eps.rows(); ++i) {
                    const int y = static_cast<int>(global_eps.eps(i, 0)), x = static_cast<int>(global_eps.eps(i, 1));
                    const auto& core = tile.core_global;
                    if (y < core.y0 || y >= core.y1 || x < core.x0 || x >= core.x1) continue;
                    ++ownership[i];
                    eps.eps(count, 0) = y - tile.tile_global.y0;
                    eps.eps(count, 1) = x - tile.tile_global.x0;
                    eps.weight_epe(count) = global_eps.weight_epe(i);
                    eps.weight_meef(count) = global_eps.weight_meef(i);
                    ++count;
                }
                eps.eps.conservativeResize(count, 2);
                eps.weight_epe.conservativeResize(count);
                eps.weight_meef.conservativeResize(count);
                const auto image = direct_imaging.compute(masks[tile.id], 0.25, 8.0);
                const double direct_loss = Loss::evaluate_epe(image.aerial_image, eps.eps, 0.25, options.epe_pixel_size).total_epe;
                const auto direct = direct_gradient.epe_gradient(
                    image.aerial_image, direct_imaging.get_electric_field(), eps, 0.25, options.epe_pixel_size);
                const auto& local = result.locals[tile.id];
                require(local.epe_loss == direct_loss, "EPE adapter changed existing C++ loss");
                const Eigen::MatrixXd expected = type == MicroLossType::EPE ? direct :
                    (options.pe_weight * pure_pe.locals[tile.id].raw_mask_gradient + options.epe_weight * direct).eval();
                require((local.raw_mask_gradient - expected).squaredNorm() == 0.0, "EPE adapter changed existing C++ gradient");
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
    std::cout << (socs ? "SOCS" : "Abbe") << ": PE/CEPT derivatives, existing C++ EPE reuse and fusion switches passed" << std::endl;
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

// 独立从成像输出和周期像素坐标验证 LSM 的损失区域；CEPT 以冻结邻居 Core 的本块损失做有限差分。
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
            const Eigen::MatrixXd wafer = image.wafer_image.block(w.y0, w.x0, w.rows(), w.cols());
            const Eigen::MatrixXd goal = local_target.block(w.y0, w.x0, w.rows(), w.cols());
            const double expected = type == MicroLossType::PE ? Loss::pe_loss(wafer, goal) :
                Loss::pe_loss_with_penalty(wafer, goal, options.penalty_weight, options.penalty_threshold);
            require(std::abs(expected - result.locals[tile.id].data_loss) < 1e-12, "data loss includes pixels outside Core + Guard");
        }
        check_joint_derivative(exchange, masks, target, options);
    }

    // 按局部窗口中的每个像素查找对应的周期全局 EP，避免复用被测函数的 EP 平移/筛选逻辑。
    const auto global_eps = EpSelect(target, 1.0, 1.0).select_eps_via(2);
    options.loss_type = MicroLossType::EPE;
    const auto epe = exchange.compute(masks, target, options);
    int guard_points = 0, periodic_points = 0;
    for (const auto& tile : meta.tiles) {
        std::vector<Eigen::Vector2d> positions;
        std::vector<int> indices;
        const auto& w = tile.update_local;
        const auto& core = tile.core_local;
        for (int y = w.y0; y < w.y1; ++y) {
            for (int x = w.x0; x < w.x1; ++x) {
                const int uy = tile.tile_global.y0 + y, ux = tile.tile_global.x0 + x;
                const int gy = (uy % 32 + 32) % 32, gx = (ux % 32 + 32) % 32;
                for (int i = 0; i < global_eps.eps.rows(); ++i) {
                    if (global_eps.eps(i, 0) != gy || global_eps.eps(i, 1) != gx) continue;
                    positions.emplace_back(y, x);
                    indices.push_back(i);
                    guard_points += y < core.y0 || y >= core.y1 || x < core.x0 || x >= core.x1;
                    periodic_points += uy < 0 || uy >= 32 || ux < 0 || ux >= 32;
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
        const double expected = Loss::epe_loss(image.aerial_image, eps, 0.25, options.epe_pixel_size);
        const auto expected_gradient = gradient.epe_gradient(
            image.aerial_image, imaging.get_electric_field(), eps, 0.25, options.epe_pixel_size);
        require(std::abs(expected - epe.locals[tile.id].epe_loss) < 1e-11 * std::max(1.0, expected), "wrong Core + Guard EPE loss");
        require((expected_gradient - epe.locals[tile.id].raw_mask_gradient).norm() < 1e-11 * std::max(1.0, expected_gradient.norm()),
                "wrong Core + Guard EPE gradient");
    }
    require(guard_points > 0 && periodic_points > 0, "EPE fixture missed Guard or periodic points");

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
    std::cout << (socs ? "SOCS" : "Abbe") << ": Core+Guard PE/EPE, periodic EP and Guard-only CEPT passed\n";
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--real") {
            check_real_patterns();
        } else {
            for (bool socs : {false, true}) {
                check_case(litho::Partition::build(24, 28, 2, 3, 5, 2), socs);
                check_case(litho::Partition::build(25, 31, 3, 3, 5, 2), socs);
                check_case(litho::Partition::build(16, 16, 2, 2, 0, 0), socs);
                check_case(litho::Partition::build(8, 10, 1, 1, 12, 4), socs);
                check_options(socs);
                check_lsm_regions(socs);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
