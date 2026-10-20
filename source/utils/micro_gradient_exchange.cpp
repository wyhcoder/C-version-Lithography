#include "micro_gradient_exchange.h"

#include <algorithm>
#include <opencv2/imgproc.hpp>
#include "epe_utils.h"
#include "loss.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace litho {

namespace {

// region 为局部半开像素范围；筛出区域内且 epe_window=1 的全局 EP，转为局部坐标并保留原权重。
// Core + Guard 可跨芯片边界，因此同时筛选相邻周期的 EP 副本，与周期提取的 local_target 一致。
EpsResult region_eps_local(const EpsResult& global_eps, const MicroTile& tile, const PartitionRect& region,
                          int height, int width, bool periodic_guard, const Eigen::MatrixXd& epe_window) {
    std::vector<int> selected;
    std::vector<Eigen::Vector2d> positions;
    const PartitionRect global_region = region.shifted(tile.tile_global.y0, tile.tile_global.x0);
    // height/width 为全图像素尺寸；py/px 为周期副本编号，-1/0/1 分别表示前一周期、原图、后一周期。
    const int periods = periodic_guard ? 1 : 0;
    for (int i = 0; i < global_eps.eps.rows(); ++i) {
        for (int py = -periods; py <= periods; ++py) {
            for (int px = -periods; px <= periods; ++px) {
                const double y = global_eps.eps(i, 0) + py * height, x = global_eps.eps(i, 1) + px * width;
                if (y >= global_region.y0 && y < global_region.y1 && x >= global_region.x0 && x < global_region.x1) {
                    // EP 坐标保持浮点；任一双线性支撑像素落入 EPE 窗口即视为该点与窗口相交。
                    const double local_y = y - tile.tile_global.y0, local_x = x - tile.tile_global.x0;
                    const auto stencil = make_bilinear_stencil(epe_window.rows(), epe_window.cols(), local_y, local_x);
                    if (!bilinear_touches_window(epe_window, stencil)) continue;
                    selected.push_back(i);
                    positions.emplace_back(local_y, local_x);
                }
            }
        }
    }
    const int count = static_cast<int>(selected.size());
    EpsResult local{Eigen::MatrixXd(count, 2), Eigen::RowVectorXd(count), Eigen::RowVectorXd(count)};
    for (int j = 0; j < count; ++j) {
        const int i = selected[j];
        local.eps.row(j) = positions[j].transpose();
        local.weight_epe(j) = global_eps.weight_epe(i);
        local.weight_meef(j) = global_eps.weight_meef(i);
    }
    return local;
}

// 线性融合：靠近 Core 时邻居贡献为 0，向 Guard 外侧增加到 1；单像素 Guard 的贡献为 1。
double linear_strength(NeighborDirection direction, int row, int col, int rows, int cols) {
    double strength = 1.0;
    switch (direction) {
        case NeighborDirection::Left: strength = cols == 1 ? 1.0 : 1.0 - static_cast<double>(col) / (cols - 1); break;
        case NeighborDirection::Right: strength = cols == 1 ? 1.0 : static_cast<double>(col) / (cols - 1); break;
        case NeighborDirection::Top: strength = rows == 1 ? 1.0 : 1.0 - static_cast<double>(row) / (rows - 1); break;
        case NeighborDirection::Bottom: strength = rows == 1 ? 1.0 : static_cast<double>(row) / (rows - 1); break;
    }
    return strength;
}

} // namespace

MicroLossWindows build_micro_loss_windows(const Eigen::MatrixXd& local_target, const PartitionRect& region,
                                         const MicroGradientOptions& options) {
    const int height = static_cast<int>(local_target.rows()), width = static_cast<int>(local_target.cols());
    MicroLossWindows windows{Eigen::MatrixXd::Zero(height, width), Eigen::MatrixXd::Zero(height, width),
                             Eigen::MatrixXd::Zero(height, width)};
    cv::Mat main_region;
    if (options.loss_type == MicroLossType::PE_EPE) {
        main_region.create(height, width, CV_8UC1);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) main_region.at<uchar>(y, x) = local_target(y, x) > 0.5;
        }
        // 仅混合模式膨胀主图：3×3 结构元素，局部数组边界外为 0，先膨胀再截到半开评价区间。
        cv::dilate(main_region, main_region, cv::Mat::ones(3, 3, CV_8UC1), cv::Point(-1, -1),
                   options.main_region_dilation_pixels, cv::BORDER_CONSTANT, cv::Scalar(0));
    }
    for (int y = region.y0; y < region.y1; ++y) {
        for (int x = region.x0; x < region.x1; ++x) {
            windows.penalty(y, x) = local_target(y, x) < 0.5;
            // EP 已由 target 主图轮廓选出；纯 EPE 不用像素前景值过滤，避免丢掉落在轮廓外侧的 EP。
            windows.epe(y, x) = 1.0;
            windows.pe(y, x) = options.loss_type == MicroLossType::PEPenalty ? local_target(y, x) > 0.5 : 1.0;
            if (options.loss_type == MicroLossType::PE_EPE) {
                windows.epe(y, x) = main_region.at<uchar>(y, x) != 0;
                windows.pe(y, x) = 1.0 - windows.epe(y, x);
            }
        }
    }
    return windows;
}

MicroLossWeights micro_loss_weights(const MicroGradientOptions& options) {
    return {options.loss_type == MicroLossType::EPE ? 0.0 : options.pe_weight,
            options.loss_type == MicroLossType::EPE || options.loss_type == MicroLossType::PE_EPE ? options.epe_weight : 0.0,
            options.loss_type == MicroLossType::PEPenalty ? options.penalty_weight : 0.0};
}

EpsResult select_micro_epe_points(const Eigen::MatrixXd& global_target, const MicroGradientOptions& options) {
    (void)options.epe_sampling; // 版图类型不再切换算法，Contact/Metal/Via 均使用同一套 Python 全芯片选点。
    const EpSelect selector(global_target, 5.0, 1.0);
    EpsResult selected = selector.select_eps_subpixel(options.epe_interval_line, options.epe_interval_corner);
    // 只统一候选点权重，不改变亚像素坐标、候选数量或 MEEF 权重。
    if (options.epe_all_eps) selected.weight_epe.setOnes();
    return selected;
}

MicroGradientExchange::Worker::Worker(const ImagingCache& cache, double threshold, double alpha)
    : imaging(cache), gradient(cache, threshold, alpha) {}

MicroGradientExchange::MicroGradientExchange(const PartitionMeta& meta, const ImagingCache& cache, double threshold, double alpha)
    : _meta(meta), _threshold(threshold), _alpha(alpha) {
    int worker_count = 1;
#ifdef _OPENMP
    worker_count = std::min(static_cast<int>(meta.tiles.size()), omp_get_max_threads());
#endif
    // FFTW 计划的创建在串行区完成，避免多个线程同时调用规划器；计算阶段只执行已有计划。
    _workers.reserve(worker_count);
    for (int i = 0; i < worker_count; ++i) _workers.push_back(std::make_unique<Worker>(cache, threshold, alpha));
}

MicroGradientExchangeResult MicroGradientExchange::compute(const Eigen::MatrixXd& global_mask,
                                                         const Eigen::MatrixXd& global_target, bool parallel_micro) {
    return compute(global_mask, global_target, MicroGradientOptions{}, parallel_micro);
}

MicroGradientExchangeResult MicroGradientExchange::compute(const Eigen::MatrixXd& global_mask, const Eigen::MatrixXd& global_target,
                                                         const MicroGradientOptions& options, bool parallel_micro) {
    return compute(Partition::extract_all(global_mask, _meta), global_target, options, parallel_micro);
}

MicroGradientExchangeResult MicroGradientExchange::compute(const std::vector<Eigen::MatrixXd>& local_masks,
                                                         const Eigen::MatrixXd& global_target,
                                                         const MicroGradientOptions& options, bool parallel_micro) {
    MicroGradientExchangeResult result;
    result.loss = 0.0;
    result.data_loss = 0.0;
    result.cept_loss = 0.0;
    const int tile_count = static_cast<int>(_meta.tiles.size());
    result.locals.resize(tile_count);
    const auto weights = micro_loss_weights(options);
    const bool evaluate_epe = options.loss_type == MicroLossType::EPE || options.loss_type == MicroLossType::PE_EPE ||
                              options.collect_core_metrics;
    EpsResult global_eps;
    // Core 统计也需要 EP，即使优化目标仅为 PE；EP 在全局原始 target 上选，不在裁切后的轮廓上重新选。
    if (evaluate_epe) global_eps = select_micro_epe_points(global_target, options);

    // 阶段一：各线程读取同一轮冻结的 mask，各自完成 Micro 的成像和反传，只写自己的 locals[id]。
    // 在外层 Micro 并行区中，Imaging/Gradient 自动关闭内部光学核并行。
    // _OPENMP 由编译器在启用 OpenMP 时定义；#ifdef 在编译阶段决定是否保留下面的并行代码。
#ifdef _OPENMP
    // 线程数不超过 Worker 数和当前 OpenMP 最大线程数，保证每个线程有自己的工作对象。
    const int worker_count = std::min(static_cast<int>(_workers.size()), omp_get_max_threads());
    // 每个线程都进入下面的大括号；parallel_micro=false 时，同一代码块改为只由一个线程执行。
#pragma omp parallel num_threads(worker_count) if(parallel_micro)
#else
    // 未启用 OpenMP 时仍执行下面的串行代码；此处仅表示参数已被使用，避免未使用参数的编译警告。
    (void)parallel_micro;
#endif
    {
        int worker_id = 0; // 串行时使用第 0 个 Worker；并行时改为当前线程在本线程组中的编号。
#ifdef _OPENMP
        worker_id = omp_get_thread_num();
#endif
        // worker_id 是线程编号，不是 Micro 编号；同一线程处理多个 Micro 时复用自己的 Imaging/Gradient。
        // * 取得 unique_ptr 指向的 Worker，& 绑定引用，不复制对象；各线程的 FFT 缓冲区和电场互不共用。
        Worker& worker = *_workers[worker_id];
#ifdef _OPENMP
        // 将 Micro 循环按编号静态分配给线程，每个 id 只计算一次；64 块、4 线程时每线程处理 16 块。
        // 循环末尾有隐式等待，所有线程完成本阶段计算后才能继续；后续邻居交换在并行区外串行执行。
#pragma omp for schedule(static)
#endif
        for (int id = 0; id < tile_count; ++id) {
            const MicroTile& tile = _meta.tiles[id];
            const Eigen::MatrixXd& local_mask = local_masks[tile.id];
            const Eigen::MatrixXd local_target = Partition::extract_local_tile(global_target, tile);
            const Imaging_Result image = worker.imaging.compute(local_mask, _threshold, _alpha);
            const PartitionRect& region = options.loss_region == MicroLossRegion::CoreGuard ? tile.update_local : tile.core_local;

            // 只在指定区域的输出像素上统计误差；光学反传仍覆盖完整局部 mask，包含外围 Halo。
            const auto windows = build_micro_loss_windows(local_target, region, options);
            MicroGradientData& local = result.locals[tile.id];
            const Eigen::MatrixXd region_wafer = image.wafer_image.block(region.y0, region.x0, region.rows(), region.cols());
            const Eigen::MatrixXd region_target = local_target.block(region.y0, region.x0, region.rows(), region.cols());
            const Eigen::MatrixXd region_pe_window = windows.pe.block(region.y0, region.x0, region.rows(), region.cols());
            local.pe_loss = Loss::pe_loss(region_wafer, region_target, region_pe_window);
            // PEPenalty 的主图 PE 和原始背景惩罚使用不同输出窗口；不再从带 PE 的总量中相减。
            const Eigen::MatrixXd region_bg_window = windows.penalty.block(region.y0, region.x0, region.rows(), region.cols());
            local.bg_penalty_raw = options.loss_type == MicroLossType::PEPenalty || options.collect_core_metrics ?
                Loss::background_penalty_loss(region_wafer, region_target, options.penalty_threshold, region_bg_window) : 0.0;
            local.penalty_loss = weights.penalty * local.bg_penalty_raw;
            local.epe_loss = 0.0;
            local.epe_abs_sum_nm = 0.0;
            local.epe_count = 0;
            local.cept_loss = 0.0;
            local.raw_mask_gradient = Eigen::MatrixXd::Zero(_meta.tile_size, _meta.tile_size);
            local.cept_mask_gradient = Eigen::MatrixXd::Zero(_meta.tile_size, _meta.tile_size);
            if (weights.penalty != 0.0) {
                local.raw_mask_gradient = worker.gradient.pe_gradient_with_penalty(
                    image.wafer_image, local_target, worker.imaging.get_electric_field(), weights.penalty,
                    options.penalty_threshold, windows.pe, weights.pe, windows.penalty);
            } else if (weights.pe != 0.0) {
                local.raw_mask_gradient = weights.pe * worker.gradient.pe_gradient(
                    image.wafer_image, local_target, worker.imaging.get_electric_field(), windows.pe);
            }
            if (evaluate_epe) {
                const EpsResult local_eps = region_eps_local(global_eps, tile, region, _meta.height, _meta.width,
                                                            options.loss_region == MicroLossRegion::CoreGuard, windows.epe);
                // 平方 EPE 目标和绝对 nm 统计复用同一组激活 EP、双线性 AI 样值与梯度幅值。
                const auto epe = Loss::evaluate_sparse_epe(image.aerial_image, local_eps, _threshold, options.epe_pixel_size);
                local.epe_loss = epe.weighted_squared_loss;
                local.epe_abs_sum_nm = epe.absolute_sum_nm;
                local.epe_count = epe.active_count;
                if (weights.epe != 0.0) {
                    local.raw_mask_gradient += weights.epe * worker.gradient.epe_gradient(
                        image.aerial_image, worker.imaging.get_electric_field(), local_eps, _threshold);
                }
            }
            local.data_loss = weights.pe * local.pe_loss + local.penalty_loss + weights.epe * local.epe_loss;
            if (options.collect_core_metrics) {
                const auto& core = tile.core_local; // 本块 Core 的局部像素区间 [y0,y1) × [x0,x1)，各块 Core 不重叠。
                const Eigen::MatrixXd core_wafer = image.wafer_image.block(core.y0, core.x0, core.rows(), core.cols());
                const Eigen::MatrixXd core_target = local_target.block(core.y0, core.x0, core.rows(), core.cols());
                const Eigen::MatrixXd core_pe_window = windows.pe.block(core.y0, core.x0, core.rows(), core.cols());
                local.core.pe_loss = Loss::pe_loss(core_wafer, core_target, core_pe_window);
                const Eigen::MatrixXd core_bg_window = windows.penalty.block(core.y0, core.x0, core.rows(), core.cols());
                local.core.bg_penalty_raw = Loss::background_penalty_loss(
                    core_wafer, core_target, options.penalty_threshold, core_bg_window);
                const Eigen::MatrixXd core_main_window = (core_target.array() > 0.5).cast<double>().matrix();
                local.core.pe_penalty_loss = options.pe_weight * Loss::pe_loss(core_wafer, core_target, core_main_window) +
                                             options.penalty_weight * local.core.bg_penalty_raw;
                // Core EP 只筛原图，不加周期副本；在分界线上用半开区间确定唯一所属块。
                const EpsResult core_eps = region_eps_local(global_eps, tile, core, _meta.height, _meta.width, false, windows.epe);
                // 在完整局部 AI 上求空间梯度并双线性取样；只限制 EP 归属，不能先裁 Core。
                const auto core_epe = Loss::evaluate_sparse_epe(image.aerial_image, core_eps, _threshold, options.epe_pixel_size);
                local.core.epe_loss = core_epe.weighted_squared_loss;
                local.core.epe_abs_sum_nm = core_epe.absolute_sum_nm;
                local.core.ep_count = core_epe.active_count;
                local.core.epe_mean = local.core.ep_count == 0 ? 0.0 : local.core.epe_abs_sum_nm / local.core.ep_count;
                local.core.data_loss = weights.pe * local.core.pe_loss + weights.penalty * local.core.bg_penalty_raw +
                                       weights.epe * local.core.epe_loss;
            }
            local.loss = local.data_loss; // 本块指定区域的数据损失，随后再加入加权 CEPT。
            local.mask_gradient = local.raw_mask_gradient; // 这个是最终驱动的梯度， 还没进行 Guard 融合和 CEPT 约束。
            local.received_neighbors.resize(tile.neighbors.size());
        }
    }

    // 每个 Micro 已完成成像和反传，得到 loss 与 mask_gradient；接下来串行交换邻居信息并计算 CEPT。
    // 阶段二：并行区结束后，串行读取邻居原始结果，进行交换、CEPT 和融合。
    for (int id = 0; id < tile_count; ++id) {
        const MicroTile& tile = _meta.tiles[id]; // 当前块 A 的分区信息，id 是 A 的 Micro 编号。
        MicroGradientData& local = result.locals[id]; // A 的计算结果：loss、完整局部梯度及收到的邻居数据。
        // j 是邻居在 tile.neighbors 中的索引，neighbor.id 是邻居 Micro id。
        for (int j = 0; j < static_cast<int>(tile.neighbors.size()); ++j) {
            const MicroNeighbor& neighbor = tile.neighbors[j]; // A 与邻居 B 的重叠关系；neighbor.id 是 B 的 Micro 编号。
            // B 的完整局部数据项梯度，尺寸为 tile_size × tile_size，含 Core 和 Halo；尚未融合或加入 CEPT。
            const Eigen::MatrixXd& neighbor_gradient = result.locals[neighbor.id].raw_mask_gradient;
            NeighborGradientData& received = result.locals[id].received_neighbors[j]; // A 中用于保存 B 提供的数据的槽位。
            received.neighbor_id = neighbor.id;
            received.direction = neighbor.direction;

            // 以下统一称当前块为 A、邻居为 B。四个 PartitionRect 都是坐标范围，不是 mask 或梯度值。
            // 矩形采用局部像素索引 [y0,y1) × [x0,x1)，y 是行、x 是列，结束索引不包含在区域内。
            // 第一对：guard 在 A 中定位 A 的 Guard；guard_source 在 B 中定位覆盖同一位置的 B Core。
            // 第二对：overlap 在 A 中定位 A 的 Core 边缘；overlap_source 在 B 中定位覆盖同一位置的 B Guard。

            // guard_source：B 局部矩阵中的取数坐标，取出 B Core 中与 A Guard 重叠的部分。
            const PartitionRect& guard_source = neighbor.neighbor_overlap_local;
            // guard_gradient：取出的 B Core 梯度值；名称 guard 表示它将用于 A 的 Guard，数据来源仍是 B。
            received.guard_gradient = neighbor_gradient.block(guard_source.y0, guard_source.x0, guard_source.rows(), guard_source.cols());
            // overlap_source：B 局部矩阵中的取数坐标，取出 B Guard 中与 A Core 边缘重叠的部分。
            const PartitionRect& overlap_source = neighbor.neighbor_guard_local;
            // overlap_gradient 是 B Guard 梯度，对应 A 的 Core 边缘；当前融合分支只使用 guard_gradient。
            received.overlap_gradient = neighbor_gradient.block(overlap_source.y0, overlap_source.x0,
                                                                overlap_source.rows(), overlap_source.cols());
            const PartitionRect& guard = neighbor.my_guard_local; // A 局部坐标：A Guard 与 B Core 重叠的区域。
            const PartitionRect& overlap = neighbor.my_overlap_local; // A 局部坐标：A Core 与 B Guard 重叠的区域。
            if (options.enable_cept) { // CEPT 比较同一全局位置上两块独立保存的 mask 像素值。
                // neighbor_mask：B 的完整局部 mask；local_masks[id] 则是 A 的完整局部 mask。
                const Eigen::MatrixXd& neighbor_mask = local_masks[neighbor.id];
                // guard_mask：取出的 B Core 像素值，与 A Guard 一一对应；取出后矩阵自身的行列索引从 0 开始。
                received.guard_mask = neighbor_mask.block(guard_source.y0, guard_source.x0, guard_source.rows(), guard_source.cols());
                // guard_difference = A Guard 的 mask - B Core 的 mask，是逐像素差值矩阵。
                const Eigen::MatrixXd guard_difference =
                    local_masks[id].block(guard.y0, guard.x0, guard.rows(), guard.cols()) - received.guard_mask;
                // A 负责统计这一项损失：Σ(A_Guard-B_Core)²；squaredNorm() 将矩阵中所有差值平方后求和。
                // cept_loss 尚未乘 cept_weight，遍历完 A 的全部邻居后统一加权。
                local.cept_loss += guard_difference.squaredNorm();
                // 此项对 A Guard 像素的导数是 2*(A_Guard-B_Core)，累加到 A 局部梯度矩阵的 guard 区域。
                local.cept_mask_gradient.block(guard.y0, guard.x0, guard.rows(), guard.cols()) += 2.0 * guard_difference;

                if (!options.cept_guard_only) {
                    // 可选双端模式补上 Σ(B_Guard-A_Core)² 对 A Core 的导数；损失由 B 统计，不重复加。
                    // 全芯片 CTM/LSM 将 B Core 作为冻结参考，CEPT 只写 A Guard，不进入此分支。
                    received.overlap_mask = neighbor_mask.block(overlap_source.y0, overlap_source.x0,
                                                                overlap_source.rows(), overlap_source.cols());
                    const Eigen::MatrixXd overlap_difference =
                        local_masks[id].block(overlap.y0, overlap.x0, overlap.rows(), overlap.cols()) - received.overlap_mask;
                    local.cept_mask_gradient.block(overlap.y0, overlap.x0, overlap.rows(), overlap.cols()) += 2.0 * overlap_difference;
                }
            }
            if (options.enable_grad_fusion) {
                // 四邻域 Guard 条带互不重叠；只改变 Guard，Core 原始数据梯度保持不变。guard 是 A 的局部坐标
                for (int y = 0; y < guard.rows(); ++y) {
                    for (int x = 0; x < guard.cols(); ++x) {
                        const double strength = options.fusion_mode == GradientFusionMode::Average ? 1.0 :
                                                linear_strength(neighbor.direction, y, x, guard.rows(), guard.cols());
                        const double neighbor_weight = options.fusion_weight_neighbor * strength;
                        // 线性过渡的 Core 一端强度为 0，该处保持本块原始梯度。
                        if (neighbor_weight == 0.0) continue;
                        const double local_value = local.raw_mask_gradient(guard.y0 + y, guard.x0 + x);
                        local.mask_gradient(guard.y0 + y, guard.x0 + x) =
                            (options.fusion_weight_owner * local_value + neighbor_weight * received.guard_gradient(y, x)) /
                            (options.fusion_weight_owner + neighbor_weight);
                    }
                }
            }
        }
        if (options.enable_cept) {
            local.mask_gradient += options.cept_weight * local.cept_mask_gradient;
            local.loss += options.cept_weight * local.cept_loss;
        }
    }
    // 固定按 Micro id 顺序汇总标量，计算任务的调度顺序不改变归约顺序。
    for (const MicroGradientData& local : result.locals) {
        result.loss += local.loss;
        result.data_loss += local.data_loss;
        result.cept_loss += local.cept_loss;
    }
    return result;
}

} // namespace litho
