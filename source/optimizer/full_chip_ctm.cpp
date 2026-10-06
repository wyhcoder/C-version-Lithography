#include "full_chip_ctm.h"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <utility>
#include "CTM_Optimizer.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace litho {

FullChipCTMOptimizer::FullChipCTMOptimizer(const PartitionMeta& meta, const ImagingCache& cache, double threshold, double alpha)
    : _meta(meta), _exchange(meta, cache, threshold, alpha) {}

FullChipCTMResult FullChipCTMOptimizer::optimize(const Eigen::MatrixXd& global_target, const FullChipCTMOptions& options) {
    return optimize(global_target, global_target, options);
}

FullChipCTMResult FullChipCTMOptimizer::optimize(const Eigen::MatrixXd& initial_pattern, const Eigen::MatrixXd& global_target,
                                              const FullChipCTMOptions& options) {
    const auto& gradient_options = options.gradient_options;
    // 与 evaluator 的损失选择一致：只有 PE_EPE 使用混合权重，其余模式的有效权重为 0 或 1。
    const double pe_weight = gradient_options.loss_type == MicroLossType::EPE ? 0.0 :
                             gradient_options.loss_type == MicroLossType::PE_EPE ? gradient_options.pe_weight : 1.0;
    const double epe_weight = gradient_options.loss_type == MicroLossType::EPE ? 1.0 :
                              gradient_options.loss_type == MicroLossType::PE_EPE ? gradient_options.epe_weight : 0.0;
    const char* loss_type = "";
    switch (gradient_options.loss_type) {
        case MicroLossType::PE: loss_type = "PE"; break;
        case MicroLossType::PEPenalty: loss_type = "PEPenalty"; break;
        case MicroLossType::EPE: loss_type = "EPE"; break;
        case MicroLossType::PE_EPE: loss_type = "PE_EPE"; break;
    }
    if (options.verbose) {
        std::cout << "CTM 损失配置: type=" << loss_type << ", 数据区域="
                  << (gradient_options.loss_region == MicroLossRegion::CoreGuard ? "Core+Guard(包含角区)" : "Core")
                  << ", data_loss=";
        if (gradient_options.loss_type == MicroLossType::PEPenalty) {
            std::cout << "PE + " << gradient_options.penalty_weight << "*BG_penalty_raw";
        } else if (gradient_options.loss_type == MicroLossType::PE) {
            std::cout << "PE";
        } else if (gradient_options.loss_type == MicroLossType::EPE) {
            std::cout << "EPE";
        } else {
            std::cout << pe_weight << "*PE + " << epe_weight << "*EPE";
        }
        std::cout << ", total_loss=data_loss";
        if (gradient_options.enable_cept) std::cout << " + " << gradient_options.cept_weight << "*CEPT_raw";
        std::cout << '\n';
        if (gradient_options.loss_type == MicroLossType::PEPenalty) {
            std::cout << "  BG_penalty_raw=Σ[target<0.5] max(WI-" << gradient_options.penalty_threshold << ",0)^2\n";
        }
        if (gradient_options.enable_cept) {
            std::cout << "  CEPT_raw=Σ(本块Guard mask-四邻域Core mask)^2；"
                      << (gradient_options.cept_guard_only ? "邻居Core作为冻结参考。" : "梯度作用于Guard及对应邻居Core两端。")
                      << '\n';
        }
        std::cout << "  *_weighted均为已加权贡献；梯度融合不计入total_loss；theta_step为theta域固定学习率。" << std::endl;
    }
    FullChipCTMResult state; // 初始化结果
    state.local_theta = Partition::extract_all(CTM_Optimizer::initialize_theta(initial_pattern), _meta);
    const int tile_count = static_cast<int>(_meta.tiles.size()); // micro数量
    state.local_masks.resize(tile_count);
    state.history.reserve(options.iterations + 1);
    for (int id = 0; id < tile_count; ++id) state.local_masks[id] = (0.5 * (1.0 + state.local_theta[id].array().cos())).matrix();
    state.global_mask = Partition::assemble_global(state.local_masks, _meta); // 将每个 micro 的 mask 回拼成全局 mask，供后续评价使用

    // 同一轮所有块读取冻结的局部状态；交换结果全部生成后，才允许任何块更新 theta。
    auto evaluation = _exchange.compute(state.local_masks, global_target, options.gradient_options, options.parallel_micro);
    FullChipCTMResult best;
    best.best_loss = std::numeric_limits<double>::infinity();
    auto record = [&](int iteration) {
        // 普通 PE 与已加权的背景惩罚分开记录；这些分项只用于展示，不改变总损失或更新梯度。
        double pe = 0.0, penalty = 0.0, epe = 0.0;
        for (const auto& local : evaluation.locals) {
            pe += pe_weight * (local.pe_loss - local.penalty_loss);
            penalty += local.penalty_loss;
            epe += epe_weight * local.epe_loss;
        }
        const double cept = gradient_options.enable_cept ? gradient_options.cept_weight * evaluation.cept_loss : 0.0;
        state.history.push_back({iteration, evaluation.loss, evaluation.data_loss, evaluation.cept_loss, pe, penalty, epe, cept});
        if (evaluation.loss < best.best_loss) {
            // CEPT 依赖邻居关系，必须保存同一轮的所有块及其 Halo/Guard，保证返回状态与最佳 loss 对应。
            best.best_iteration = iteration;
            best.best_loss = evaluation.loss;
            best.local_masks = state.local_masks;
            best.local_theta = state.local_theta;
        }
        if (options.verbose) {
            // 局部字符串控制精度，避免改变调用者 cout 格式；按实际损失类型展示参与的分项。
            std::ostringstream line;
            line << std::fixed << std::setprecision(3) << "CTM iter " << iteration << '/' << options.iterations
                 << " total_loss=" << evaluation.loss << " | data_loss(" << loss_type << ")=" << evaluation.data_loss << " [";
            if (gradient_options.loss_type != MicroLossType::EPE) line << "PE_weighted=" << pe;
            if (gradient_options.loss_type == MicroLossType::PEPenalty) line << ", BG_penalty_weighted=" << penalty;
            if (gradient_options.loss_type == MicroLossType::EPE || gradient_options.loss_type == MicroLossType::PE_EPE) {
                line << (gradient_options.loss_type == MicroLossType::PE_EPE ? ", " : "") << "EPE_weighted=" << epe;
            }
            line << "] | ";
            if (gradient_options.enable_cept) {
                line << "CEPT_raw=" << evaluation.cept_loss << ", CEPT_weighted=" << cept;
            } else {
                line << "CEPT=off";
            }
            line << " | theta_step=" << std::setprecision(6) << options.learning_rate;
            std::cout << line.str() << std::endl;
        }
    };
    record(0);

    for (int iteration = 1; iteration <= options.iterations; ++iteration) {
        // 梯度已交换/融合；各线程只更新本块 Core + Guard，外层 Halo 留给后面的串行上下文刷新。
#ifdef _OPENMP
        const int worker_count = std::min(tile_count, omp_get_max_threads());
#pragma omp parallel for num_threads(worker_count) if(options.parallel_micro) schedule(static)
#endif
        for (int id = 0; id < tile_count; ++id) {
            const PartitionRect& window = _meta.tiles[id].update_local;
            auto theta = state.local_theta[id].block(window.y0, window.x0, window.rows(), window.cols());
            const auto gradient = evaluation.locals[id].mask_gradient.block(window.y0, window.x0, window.rows(), window.cols());
            // 链式法则：G_theta = -0.5 * G_mask * sin(theta)，固定步长更新 theta，不直接对 mask 做减法。
            const Eigen::MatrixXd theta_gradient = (-0.5 * gradient.array() * theta.array().sin()).matrix();
            theta -= options.learning_rate * theta_gradient;
            state.local_masks[id] = (0.5 * (1.0 + state.local_theta[id].array().cos())).matrix();
        }

        // 并行更新结束后再回拼和刷新；Guard 独立保留，因此更新后可产生非零 CEPT。
        state.global_mask = Partition::assemble_global(state.local_masks, _meta);
        _refresh_outer_halo(state);
        // 评价更新后的状态；CEPT 使用本轮更新后的像素值，得到的梯度供下一次更新使用。
        evaluation = _exchange.compute(state.local_masks, global_target, options.gradient_options, options.parallel_micro);
        record(iteration);
    }
    best.global_mask = Partition::assemble_global(best.local_masks, _meta);
    best.history = std::move(state.history);
    return best;
}

void FullChipCTMOptimizer::_refresh_outer_halo(FullChipCTMResult& state) {
    const Eigen::MatrixXd global_theta = Partition::assemble_global(state.local_theta, _meta);
    for (const MicroTile& tile : _meta.tiles) {
        const Eigen::MatrixXd context = Partition::extract_local_tile(global_theta, tile);
        const PartitionRect& window = tile.update_local;
        // 上下整条带 + 左右中间条带，正好覆盖更新窗口外部；避免覆盖独立优化的 Guard。
        auto& theta = state.local_theta[tile.id];
        theta.topRows(window.y0) = context.topRows(window.y0);
        theta.bottomRows(theta.rows() - window.y1) = context.bottomRows(theta.rows() - window.y1);
        theta.block(window.y0, 0, window.rows(), window.x0) = context.block(window.y0, 0, window.rows(), window.x0);
        theta.block(window.y0, window.x1, window.rows(), theta.cols() - window.x1) =
            context.block(window.y0, window.x1, window.rows(), theta.cols() - window.x1);
        state.local_masks[tile.id] = (0.5 * (1.0 + theta.array().cos())).matrix();
    }
}

} // namespace litho
