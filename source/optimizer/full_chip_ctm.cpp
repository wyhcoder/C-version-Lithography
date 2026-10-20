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
    auto gradient_options = options.gradient_options;
    // 与全芯片 LSM 一致：邻居 Core 是冻结参考，CEPT 只驱动本块 Guard，不补反向 Core 端导数。
    gradient_options.cept_guard_only = true;
    // CTM/LSM 共用四种损失语义及配置权重，日志分解与实际评价目标一致。
    const auto [pe_weight, epe_weight, penalty_weight] = micro_loss_weights(gradient_options);
    const char* loss_type = "";
    switch (gradient_options.loss_type) {
        case MicroLossType::PE: loss_type = "PE"; break;
        case MicroLossType::PEPenalty: loss_type = "PEPenalty"; break;
        case MicroLossType::EPE: loss_type = "EPE"; break;
        case MicroLossType::PE_EPE: loss_type = "PE_EPE"; break;
    }
    const bool show_pe = gradient_options.loss_type != MicroLossType::EPE;
    const bool show_epe = gradient_options.loss_type == MicroLossType::EPE || gradient_options.loss_type == MicroLossType::PE_EPE;
    const bool show_bg = gradient_options.loss_type == MicroLossType::PEPenalty;
    const char* pe_label = show_bg ? "PE-main(raw)" : gradient_options.loss_type == MicroLossType::PE ? "PE-all(raw)" : "PE-bg(raw)";
    if (options.verbose) {
        std::cout << "CTM 损失配置: type=" << loss_type << ", 数据区域="
                  << (gradient_options.loss_region == MicroLossRegion::CoreGuard ? "Core+Guard(包含角区)" : "Core")
                  << ", data_loss=";
        if (gradient_options.loss_type == MicroLossType::PEPenalty) {
            std::cout << pe_weight << "*PE-main(raw) + " << penalty_weight << "*BG(raw)";
        } else if (gradient_options.loss_type == MicroLossType::PE) {
            std::cout << pe_weight << "*PE-all(raw)";
        } else if (gradient_options.loss_type == MicroLossType::EPE) {
            std::cout << epe_weight << "*EPE-sq(raw)";
        } else {
            std::cout << pe_weight << "*PE-bg(raw) + " << epe_weight << "*EPE-sq(raw)";
            std::cout << "（主图3×3膨胀" << gradient_options.main_region_dilation_pixels << "次，外部评价PE）";
        }
        std::cout << ", total_loss=data_loss";
        if (gradient_options.enable_cept) std::cout << " + " << gradient_options.cept_weight << "*CEPT_raw";
        std::cout << '\n';
        if (show_bg) {
            std::cout << "  BG_penalty_raw=Σ[target<0.5] max(WI-" << gradient_options.penalty_threshold << ",0)^2\n";
        }
        if (gradient_options.enable_cept) {
            std::cout << "1.CEPT_raw=Σ(本块Guard mask-四邻域Core mask)^2；"
                      << "邻居Core作为冻结参考，CEPT只驱动本块Guard。\n";
        }
        std::cout << "2.theta仅更新Core+Guard；窗口外Halo的theta保持本块初值，不从回拼图刷新。\n";
        std::cout << "3.原始损失不乘系数；Total-loss按上述公式加权，梯度融合不计入损失；theta_step="
                  << options.learning_rate << '\n';
        std::cout << "CTM " << std::right << std::setw(7) << "Iter";
        if (show_pe) std::cout << std::setw(15) << pe_label;
        if (show_epe) std::cout << std::setw(15) << "EPE-sq(raw)";
        if (show_bg) std::cout << std::setw(15) << "BG(raw)";
        if (gradient_options.enable_cept) std::cout << std::setw(15) << "CEPT(raw)";
        std::cout << std::setw(15) << "Total-loss" << std::endl;
    }
    FullChipCTMResult state; // 初始化结果
    state.local_theta = Partition::extract_all(CTM_Optimizer::initialize_theta(initial_pattern), _meta);
    const int tile_count = static_cast<int>(_meta.tiles.size()); // micro数量
    state.local_masks.resize(tile_count);
    state.history.reserve(options.iterations + 1);
    for (int id = 0; id < tile_count; ++id) state.local_masks[id] = (0.5 * (1.0 + state.local_theta[id].array().cos())).matrix();

    // 同一轮所有块读取冻结的局部状态；交换结果全部生成后，才允许任何块更新 theta。
    auto evaluation = _exchange.compute(state.local_masks, global_target, gradient_options, options.parallel_micro);
    FullChipCTMResult best;
    best.best_loss = std::numeric_limits<double>::infinity();
    auto record = [&](int iteration) {
        // 普通 PE 与已加权的背景惩罚分开记录；这些分项只用于展示，不改变总损失或更新梯度。
        double pe = 0.0, penalty = 0.0, epe = 0.0, pe_raw = 0.0, bg_raw = 0.0, epe_raw = 0.0, epe_abs_nm = 0.0;
        int epe_count = 0;
        for (const auto& local : evaluation.locals) {
            pe += pe_weight * local.pe_loss;
            penalty += local.penalty_loss;
            epe += epe_weight * local.epe_loss;
            pe_raw += local.pe_loss;
            bg_raw += local.bg_penalty_raw;
            epe_raw += local.epe_loss;
            epe_abs_nm += local.epe_abs_sum_nm;
            epe_count += local.epe_count;
        }
        const double cept = gradient_options.enable_cept ? gradient_options.cept_weight * evaluation.cept_loss : 0.0;
        state.history.push_back({iteration, evaluation.loss, evaluation.data_loss, evaluation.cept_loss,
                                 pe, penalty, epe, cept, pe_raw, bg_raw, epe_raw, epe_abs_nm, epe_count});
        if (evaluation.loss < best.best_loss) {
            // CEPT 依赖邻居关系，必须保存同一轮的所有块及其 Halo/Guard，保证返回状态与最佳 loss 对应。
            best.best_iteration = iteration;
            best.best_loss = evaluation.loss;
            best.local_masks = state.local_masks;
            best.local_theta = state.local_theta;
        }
        if (options.verbose) {
            // 仅展示所选类型的原始分项；系数在配置行显示，总损失包括可选的加权 CEPT。
            std::ostringstream line;
            line << std::fixed << std::setprecision(3) << "CTM " << std::setw(7)
                 << (std::to_string(iteration) + "/" + std::to_string(options.iterations));
            if (show_pe) line << std::setw(15) << pe_raw;
            if (show_epe) line << std::setw(15) << epe_raw;
            if (show_bg) line << std::setw(15) << bg_raw;
            if (gradient_options.enable_cept) line << std::setw(15) << evaluation.cept_loss;
            line << std::setw(15) << evaluation.loss;
            std::cout << line.str() << std::endl;
        }
    };
    record(0);

    for (int iteration = 1; iteration <= options.iterations; ++iteration) {
        // 梯度已交换/融合；各线程只更新局部 [y0,y1) × [x0,x1) 的 Core + Guard，外围 Halo 保留初始 theta。
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

        // 等待所有块更新结束；Guard 独立保留，窗口外 theta 不刷新，当前局部 mask 直接供下一轮成像。
        // 评价更新后的状态；CEPT 使用本轮更新后的像素值，得到的梯度供下一次更新使用。
        evaluation = _exchange.compute(state.local_masks, global_target, gradient_options, options.parallel_micro);
        record(iteration);
    }
    best.global_mask = Partition::assemble_global(best.local_masks, _meta);
    if (options.retain_final_masks) best.final_local_masks = std::move(state.local_masks);
    best.history = std::move(state.history);
    return best;
}

} // namespace litho
