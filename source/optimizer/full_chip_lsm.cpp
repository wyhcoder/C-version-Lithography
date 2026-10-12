#include "full_chip_lsm.h"
#include "level_set_utils.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace litho {

namespace {

struct EvolutionTerm {
    Eigen::MatrixXd normal; // 原 C++ LSM 的演化右端：曲率项 - 法向项，尺寸与局部 phi 相同。
    double dt; // 根据本块完整 H1/H2 场得到的 CFL 步长；两个 RK 阶段使用第一阶段的 dt。
};

// 直接复用小版图 LSM 的 WENO/Godunov、曲率和 CFL 公式；gradient 作为法向速度，不乘 CTM 的 sin(theta)。
EvolutionTerm evolution_term(const Eigen::MatrixXd& phi, const Eigen::MatrixXd& gradient, const FullChipLSMOptions& options) {
    const auto params = LevelSetUtils::evolve_normal_WENO_godunov(phi, gradient, options.dy, options.dx);
    const Eigen::MatrixXd kappa = LevelSetUtils::evolve_kappa(phi, options.dx, options.dy, options.curvature_weight);
    const double dt = LevelSetUtils::get_dt_normal_kappa(options.cfl, options.dx, options.dy,
                                                       params.H1_abs, params.H2_abs, options.curvature_weight);
    return {kappa - params.delta, dt};
}

} // namespace

FullChipLSMOptimizer::FullChipLSMOptimizer(const PartitionMeta& meta, const ImagingCache& cache, double threshold, double alpha)
    : _meta(meta), _exchange(meta, cache, threshold, alpha) {}

FullChipLSMResult FullChipLSMOptimizer::optimize(const Eigen::MatrixXd& initial_binary, const Eigen::MatrixXd& global_target,
                                              const FullChipLSMOptions& options) {
    FullChipLSMResult state;
    // 评价范围固定 Core + Guard，具体主图/背景窗口由配置的 loss_type 决定；反传覆盖完整 Halo。
    auto gradient_options = options.gradient_options;
    gradient_options.loss_region = MicroLossRegion::CoreGuard;
    gradient_options.cept_guard_only = true;
    gradient_options.collect_core_metrics = true;
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
        // 配置按损失、边界协调和 EP 统计分组；中文标签按显示宽度对齐，不用 setw() 计算 UTF-8 字节宽度。
        // 用局部字符串统一输出，避免修改调用者 cout 的精度；这里打印的是实际生效的系数。
        std::ostringstream config;
        config << "\n----------------------- LSM 配置 -----------------------\n"
               << "  Mask渲染 : ";
        if (options.mask_type == LevelSetMaskType::Gray) {
            config << "Gray | phi 双线性放大 " << options.render_scale << " 倍 → 零阈值二值化 → 块平均\n";
        } else {
            config << "Binary | 原像素 phi >= 0\n";
        }
        config << "  损失类型 : " << loss_type << '\n'
               << "  评价区域 : Core + Guard（含角区）；各块 Guard 重叠部分分别评价\n"
               << "  损失窗口 : ";
        if (gradient_options.loss_type == MicroLossType::PE) config << "PE 包含主图形及背景全部像素\n";
        else if (show_bg) config << "PE 仅 target > 0.5 的原主图；BG 仅 target < 0.5 的原背景\n";
        else if (gradient_options.loss_type == MicroLossType::EPE) config << "原 target 主图轮廓 EP，不膨胀选点\n";
        else config << "主图 3×3 膨胀 " << gradient_options.main_region_dilation_pixels << " 次后评价 EPE，外部评价 PE\n";
        config << "  梯度范围 : 完整 Core + Halo 光学反传；phi 仅更新 Core + Guard\n"
               << "  数据损失 : ";
        if (gradient_options.loss_type == MicroLossType::PEPenalty) {
            config << pe_weight << " * PE-main(raw) + " << penalty_weight << " * BG(raw)";
        } else if (gradient_options.loss_type == MicroLossType::PE) {
            config << pe_weight << " * PE-all(raw)";
        } else if (gradient_options.loss_type == MicroLossType::EPE) {
            config << epe_weight << " * EPE(raw)";
        } else {
            config << pe_weight << " * PE-bg(raw) + " << epe_weight << " * EPE(raw)";
        }
        config << "\n  总损失   : data_loss";
        if (gradient_options.enable_cept) config << " + " << gradient_options.cept_weight << " * CEPT_raw";
        config << '\n';
        if (show_bg) {
            config << "  背景惩罚 : BG = Σ[背景窗口且 target < 0.5] max(WI - "
                   << gradient_options.penalty_threshold << ", 0)^2\n";
        }
        config << "\n  CEPT约束 : ";
        if (gradient_options.enable_cept) {
            config << "开启；CEPT_raw = Σ(本块 Guard mask - 邻居 Core mask)^2\n"
                   << "             四邻域 Core 为冻结参考，仅驱动本块 Guard\n";
        } else {
            config << "关闭\n";
        }
        config << "  梯度融合 : ";
        if (gradient_options.enable_grad_fusion) {
            config << (gradient_options.fusion_mode == GradientFusionMode::Average ? "Average" : "Linear")
                   << "；本块/邻居权重 = " << gradient_options.fusion_weight_owner << " / "
                   << gradient_options.fusion_weight_neighbor << "，仅融合 Guard 数据梯度\n";
        } else {
            config << "关闭\n";
        }
        config << "\n  评价选点 : ";
        if (gradient_options.epe_sampling == MicroEpeSampling::Contact) {
            config << "contact | select_eps_via(0)，仅四边中点\n";
        } else if (gradient_options.epe_sampling == MicroEpeSampling::Metal) {
            config << "metal | select_eps_others(line=" << gradient_options.epe_interval_line
                   << ", corner=" << gradient_options.epe_interval_corner << ")\n";
        } else {
            config << "select_eps_via(2)\n";
        }
        config << "  选点权重 : " << (gradient_options.epe_all_eps ? "全部已选 EP 的 EPE 权重为 1" : "沿用原选点函数的 EPE 权重")
               << '\n'
               << "  测量尺度 : dx = " << gradient_options.epe_pixel_size << "；传入 nm/像素时，EPE 单位为 nm\n"
               << "\n  日志说明 : 只打印所选类型包含的原始损失；Total-loss 是各项乘配置权重后的总和\n"
               << "             EPE-mean = EPE 区域原始 EPE 之和 / 实际评价 EP 数，含 Guard 重叠/周期副本\n"
               << "             EPE 报告绝对值；速度项沿用冻结光强斜率的平方 EPE 近似驱动\n"
               << "             CEPT 单独评价 Guard 与邻居 Core；融合和曲率不计入损失\n"
               << "--------------------------------------------------------\n";
        // 迭代表头与数据列共用固定宽度；ASCII 列名适合 setw() 对齐，不输出 Core 明细或时间步。
        config << "LSM " << std::right << std::setw(7) << "Iter";
        if (show_pe) config << std::setw(15) << pe_label;
        if (show_epe) config << std::setw(15) << "EPE(raw)";
        if (show_bg) config << std::setw(15) << "BG(raw)";
        if (gradient_options.enable_cept) config << std::setw(15) << "CEPT(raw)";
        config << std::setw(15) << "Total-loss";
        if (show_epe) config << std::setw(15) << "EPE-mean";
        config << '\n';
        std::cout << config.str() << std::flush;
    }
    const int count = static_cast<int>(_meta.tiles.size());
    state.local_phi.resize(count);
    state.local_masks.resize(count);
    state.history.reserve(options.iterations + 1);
#ifdef _OPENMP
    const int workers = std::min(count, omp_get_max_threads());
#pragma omp parallel for num_threads(workers) if(options.parallel_micro) schedule(static)
#endif
    for (int id = 0; id < count; ++id) {
        // 与 Python 全芯片流程一致：先提取含 Halo 的二值 mask，再各自初始化 phi；内部为正、背景为负。
        const Eigen::MatrixXd mask = Partition::extract_local_tile(initial_binary, _meta.tiles[id]);
        state.local_phi[id] = LevelSetUtils::reinit_sdf((mask.array() - 0.5).matrix(), options.dx, options.dy);
        state.local_masks[id] = render_levelset_mask(state.local_phi[id], options.mask_type, options.render_scale);
    }
    state.initial_global_phi = Partition::assemble_global(state.local_phi, _meta);
    state.initial_global_mask = Partition::assemble_global(state.local_masks, _meta);

    // 同一轮的全部局部 mask 先成像/反传；compute() 的并行区结束后，内部串行完成 CEPT 和梯度融合。
    auto evaluation = _exchange.compute(state.local_masks, global_target, gradient_options, options.parallel_micro);
    FullChipLSMResult best;
    best.best_loss = std::numeric_limits<double>::infinity();
    auto record = [&](int iteration, double min_dt, double max_dt) {
        // 各 Micro 的普通 PE、背景惩罚和 EPE 分别汇总；权重与 evaluator 的损失选择一致。
        double pe = 0.0, penalty = 0.0, epe = 0.0;
        double total_pe = 0.0, total_bg = 0.0, total_epe = 0.0;
        int ep_count = 0;
        CoreLossMetrics core;
        for (const auto& local : evaluation.locals) {
            pe += pe_weight * local.pe_loss;
            penalty += local.penalty_loss;
            epe += epe_weight * local.epe_loss;
            total_pe += local.pe_loss;
            total_bg += local.bg_penalty_raw;
            total_epe += local.epe_loss;
            ep_count += local.epe_count;
            core.pe_loss += local.core.pe_loss;
            core.bg_penalty_raw += local.core.bg_penalty_raw;
            core.pe_penalty_loss += local.core.pe_penalty_loss;
            core.epe_loss += local.core.epe_loss;
            core.ep_count += local.core.ep_count;
            core.data_loss += local.core.data_loss;
        }
        core.epe_mean = core.ep_count == 0 ? 0.0 : core.epe_loss / core.ep_count;
        const double cept = gradient_options.enable_cept ? gradient_options.cept_weight * evaluation.cept_loss : 0.0;
        // 使用实际 EPE 窗口点集求均值；Core + Guard 的重叠点和周期副本参与评价，不平均各块均值。
        const double epe_mean = ep_count == 0 ? 0.0 : total_epe / ep_count;
        state.history.push_back({iteration, evaluation.loss, evaluation.data_loss, evaluation.cept_loss, min_dt, max_dt,
                                 pe, penalty, epe, cept, core, total_pe, total_bg, total_epe, ep_count, epe_mean});
        if (evaluation.loss < best.best_loss) {
            // 一次保存所有块的同轮状态，确保 CEPT、phi 和 mask 与 best_loss 一致。
            best.best_iteration = iteration;
            best.best_loss = evaluation.loss;
            best.local_phi = state.local_phi;
            best.local_masks = state.local_masks;
            best.best_core_metrics.clear();
            for (const auto& local : evaluation.locals) best.best_core_metrics.push_back(local.core);
        }
        if (options.verbose) {
            // 表头与数据按所选损失类型同步裁列；未参与优化的诊断值仅保存在 history，不混入迭代日志。
            std::ostringstream line;
            line << std::fixed << std::setprecision(3) << "LSM " << std::setw(7)
                 << (std::to_string(iteration) + "/" + std::to_string(options.iterations));
            if (show_pe) line << std::setw(15) << total_pe;
            if (show_epe) line << std::setw(15) << total_epe;
            if (show_bg) line << std::setw(15) << total_bg;
            if (gradient_options.enable_cept) line << std::setw(15) << evaluation.cept_loss;
            line << std::setw(15) << evaluation.loss;
            if (show_epe) line << std::setw(15) << epe_mean;
            std::cout << line.str() << std::endl;
        }
    };
    record(0, 0.0, 0.0);

    for (int iteration = 1; iteration <= options.iterations; ++iteration) {
        std::vector<Eigen::MatrixXd> next_phi(count);
        std::vector<Eigen::MatrixXd> next_masks(count);
        std::vector<double> steps(count);
#ifdef _OPENMP
#pragma omp parallel for num_threads(workers) if(options.parallel_micro) schedule(static)
#endif
        for (int id = 0; id < count; ++id) {
            const auto& tile = _meta.tiles[id];
            const auto& window = tile.update_local; // 局部 [y0,y1) × [x0,x1) 的 Core + Guard 更新窗口。
            const auto& phi = state.local_phi[id]; // 冻结的旧 phi；所有 next_phi 算完前不写回权威状态。
            // Python 流程也将总驱动乘更新窗口；外围 Halo 只提供导数上下文，不参与本块的法向演化。
            const Eigen::MatrixXd gradient = evaluation.locals[id].mask_gradient.cwiseProduct(Partition::update_window(tile));
            const auto first = evolution_term(phi, gradient, options);
            steps[id] = first.dt; // 每块独立的 CFL dt，第二阶段使用同一个 dt 和同一轮已交换的梯度。
            Eigen::MatrixXd phi1 = phi;
            phi1.block(window.y0, window.x0, window.rows(), window.cols()) +=
                first.dt * first.normal.block(window.y0, window.x0, window.rows(), window.cols());
            const auto second = evolution_term(phi1, gradient, options);
            next_phi[id] = phi;
            // 原 C++ 两阶段 RK2：phi_new = 0.5*phi + 0.5*(phi1 + dt*Normal(phi1))，仅提交窗口内的值。
            next_phi[id].block(window.y0, window.x0, window.rows(), window.cols()) =
                0.5 * phi.block(window.y0, window.x0, window.rows(), window.cols()) +
                0.5 * (phi1.block(window.y0, window.x0, window.rows(), window.cols()) +
                       first.dt * second.normal.block(window.y0, window.x0, window.rows(), window.cols()));
            if (options.reinit_interval > 0 && iteration % options.reinit_interval == 0) {
                // 重初始化按 phi 的符号构建 +/-0.5，不对灰度 mask 再阈值化，保持原像素中心的内外归属。
                const Eigen::MatrixXd signs = ((next_phi[id].array() >= 0.0).cast<double>() - 0.5).matrix();
                const Eigen::MatrixXd sdf = LevelSetUtils::reinit_sdf(signs, options.dx, options.dy);
                next_phi[id].block(window.y0, window.x0, window.rows(), window.cols()) =
                    sdf.block(window.y0, window.x0, window.rows(), window.cols());
            }
            // 渲染完整局部 phi 的实际成像 mask；梯度继续作为 PDE 法向速度，不对硬阈值 renderer 求 phi 导数。
            next_masks[id] = render_levelset_mask(next_phi[id], options.mask_type, options.render_scale);
        }
        // OpenMP 隐式等待后统一提交；Core + Guard 之外的 phi 在 RK2、重初始化和交换中都不改写。
        state.local_phi = std::move(next_phi);
        state.local_masks = std::move(next_masks);
        evaluation = _exchange.compute(state.local_masks, global_target, gradient_options, options.parallel_micro);
        record(iteration, *std::min_element(steps.begin(), steps.end()), *std::max_element(steps.begin(), steps.end()));
    }
    best.global_phi = Partition::assemble_global(best.local_phi, _meta);
    best.global_mask = Partition::assemble_global(best.local_masks, _meta);
    best.initial_global_phi = std::move(state.initial_global_phi);
    best.initial_global_mask = std::move(state.initial_global_mask);
    best.history = std::move(state.history);
    return best;
}

} // namespace litho
