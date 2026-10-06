#include "full_chip_lsm.h"
#include "level_set_utils.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
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
    // LSM 固定采用 Python 的区域规则；共享 evaluator 的默认设置仍供 CTM 使用。
    auto gradient_options = options.gradient_options;
    gradient_options.loss_region = MicroLossRegion::CoreGuard;
    gradient_options.cept_guard_only = true;
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
        // 打印实际参与优化的权重；PEPenalty 不使用配置中的 PE_EPE 混合权重。
        std::cout << "LSM 损失配置: type=" << loss_type << ", 数据区域=Core+Guard(包含角区), data_loss=";
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
            std::cout << "  CEPT_raw=Σ(本块Guard mask-四邻域Core mask)^2；邻居Core作为冻结参考。\n";
        }
        std::cout << "  *_weighted均为已加权贡献；梯度融合和曲率平滑不计入total_loss；dt[min,max]为各Micro数值时间步范围。"
                  << std::endl;
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
        state.local_masks[id] = (state.local_phi[id].array() >= 0.0).cast<double>().matrix();
    }
    state.initial_global_phi = Partition::assemble_global(state.local_phi, _meta);

    // 同一轮的全部局部 mask 先成像/反传；compute() 的并行区结束后，内部串行完成 CEPT 和梯度融合。
    auto evaluation = _exchange.compute(state.local_masks, global_target, gradient_options, options.parallel_micro);
    FullChipLSMResult best;
    best.best_loss = std::numeric_limits<double>::infinity();
    auto record = [&](int iteration, double min_dt, double max_dt) {
        // 各 Micro 的普通 PE、背景惩罚和 EPE 分别汇总；权重与 evaluator 的损失选择一致。
        double pe = 0.0, penalty = 0.0, epe = 0.0;
        for (const auto& local : evaluation.locals) {
            pe += pe_weight * (local.pe_loss - local.penalty_loss);
            penalty += local.penalty_loss;
            epe += epe_weight * local.epe_loss;
        }
        const double cept = gradient_options.enable_cept ? gradient_options.cept_weight * evaluation.cept_loss : 0.0;
        state.history.push_back({iteration, evaluation.loss, evaluation.data_loss, evaluation.cept_loss, min_dt, max_dt,
                                 pe, penalty, epe, cept});
        if (evaluation.loss < best.best_loss) {
            // 一次保存所有块的同轮状态，确保 CEPT、phi 和 mask 与 best_loss 一致。
            best.best_iteration = iteration;
            best.best_loss = evaluation.loss;
            best.local_phi = state.local_phi;
            best.local_masks = state.local_masks;
        }
        if (options.verbose) {
            // 使用局部字符串控制精度，不改变调用者 cout 的格式；损失三位小数，时间步六位小数。
            std::ostringstream line;
            line << std::fixed << std::setprecision(3) << "LSM iter " << iteration << '/' << options.iterations
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
            line << " | dt[min,max]=" << std::setprecision(6) << '[' << min_dt << ',' << max_dt << ']';
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
                // 先按原渲染规则转为 +/-0.5，令 SDF 重初始化保持 phi>=0 的像素归属；只重写 Core + Guard。
                const Eigen::MatrixXd signs = ((next_phi[id].array() >= 0.0).cast<double>() - 0.5).matrix();
                const Eigen::MatrixXd sdf = LevelSetUtils::reinit_sdf(signs, options.dx, options.dy);
                next_phi[id].block(window.y0, window.x0, window.rows(), window.cols()) =
                    sdf.block(window.y0, window.x0, window.rows(), window.cols());
            }
            // 仅渲染新 mask，不从邻居 Core 刷新 phi；窗口外保留本块初始化时的完整 SDF。
            next_masks[id] = (next_phi[id].array() >= 0.0).cast<double>().matrix();
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
    best.history = std::move(state.history);
    return best;
}

} // namespace litho
