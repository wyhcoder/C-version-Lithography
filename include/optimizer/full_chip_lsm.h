#pragma once

#include "micro_gradient_exchange.h"
#include "levelset_mask_renderer.h"

namespace litho {

struct FullChipLSMOptions {
    int iterations = 200; // phi 更新次数；0 表示只初始化和评价，history 始终包含初始状态。
    double dx = 1.0, dy = 1.0; // 水平集网格步长；phi 的长度单位与步长一致，现有 SDF 初始化要求 dx == dy。
    double cfl = 0.7; // 各 Micro 自己计算 CFL 时间步长，不使用 CTM 的固定 learning_rate。
    double curvature_weight = 0.01; // 原 C++ evolve_kappa() 的 b，控制曲率平滑。
    int reinit_interval = 20; // 每完成多少次更新重新计算 SDF；0 表示关闭。
    LevelSetMaskType mask_type = LevelSetMaskType::Binary; // 成像/梯度/CEPT 共用此渲染结果；Gray 使用子像素覆盖率。
    int render_scale = 4; // Gray 模式每方向的放大倍数；4 对应每个原像素内 16 个子像素，输出尺寸不变。
    MicroGradientOptions gradient_options{.epe_sampling = MicroEpeSampling::Contact, .epe_all_eps = true}; // LSM 的已选 EP 全部等权。
    // LSM 固定 Core + Guard 评价、Guard-only CEPT 和 Core 指标统计；仅 PE_EPE 按膨胀主图分区。
    bool parallel_micro = true; // Micro 成像/反传、SDF 初始化和 PDE 更新并行；四邻域交换串行。
    bool verbose = true;
};

struct FullChipLSMIteration {
    int iteration; // 已完成的更新次数；0 是 CTM 二值结果初始化得到的状态。
    double loss, data_loss, cept_loss; // 总损失、分区数据损失、未加权 CEPT；loss = data_loss + 加权 CEPT。
    double min_dt, max_dt; // 本次更新中各 Micro CFL 步长的最小/最大值；初始状态为 0。
    double pe_contribution, penalty_contribution, epe_contribution, cept_contribution; // 各项已加权贡献，未启用的项为 0。
    // data_loss = 普通 PE 贡献 + 背景惩罚贡献 + EPE 贡献；loss = data_loss + CEPT 贡献。
    CoreLossMetrics core; // 所有不重叠 Core 的指标之和；EPE 平均值按全图 EP 总数计算，不平均各块平均值。
    double total_pe_loss, total_bg_penalty_raw; // Core + Guard 的所选 PE 窗口/原背景原始值；PE 窗口由 loss_type 决定。
    double total_epe_loss; // Core + Guard 的主图 EP 原始 EPE 总和；仅 PE_EPE 用膨胀窗口筛选，未乘 epe_weight。
    int total_ep_count; // 实际评价的 EP 数，包含 Guard 重叠点和周期副本；唯一 Core 点数另存 core.ep_count。
    double epe_mean; // total_epe_loss / total_ep_count；不平均各 Micro 均值，单位与 EPE 相同。
};

struct FullChipLSMResult {
    int best_iteration;
    double best_loss; // 按全部初始/更新后状态中的最小总 loss 选择结果，相等时保留首次达到的状态。
    Eigen::MatrixXd global_mask; // 最佳轮实际用于评价的 mask（Binary/Gray），仅回拼 Core，尺寸为 meta.height × meta.width。
    Eigen::MatrixXd global_phi; // 同一最佳轮 phi 的 Core 回拼；拼接后的场不保证处处是严格 SDF。
    Eigen::MatrixXd initial_global_phi; // 各块初始化 SDF 的 Core 回拼，便于与最终状态比较。
    Eigen::MatrixXd initial_global_mask; // 第 0 轮实际渲染 mask 的 Core 回拼；Gray 时与输入的 CTM 二值图不同。
    std::vector<Eigen::MatrixXd> local_masks; // 最佳轮各块完整渲染 mask，含独立的 Guard，下标为 Micro id。
    std::vector<Eigen::MatrixXd> local_phi; // 与 local_masks 同一轮的完整 phi；mask 由 render_levelset_mask() 得到。
    std::vector<FullChipLSMIteration> history; // 初始及所有更新后的评价，共 iterations + 1 项。
    std::vector<CoreLossMetrics> best_core_metrics; // 最佳轮每个 Micro 的 Core 指标，与返回的 local_masks 同一轮。
};

class FullChipLSMOptimizer {
public:
    // 外部按 meta.tile_size 初始化 cache；meta/cache 须比本对象存活更久，线程工作缓存跨轮复用。
    FullChipLSMOptimizer(const PartitionMeta& meta, const ImagingCache& cache, double threshold = 0.25, double alpha = 85.0);

    // initial_binary 是原 C++ CTM 二值化后的全局 0/1 mask；global_target 是同尺寸的原始目标图。
    // 每轮在 Core + Guard 内按 loss_type 评价：PE 全区域，PEPenalty 原主图 PE+背景惩罚，PE_EPE 膨胀分区。
    // 按区域损失反传完整 Core + Halo 梯度；交换/融合后只使用 Core + Guard 梯度并更新这里的 phi。
    // 更新窗口外 phi 保持初始值，返回按总损失选择的历史最佳状态；CEPT 是单独的 Guard 一致性约束。
    // Gray 渲染先处理完整局部 phi，再回拼各块 Core；不在拼接后的 global_phi 上重新渲染。
    FullChipLSMResult optimize(const Eigen::MatrixXd& initial_binary, const Eigen::MatrixXd& global_target,
                              const FullChipLSMOptions& options = {});

private:
    const PartitionMeta& _meta;
    MicroGradientExchange _exchange; // 复用 CTM 阶段的逐块成像、原 C++ PE/EPE 梯度、CEPT 和四邻域交换。
};

} // namespace litho
