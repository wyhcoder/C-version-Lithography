#pragma once

#include "micro_gradient_exchange.h"

namespace litho {

struct FullChipLSMOptions {
    int iterations = 200; // phi 更新次数；0 表示只初始化和评价，history 始终包含初始状态。
    double dx = 1.0, dy = 1.0; // 水平集网格步长；phi 的长度单位与步长一致，现有 SDF 初始化要求 dx == dy。
    double cfl = 0.7; // 各 Micro 自己计算 CFL 时间步长，不使用 CTM 的固定 learning_rate。
    double curvature_weight = 0.01; // 原 C++ evolve_kappa() 的 b，控制曲率平滑。
    int reinit_interval = 20; // 每完成多少次更新重新计算 SDF；0 表示关闭。
    MicroGradientOptions gradient_options; // 沿用损失及融合选项；LSM 固定 CoreGuard 损失区域和 cept_guard_only=true。
    bool parallel_micro = true; // Micro 成像/反传、SDF 初始化和 PDE 更新并行；四邻域交换串行。
    bool verbose = true;
};

struct FullChipLSMIteration {
    int iteration; // 已完成的更新次数；0 是 CTM 二值结果初始化得到的状态。
    double loss, data_loss, cept_loss; // 总损失、Core + Guard 数据损失、未加权 CEPT；loss = data_loss + 加权 CEPT。
    double min_dt, max_dt; // 本次更新中各 Micro CFL 步长的最小/最大值；初始状态为 0。
    double pe_contribution, penalty_contribution, epe_contribution, cept_contribution; // 各项已加权贡献，未启用的项为 0。
    // data_loss = 普通 PE 贡献 + 背景惩罚贡献 + EPE 贡献；loss = data_loss + CEPT 贡献。
};

struct FullChipLSMResult {
    int best_iteration;
    double best_loss; // 按全部初始/更新后状态中的最小总 loss 选择结果，相等时保留首次达到的状态。
    Eigen::MatrixXd global_mask; // 最佳轮二值 mask，只回拼 Core，尺寸为 meta.height × meta.width。
    Eigen::MatrixXd global_phi; // 同一最佳轮 phi 的 Core 回拼；拼接后的场不保证处处是严格 SDF。
    Eigen::MatrixXd initial_global_phi; // 各块初始化 SDF 的 Core 回拼，便于与最终状态比较。
    std::vector<Eigen::MatrixXd> local_masks; // 最佳轮各块完整二值 mask，含独立的 Guard，下标为 Micro id。
    std::vector<Eigen::MatrixXd> local_phi; // 与 local_masks 同一轮的完整 phi，mask = (phi >= 0)。
    std::vector<FullChipLSMIteration> history; // 初始及所有更新后的评价，共 iterations + 1 项。
};

class FullChipLSMOptimizer {
public:
    // 外部按 meta.tile_size 初始化 cache；meta/cache 须比本对象存活更久，线程工作缓存跨轮复用。
    FullChipLSMOptimizer(const PartitionMeta& meta, const ImagingCache& cache, double threshold = 0.25, double alpha = 85.0);

    // initial_binary 是原 C++ CTM 二值化后的全局 0/1 mask；global_target 是同尺寸的原始目标图。
    // 每轮交换/融合梯度后并行更新 Core + Guard，窗口外 phi 保持初始值；评价并记录历史最佳状态。
    FullChipLSMResult optimize(const Eigen::MatrixXd& initial_binary, const Eigen::MatrixXd& global_target,
                              const FullChipLSMOptions& options = {});

private:
    const PartitionMeta& _meta;
    MicroGradientExchange _exchange; // 复用 CTM 阶段的逐块成像、原 C++ PE/EPE 梯度、CEPT 和四邻域交换。
};

} // namespace litho
