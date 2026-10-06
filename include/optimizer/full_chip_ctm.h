#pragma once

#include <Eigen/Dense>
#include <vector>

#include "micro_gradient_exchange.h"

namespace litho {

struct FullChipCTMOptions {
    int iterations = 50; // 固定执行的 theta 更新次数；0 表示只初始化并评价。
    double learning_rate = 0.9; // theta 域固定步长，每个 Micro、每一轮使用相同值，不做线搜索。
    MicroGradientOptions gradient_options; // 复用现有 PE/EPE、CEPT、Guard 梯度融合选项。
    bool parallel_micro = true; // 控制 Micro 成像/反传与 theta 更新的并行；信息交换及 Halo 刷新始终串行。
    bool verbose = true; // 输出损失配置及每轮数据项、加权 CEPT 和 theta 固定步长。
};

struct FullChipCTMIteration {
    int iteration; // 已完成的更新次数；0 是初始状态，最后一项为 options.iterations。
    double loss; // 该状态的 data_loss + cept_weight * cept_loss。
    double data_loss; // 各 Micro 指定区域的数据损失之和；默认 Core，可设置为 Core + Guard。
    double cept_loss; // 该状态中各块 Guard 与邻居 Core 的像素平方差之和，未乘权重。
    double pe_contribution, penalty_contribution, epe_contribution, cept_contribution; // 各项已加权贡献，未启用的项为 0。
    // data_loss = 普通 PE 贡献 + 背景惩罚贡献 + EPE 贡献；loss = data_loss + CEPT 贡献。
};

struct FullChipCTMResult {
    int best_iteration; // 总 loss 最小的迭代编号，初始状态也参与比较；相等时保留首次达到最小值的状态。
    double best_loss; // 历史最小总 loss，包含所选数据项及加权 CEPT。
    Eigen::MatrixXd global_mask; // 最佳轮灰度 mask，由同一轮各块 Core 回拼，尺寸为 meta.height × meta.width。
    std::vector<Eigen::MatrixXd> local_masks; // 最佳轮完整局部 mask，含各块独立的 Guard；下标为 Micro id。
    std::vector<Eigen::MatrixXd> local_theta; // 同一最佳轮的 theta，与 local_masks 一一对应，mask = 0.5 * (1 + cos(theta))。
    std::vector<FullChipCTMIteration> history; // 初始状态加每次更新后的评价，共 iterations + 1 项。
};

class FullChipCTMOptimizer {
public:
    // meta/cache 在外部准备并须比本对象存活更久；cache.N == meta.tile_size，工作对象及 FFT 计划跨轮复用。
    FullChipCTMOptimizer(const PartitionMeta& meta, const ImagingCache& cache, double threshold = 0.25, double alpha = 85.0);

    // 从全局目标图形初始化各块 theta，固定步长优化指定次数；更新 Core + Guard，返回总 loss 最小的状态。
    FullChipCTMResult optimize(const Eigen::MatrixXd& global_target, const FullChipCTMOptions& options = {});

    // initial_pattern 与 target 均为全局尺寸；初始化沿用 CTM 的非零/零像素映射，可指定不同初始图形。
    FullChipCTMResult optimize(const Eigen::MatrixXd& initial_pattern, const Eigen::MatrixXd& global_target,
                              const FullChipCTMOptions& options = {});

private:
    // 用最新 Core 回拼图周期提取局部上下文，仅刷新 Core + Guard 之外的 Halo，同步 theta 与 mask。
    void _refresh_outer_halo(FullChipCTMResult& state);
    const PartitionMeta& _meta; // 全局/局部坐标、Core + Guard 更新窗口和四邻域关系。
    MicroGradientExchange _exchange; // 并行成像/梯度，等待完成后串行交换、CEPT 与可选融合。
};

} // namespace litho
