#pragma once

#include "partition.h"

#include <cstddef>

namespace litho {

struct MaskContinuityMetrics {
    std::size_t overlap_samples = 0; // 有向 Guard→邻居 Core 的同位置像素对数；包括两个方向，不包括角区。
    double overlap_sse = 0.0; // Σ(A_Guard-B_Core)²；无论是否启用 CEPT 都独立统计。
    double overlap_rmse = 0.0, overlap_mae = 0.0, overlap_max_abs = 0.0; // mask 值的无量纲差异。
    std::size_t seam_samples = 0, flat_seam_samples = 0, foreground_seam_samples = 0;
    double seam_jump_mae = 0.0; // Core 回拼边界两侧相邻像素的平均 |Δmask|，含目标本身的真实边缘。
    double seam_reference_error_mae = 0.0; // 平均 |Δmask-Δtarget|，用于区分目标变化与额外拼接跳变。
    double flat_seam_jump_mae = 0.0; // 仅 target 两侧完全相同的接缝像素对；背景与前景均计入。
    double foreground_seam_jump_mae = 0.0; // 仅 target 两侧均 >0.5 的接缝像素对。
};

// 输入最佳或末轮的完整局部 mask 和全局 target，分别检查同位置重叠一致性与实际 Core 接缝跳变。
// 所有坐标来自 PartitionMeta；Core 接缝只遍历右/下邻居一次，芯片周期外边界不作为逻辑接缝。
// 指标与 enable_cept/enable_grad_fusion 无关，不调用优化器或改变 mask。
MaskContinuityMetrics evaluate_mask_continuity(const PartitionMeta& meta, const std::vector<Eigen::MatrixXd>& local_masks,
                                               const Eigen::MatrixXd& target);

} // namespace litho
