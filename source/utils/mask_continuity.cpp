#include "mask_continuity.h"

#include <algorithm>
#include <cmath>

namespace litho {

MaskContinuityMetrics evaluate_mask_continuity(const PartitionMeta& meta, const std::vector<Eigen::MatrixXd>& local_masks,
                                               const Eigen::MatrixXd& target) {
    MaskContinuityMetrics result;
    double overlap_abs = 0.0;
    const Eigen::MatrixXd global = Partition::assemble_global(local_masks, meta);
    auto add_seam = [&](int ay, int ax, int by, int bx) {
        const double jump = global(by, bx) - global(ay, ax);
        const double reference = target(by, bx) - target(ay, ax);
        ++result.seam_samples;
        result.seam_jump_mae += std::abs(jump);
        result.seam_reference_error_mae += std::abs(jump - reference);
        if (reference == 0.0) {
            ++result.flat_seam_samples;
            result.flat_seam_jump_mae += std::abs(jump);
        }
        if (target(ay, ax) > 0.5 && target(by, bx) > 0.5) {
            ++result.foreground_seam_samples;
            result.foreground_seam_jump_mae += std::abs(jump);
        }
    };
    for (const auto& tile : meta.tiles) {
        for (const auto& edge : tile.neighbors) {
            const auto& a = edge.my_guard_local;
            const auto& b = edge.neighbor_overlap_local;
            const Eigen::MatrixXd difference = local_masks[tile.id].block(a.y0, a.x0, a.rows(), a.cols()) -
                local_masks[edge.id].block(b.y0, b.x0, b.rows(), b.cols());
            result.overlap_samples += difference.size();
            result.overlap_sse += difference.squaredNorm();
            overlap_abs += difference.cwiseAbs().sum();
            if (difference.size() > 0) result.overlap_max_abs = std::max(result.overlap_max_abs, difference.cwiseAbs().maxCoeff());
            // 相邻 Core 的公共边各评价一次；接缝位于 x-0.5 或 y-0.5，取它两侧的实际像素中心。
            const auto& neighbor = meta.tiles[edge.id];
            if (edge.direction == NeighborDirection::Right) {
                const int x = neighbor.core_global.x0;
                for (int y = tile.core_global.y0; y < tile.core_global.y1; ++y) add_seam(y, x - 1, y, x);
            } else if (edge.direction == NeighborDirection::Bottom) {
                const int y = neighbor.core_global.y0;
                for (int x = tile.core_global.x0; x < tile.core_global.x1; ++x) add_seam(y - 1, x, y, x);
            }
        }
    }
    // 零 Guard、单 Micro 或没有前景穿越接缝时，相应集合为空，保留零值并同时导出像素对数。
    if (result.overlap_samples > 0) {
        result.overlap_rmse = std::sqrt(result.overlap_sse / result.overlap_samples);
        result.overlap_mae = overlap_abs / result.overlap_samples;
    }
    if (result.seam_samples > 0) {
        result.seam_jump_mae /= result.seam_samples;
        result.seam_reference_error_mae /= result.seam_samples;
    }
    if (result.flat_seam_samples > 0) result.flat_seam_jump_mae /= result.flat_seam_samples;
    if (result.foreground_seam_samples > 0) result.foreground_seam_jump_mae /= result.foreground_seam_samples;
    return result;
}

} // namespace litho
