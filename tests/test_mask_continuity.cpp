#include "mask_continuity.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace litho;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    const auto meta = Partition::build(12, 16, 2, 2, 2, 1);
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(12, 16);
    target.rightCols(8).setOnes();
    auto locals = Partition::extract_all(target, meta);
    const auto reference = evaluate_mask_continuity(meta, locals, target);
    // 接缝上恰好有目标的真实 0→1 边缘：原始跳变非零，但额外参考误差和同位置重叠差异均为零。
    require(reference.overlap_samples == 56 && reference.seam_samples == 28, "wrong directed overlaps or unique seam count");
    require(reference.overlap_sse == 0.0 && reference.seam_reference_error_mae == 0.0, "target edge mislabeled as inconsistency");
    require(std::abs(reference.seam_jump_mae - 12.0 / 28.0) < 1e-14, "wrong physical target jump");
    require(reference.flat_seam_samples == 16 && reference.foreground_seam_samples == 8, "wrong target-conditioned seam count");

    target.setZero();
    for (const auto& tile : meta.tiles) locals[tile.id].setConstant(tile.col);
    const auto broken = evaluate_mask_continuity(meta, locals, target);
    require(broken.overlap_sse == 24.0 && broken.overlap_max_abs == 1.0, "wrong independent Guard-Core differences");
    require(std::abs(broken.overlap_rmse - std::sqrt(24.0 / 56.0)) < 1e-14, "wrong overlap normalization");
    require(std::abs(broken.seam_reference_error_mae - 12.0 / 28.0) < 1e-14, "missed actual Core seam discontinuity");

    locals = Partition::extract_all(target, meta);
    const auto& guard = meta.tiles[0].neighbors[0].my_guard_local;
    locals[0].block(guard.y0, guard.x0, guard.rows(), guard.cols()).setConstant(0.8);
    const auto guard_only = evaluate_mask_continuity(meta, locals, target);
    require(std::abs(guard_only.overlap_sse - 3.84) < 1e-14, "wrong Guard-only overlap discrepancy");
    require(guard_only.seam_jump_mae == 0.0, "Guard discrepancy confused with assembled Core seam jump");
    std::cout << "Independent overlaps, unique seams and target-edge distinction passed.\n";
}
