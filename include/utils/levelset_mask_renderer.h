#pragma once

#include <Eigen/Dense>

namespace litho {

enum class LevelSetMaskType {
    Binary, // 原始像素中心处按 phi >= 0 得到 0/1 mask。
    Gray // 在子像素网格二值化后取块平均，灰度表示图形在原像素内的覆盖率。
};

// 输入完整局部 phi（行 y、列 x，图形内部为正）；输出同尺寸、范围 [0,1] 的成像 mask。
// Gray 模式与 Python 一致：float32 phi → 双线性放大 → phi >= 0 → 每块 scale² 个子像素取平均。
// scale 是每个方向的正整数放大倍数，不改变光学 FFT 网格或像素物理尺寸；Binary 模式不使用 scale。
Eigen::MatrixXd render_levelset_mask(const Eigen::MatrixXd& phi, LevelSetMaskType type, int scale = 4);

} // namespace litho
