#include "epe_utils.h"

#include <algorithm>
#include <cmath>
#include <utility>

namespace litho {

SobelGradientMaps sobel_gradient_maps(const Eigen::MatrixXd& field) {
    static constexpr double kernel_x[3][3] = {{-1.0, 0.0, 1.0}, {-2.0, 0.0, 2.0}, {-1.0, 0.0, 1.0}};
    static constexpr double kernel_y[3][3] = {{-1.0, -2.0, -1.0}, {0.0, 0.0, 0.0}, {1.0, 2.0, 1.0}};
    SobelGradientMaps result{Eigen::MatrixXd::Zero(field.rows(), field.cols()),
                             Eigen::MatrixXd::Zero(field.rows(), field.cols()),
                             Eigen::MatrixXd::Zero(field.rows(), field.cols())};
    // PyTorch conv2d 执行互相关而非翻转卷积核；越过图像的邻域值按 padding=1 视为 0。
    for (int y = 0; y < field.rows(); ++y) {
        for (int x = 0; x < field.cols(); ++x) {
            double gx = 0.0, gy = 0.0;
            for (int ky = -1; ky <= 1; ++ky) {
                for (int kx = -1; kx <= 1; ++kx) {
                    const int yy = y + ky, xx = x + kx;
                    if (yy < 0 || yy >= field.rows() || xx < 0 || xx >= field.cols()) continue;
                    gx += kernel_x[ky + 1][kx + 1] * field(yy, xx);
                    gy += kernel_y[ky + 1][kx + 1] * field(yy, xx);
                }
            }
            // constexpr double sobel_scale = 1.0 / 8.0;

            // gx *= sobel_scale;
            // gy *= sobel_scale;
            result.x(y, x) = gx;
            result.y(y, x) = gy;
            result.magnitude(y, x) = std::hypot(gx, gy);
        }
    }
    return result ;
}

BilinearStencil make_bilinear_stencil(int height, int width, double y, double x) {
    // y/x 是矩阵中的浮点行列索引；先限制到 [0,height-1]×[0,width-1] 的有效像素中心范围。
    const double yy = std::clamp(y, 0.0, static_cast<double>(height - 1));
    const double xx = std::clamp(x, 0.0, static_cast<double>(width - 1));
    // 找到亚像素点上下、左右相邻的整数像素索引，四种组合就是最多 4 个双线性支撑像素。
    const int y0 = static_cast<int>(std::floor(yy)), x0 = static_cast<int>(std::floor(xx));
    const int y1 = std::min(y0 + 1, height - 1), x1 = std::min(x0 + 1, width - 1);
    // wy0/wy1 和 wx0/wx1 分别表示点在两个方向上分配给相邻像素的比例，每组权重之和为 1。
    const double wy1 = y1 == y0 ? 0.0 : yy - y0, wx1 = x1 == x0 ? 0.0 : xx - x0;
    const double wy0 = 1.0 - wy1, wx0 = 1.0 - wx1;

    BilinearStencil stencil;
    // 二维权重等于行方向权重乘列方向权重；图像边缘出现相同像素索引时合并其权重。
    for (const auto [py, wy] : {std::pair{y0, wy0}, std::pair{y1, wy1}}) {
        for (const auto [px, wx] : {std::pair{x0, wx0}, std::pair{x1, wx1}}) {
            const double weight = wy * wx;
            if (weight <= 0.0) continue;
            int existing = -1;
            for (int i = 0; i < stencil.count; ++i) {
                if (stencil.entries[i].y == py && stencil.entries[i].x == px) existing = i;
            }
            if (existing >= 0) stencil.entries[existing].weight += weight;
            else stencil.entries[stencil.count++] = {py, px, weight};
        }
    }
    return stencil;
}

double bilinear_sample(const Eigen::MatrixXd& field, const BilinearStencil& stencil) {
    double value = 0.0;
    for (int i = 0; i < stencil.count; ++i) {
        const auto& entry = stencil.entries[i];
        value += entry.weight * field(entry.y, entry.x);
    }
    return value;
}

void bilinear_scatter(Eigen::MatrixXd& field, const BilinearStencil& stencil, double value) {
    for (int i = 0; i < stencil.count; ++i) {
        const auto& entry = stencil.entries[i];
        field(entry.y, entry.x) += entry.weight * value;
    }
}

bool bilinear_touches_window(const Eigen::MatrixXd& window, const BilinearStencil& stencil) {
    for (int i = 0; i < stencil.count; ++i) {
        const auto& entry = stencil.entries[i];
        if (window(entry.y, entry.x) != 0.0) return true;
    }
    return false;
}

} // namespace litho
