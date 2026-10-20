#pragma once

#include <Eigen/Dense>
#include <array>

namespace litho {

inline constexpr double kPytorchEpeEpsilon = 1e-7; // epe.py 加在插值后 Sobel 幅值上的稳定项。

// 一个亚像素坐标对某个像素中心的双线性插值权重；y/x 是矩阵行列索引，均从 0 开始。
struct BilinearEntry {
    int y, x;
    double weight;
};

// 一个亚像素点最多由周围 4 个像素中心支撑；图像边缘的重复像素会合并，因此 count 可能小于 4。
struct BilinearStencil {
    std::array<BilinearEntry, 4> entries{};
    int count = 0;
};

// 对齐 epe.py 的未归一化 3×3 Sobel 空间梯度；x/y 分别沿矩阵列/行方向，边界使用一圈零填充。
struct SobelGradientMaps {
    Eigen::MatrixXd x;
    Eigen::MatrixXd y;
    Eigen::MatrixXd magnitude;
};

SobelGradientMaps sobel_gradient_maps(const Eigen::MatrixXd& field);

// 输入浮点像素坐标 (y,x) 和矩阵尺寸；坐标先裁到有效像素中心范围，再返回双线性支撑像素及权重。
BilinearStencil make_bilinear_stencil(int height, int width, double y, double x);

// 使用已经构造的 stencil 对二维场进行双线性采样。
double bilinear_sample(const Eigen::MatrixXd& field, const BilinearStencil& stencil);

// 将亚像素点上的标量按同一组双线性权重累加回二维场，用于 EPE 空中像梯度散射。
void bilinear_scatter(Eigen::MatrixXd& field, const BilinearStencil& stencil, double value);

// 只要亚像素点的任一支撑像素位于非零窗口内，就认为该点与窗口相交。
bool bilinear_touches_window(const Eigen::MatrixXd& window, const BilinearStencil& stencil);

} // namespace litho
