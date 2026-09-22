#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>

namespace litho {

// 多边形用 [N x 2] 矩阵表示，每行 (y, x)
using Polygon  = Eigen::MatrixXd;
using Polygons = std::vector<Polygon>;

class AntiAliasRenderer {
public:
    explicit AntiAliasRenderer(int msaa_level = 4);

    // 主入口：对一组多边形进行 MSAA 光栅化
    //   type="gray"   → 返回 [H x W] float，值 ∈ [0,1]
    //   type="binary" → 返回 [H x W] float，值 ∈ {0,1}（奇偶填充）
    Eigen::MatrixXd MSAA(const Polygons& polygons,
                         const Eigen::MatrixXd& mask_template,
                         const std::string& type = "gray") const;

    // 在每个子采样点对所有轮廓做奇偶填充，再计算灰度覆盖率；用于带孔图形。
    Eigen::MatrixXd MSAA_even_odd(const Polygons& polygons, const Eigen::MatrixXd& mask_template) const;

    // 论文 JOLT-D-26-04401 第 2.2 节、公式 (6)-(8) 的指示函数重构：
    // 1. 将边界细分到约 segment_fraction * grid_spacing；
    // 2. 用紧支撑三角核把 -n*delta_Gamma 散布到网格；
    // 3. 分别沿 x/y 积分，平均并裁剪到 [0,1]。
    //
    // polygons 与现有 MSAA 一样使用 (y,x) 坐标；当前工程的曲线坐标
    // 以像素为单位，因此默认 grid_spacing=1.0。论文取 segment_fraction=0.25。
    // 该方法要求每条闭合轮廓完整位于计算域内部，因为计算域边界的
    // 指示函数初值被设为 0。
    Eigen::MatrixXd rasterize_dirac_indicator(
        const Polygons& polygons,
        const Eigen::MatrixXd& mask_template,
        const std::string& type = "gray",
        double grid_spacing = 1.0,
        double segment_fraction = 0.25) const;

    // 单多边形光栅化
    Eigen::MatrixXd rasterize_polygon(const Polygon& polygon,
                                      int height, int width,
                                      const std::string& type = "gray") const;

    // 点是否在多边形内（Ray Casting）
    static Eigen::VectorXi is_inside(const Polygon& polygon,
                                     const Eigen::MatrixXd& points);

    const Eigen::MatrixXd& sample_offsets() const { return _offsets; }
    int num_samples() const { return static_cast<int>(_offsets.rows()); }

private:
    Eigen::MatrixXd _offsets;  // [S x 2]，每行 (dy, dx)

    static Eigen::MatrixXd _make_offsets_4x();
    static Eigen::MatrixXd _make_offsets_16x();
    static Eigen::MatrixXd _make_offsets_8x8_uniform();
};

}  // namespace litho
