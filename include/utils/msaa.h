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
