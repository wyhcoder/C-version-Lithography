#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include "msaa.h"

namespace litho {

// ── Bezier 切线结构 ──────────────────────────────────────────────────────
struct PointTangent {
    Eigen::Vector2d point;    // (y, x)
    Eigen::Vector2d tangent;  // 单位切向量 (ty, tx)
};

struct ContourBezierData {
    Eigen::MatrixXd               data_points;      // 原始控制点 [N x 2]
    std::vector<std::array<Eigen::Vector2d,4>> control_points; // 每段 [p0,p1,p2,p3]
    int image_height;
    int image_width;
};

// ── 参数化曲线采样结果 ────────────────────────────────────────────────────
using CurvePoints = Polygon;  // [N x 2] (y,x)

class ParametricDemo {
public:
    // curve_type: "OA"（直接折线）| "BZ"（三次 Bezier）| "BS"（B样条）
    ParametricDemo(const std::string& curve_type,
                   const Eigen::MatrixXd& mask_template,
                   int msaa_level = 16);

    // 根据控制点生成 mask
    Eigen::MatrixXd render_curve(const Polygons& cps) const;

    // 只返回曲线点（不光栅化）
    Polygons get_curve_points(const Polygons& cps, int num_points = 200) const;

    // ── 周期均匀 B 样条 ──────────────────────────────────────────────────
    // 每个输入轮廓按闭合曲线处理，输入点直接作为 B 样条控制点。
    Polygons b_spline(const Polygons& contours,
                      int num_points = 100) const;

    // ── Bezier ──────────────────────────────────────────────────────────
    std::vector<PointTangent> generate_bezier_tangent(
        const Polygon& contour) const;

    CurvePoints cubic_bezier(const Eigen::Vector2d& p0, const Eigen::Vector2d& p1,
                             const Eigen::Vector2d& p2, const Eigen::Vector2d& p3,
                             int num = 10) const;

    Polygons generate_BZ_points(
        const std::vector<ContourBezierData>& contours) const;

    Polygons compute_BZ_cps_and_points(const Polygons& initial_cps) const;

private:
    std::string        _curve_type;
    Eigen::MatrixXd    _mask_template;
    AntiAliasRenderer  _renderer;
};

}  // namespace litho
