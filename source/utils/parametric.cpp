#include "parametric.h"
#include <cmath>
#include <stdexcept>

namespace litho {

ParametricDemo::ParametricDemo(const std::string& curve_type,
                               const Eigen::MatrixXd& mask_template,
                               int msaa_level)
    : _curve_type(curve_type),
      _mask_template(mask_template),
      _renderer(msaa_level)
{}

// ── render_curve ──────────────────────────────────────────────────────────
Eigen::MatrixXd ParametricDemo::render_curve(const Polygons& cps) const {
    Polygons pts = get_curve_points(cps, 200);
    return _renderer.MSAA(pts, _mask_template, "gray");
}

// ── get_curve_points ──────────────────────────────────────────────────────
Polygons ParametricDemo::get_curve_points(const Polygons& cps,
                                           int num_points) const {
    if (_curve_type == "OA") return cps;
    if (_curve_type == "BZ") return compute_BZ_cps_and_points(cps);
    if (_curve_type == "BS") return b_spline(cps, 0.3, num_points);
    return cps;
}

// ── Catmull-Rom B 样条（周期性，替代 scipy.splprep per=True）──────────────
// scipy 的 s 参数控制平滑量；这里 smoothing 参数没有直接等价物，
// Catmull-Rom 通过 tension 参数（默认 0.5）控制曲线紧张度。
Polygons ParametricDemo::b_spline(const Polygons& contours,
                                   double /*smoothing*/,
                                   int num_points) const {
    Polygons result;
    for (const auto& contour : contours) {
        int n = static_cast<int>(contour.rows());
        if (n < 2) { result.push_back(contour); continue; }

        Eigen::MatrixXd fitted(num_points, 2);
        double tension = 0.5;   // Catmull-Rom 标准张力

        for (int k = 0; k < num_points; ++k) {
            // t ∈ [0, n)，周期映射
            double global_t = static_cast<double>(k) / num_points * n;
            int seg = static_cast<int>(std::floor(global_t)) % n;
            double t = global_t - std::floor(global_t);

            // 4个控制点（周期索引）
            auto P = [&](int i) -> Eigen::Vector2d {
                return contour.row(((i % n) + n) % n).transpose();
            };
            Eigen::Vector2d p0 = P(seg - 1);
            Eigen::Vector2d p1 = P(seg);
            Eigen::Vector2d p2 = P(seg + 1);
            Eigen::Vector2d p3 = P(seg + 2);

            // Catmull-Rom 矩阵计算
            double t2 = t * t, t3 = t2 * t;
            Eigen::Vector2d pt =
                0.5 * ((2*p1) +
                       (-p0 + p2) * t +
                       (2*p0 - 5*p1 + 4*p2 - p3) * t2 +
                       (-p0 + 3*p1 - 3*p2 + p3) * t3);
            // tension 参数缩放切线（标准 Catmull-Rom 不用，这里用于兼容）
            (void)tension;

            fitted.row(k) = pt.transpose();
        }
        result.push_back(fitted);
    }
    return result;
}

// ── 切线计算（对应 generate_bezier_tangent）──────────────────────────────
std::vector<PointTangent> ParametricDemo::generate_bezier_tangent(
    const Polygon& contour) const
{
    int n = static_cast<int>(contour.rows());
    // 转成 (x, y) 方便向量运算，最后再转回 (y, x)
    // 原 Python：approx_xy = approx[:, [1, 0]]，即列 [x, y]
    Eigen::MatrixXd xy(n, 2);
    for (int i = 0; i < n; ++i) {
        xy(i, 0) = contour(i, 1);  // x
        xy(i, 1) = contour(i, 0);  // y
    }

    std::vector<PointTangent> result(n);
    for (int i = 0; i < n; ++i) {
        Eigen::Vector2d B = xy.row(i).transpose();
        Eigen::Vector2d A = xy.row((i - 1 + n) % n).transpose();
        Eigen::Vector2d C = xy.row((i + 1) % n).transpose();

        double na = (A - B).norm() + 1e-8;
        double nc = (C - B).norm() + 1e-8;
        Eigen::Vector2d v_prev = (A - B) / na;
        Eigen::Vector2d v_next = (C - B) / nc;

        Eigen::Vector2d bisector = v_prev + v_next;
        // 逆时针旋转 90°：[dx, dy] → [-dy, dx]
        Eigen::Vector2d rotated(-bisector(1), bisector(0));

        Eigen::Vector2d tangent;
        double norm = rotated.norm();
        if (norm >= 1e-6) {
            tangent = rotated / norm;
        } else {
            tangent = (v_next.norm() > 1e-6) ? v_next.normalized()
                                              : v_prev.normalized();
        }
        // 确保与后向向量同向
        if (tangent.dot(v_next) < 0) tangent = -tangent;

        // 转回 (y, x)
        result[i].point   = { contour(i, 0), contour(i, 1) };
        result[i].tangent = { tangent(1), tangent(0) };  // ty, tx
    }
    return result;
}

// ── 三次 Bezier 曲线采样 ─────────────────────────────────────────────────
CurvePoints ParametricDemo::cubic_bezier(
    const Eigen::Vector2d& p0, const Eigen::Vector2d& p1,
    const Eigen::Vector2d& p2, const Eigen::Vector2d& p3,
    int num) const
{
    CurvePoints pts(num, 2);
    for (int i = 0; i < num; ++i) {
        double t  = static_cast<double>(i) / (num - 1);
        double t1 = 1.0 - t;
        Eigen::Vector2d p =
            t1*t1*t1 * p0 +
            3*t*t1*t1 * p1 +
            3*t*t*t1  * p2 +
            t*t*t     * p3;
        pts.row(i) = p.transpose();
    }
    return pts;
}

// ── generate_BZ_points ───────────────────────────────────────────────────
Polygons ParametricDemo::generate_BZ_points(
    const std::vector<ContourBezierData>& contours) const
{
    Polygons result;
    for (const auto& cd : contours) {
        std::vector<CurvePoints> segs;
        for (const auto& cp : cd.control_points) {
            segs.push_back(cubic_bezier(cp[0], cp[1], cp[2], cp[3], 10));
        }
        // vstack
        int total = 0;
        for (auto& s : segs) total += s.rows();
        Eigen::MatrixXd poly(total, 2);
        int row = 0;
        for (auto& s : segs) {
            poly.block(row, 0, s.rows(), 2) = s;
            row += s.rows();
        }
        result.push_back(poly);
    }
    return result;
}

// ── compute_BZ_cps_and_points ─────────────────────────────────────────────
Polygons ParametricDemo::compute_BZ_cps_and_points(
    const Polygons& initial_cps) const
{
    std::vector<ContourBezierData> contour_data;
    int H = static_cast<int>(_mask_template.rows());
    int W = static_cast<int>(_mask_template.cols());

    for (const auto& contour : initial_cps) {
        auto pt_list = generate_bezier_tangent(contour);
        int n = static_cast<int>(pt_list.size());

        ContourBezierData cd;
        cd.data_points   = contour;
        cd.image_height  = H;
        cd.image_width   = W;

        for (int i = 0; i < n; ++i) {
            Eigen::Vector2d p0 = pt_list[i].point;
            Eigen::Vector2d t0 = pt_list[i].tangent;
            Eigen::Vector2d p3 = pt_list[(i+1) % n].point;
            Eigen::Vector2d t3 = pt_list[(i+1) % n].tangent;

            double d = (p3 - p0).norm();
            double alpha = d * 0.2;
            double beta  = d * 0.2;

            Eigen::Vector2d p1 = p0 + alpha * t0;
            Eigen::Vector2d p2 = p3 - beta  * t3;

            cd.control_points.push_back({p0, p1, p2, p3});
        }
        contour_data.push_back(cd);
    }

    return generate_BZ_points(contour_data);
}

}  // namespace litho
