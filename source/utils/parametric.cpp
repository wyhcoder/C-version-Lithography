#include "parametric.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace litho {
namespace {

int wrapped_index(int index, int count) {
    return (index % count + count) % count;
}

double periodic_b_spline_control_influence(
    int count,
    int degree,
    int segment,
    double t,
    int control_index) {
    auto contribution = [&](int index, double weight) {
        return wrapped_index(index, count) == control_index ? weight : 0.0;
    };

    if (degree == 1) {
        return contribution(segment, 1.0 - t) +
               contribution(segment + 1, t);
    }
    if (degree == 2) {
        const double b0 = 0.5 * (1.0 - t) * (1.0 - t);
        const double b1 = 0.5 * (-2.0 * t * t + 2.0 * t + 1.0);
        const double b2 = 0.5 * t * t;
        return contribution(segment - 1, b0) +
               contribution(segment, b1) +
               contribution(segment + 1, b2);
    }

    const double t2 = t * t;
    const double t3 = t2 * t;
    const double b0 = (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0;
    const double b1 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0;
    const double b2 = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0;
    const double b3 = t3 / 6.0;
    return contribution(segment - 1, b0) +
           contribution(segment, b1) +
           contribution(segment + 1, b2) +
           contribution(segment + 2, b3);
}

double tent_kernel_value(double r) {
    const double value = 1.0 - std::abs(r);
    return value > 0.0 ? value : 0.0;
}

// 计算闭合周期均匀 B 样条上的一个点；控制点坐标格式为 (y,x)。
Eigen::Vector2d periodic_b_spline_point(
    const Polygon& controls,
    int degree,
    int segment,
    double t) {
    const int count = static_cast<int>(controls.rows());
    const auto point_at = [&](int index) {
        const int wrapped = wrapped_index(index, count);
        return controls.row(wrapped).transpose();
    };

    if (degree == 1) {
        return (1.0 - t) * point_at(segment) +
               t * point_at(segment + 1);
    }
    if (degree == 2) {
        const double b0 = 0.5 * (1.0 - t) * (1.0 - t);
        const double b1 = 0.5 * (-2.0 * t * t + 2.0 * t + 1.0);
        const double b2 = 0.5 * t * t;
        return b0 * point_at(segment - 1) +
               b1 * point_at(segment) +
               b2 * point_at(segment + 1);
    }

    const double t2 = t * t;
    const double t3 = t2 * t;
    const double b0 = (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0;
    const double b1 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0;
    const double b2 = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0;
    const double b3 = t3 / 6.0;
    return b0 * point_at(segment - 1) +
           b1 * point_at(segment) +
           b2 * point_at(segment + 1) +
           b3 * point_at(segment + 2);
}

}  // namespace

RasterDerivativeXY ParametricDemo::render_curve_dirac_derivative(
    const Polygons& cps,
    int contour_index,
    int control_point_index,
    int num_points,
    double grid_spacing,
    double segment_fraction) const
{
    if (_curve_type != "BS") {
        throw std::invalid_argument(
            "render_curve_dirac_derivative currently supports curve_type=BS only");
    }
    if (contour_index < 0 || contour_index >= static_cast<int>(cps.size())) {
        throw std::out_of_range("render_curve_dirac_derivative: contour index out of range");
    }
    const Polygon& controls = cps[contour_index];
    const int control_count = static_cast<int>(controls.rows());
    if (control_point_index < 0 || control_point_index >= control_count) {
        throw std::out_of_range(
            "render_curve_dirac_derivative: control-point index out of range");
    }
    if (control_count < 2 || num_points < 3) {
        throw std::invalid_argument(
            "render_curve_dirac_derivative: insufficient controls or curve samples");
    }
    if (!std::isfinite(grid_spacing) || grid_spacing <= 0.0 ||
        !std::isfinite(segment_fraction) || segment_fraction <= 0.0) {
        throw std::invalid_argument(
            "render_curve_dirac_derivative: invalid grid spacing or segment fraction");
    }

    const int height = static_cast<int>(_mask_template.rows());
    const int width = static_cast<int>(_mask_template.cols());
    RasterDerivativeXY result{
        Eigen::MatrixXd::Zero(height, width),
        Eigen::MatrixXd::Zero(height, width)};

    const Polygons sampled = b_spline({controls}, num_points);
    const Polygon& curve = sampled.front();
    const int degree = std::min(3, control_count - 1);

    std::vector<double> influence(num_points, 0.0);
    for (int sample = 0; sample < num_points; ++sample) {
        const double global_t =
            static_cast<double>(sample) / num_points * control_count;
        const int spline_segment = static_cast<int>(std::floor(global_t));
        const double local_t = global_t - spline_segment;
        influence[sample] = periodic_b_spline_control_influence(
            control_count,
            degree,
            spline_segment,
            local_t,
            control_point_index);
    }

    double twice_signed_area = 0.0;
    for (int sample = 0; sample < num_points; ++sample) {
        const int next = (sample + 1) % num_points;
        twice_signed_area +=
            curve(sample, 1) * curve(next, 0) -
            curve(next, 1) * curve(sample, 0);
    }
    if (std::abs(twice_signed_area) < 1e-12) {
        throw std::invalid_argument(
            "render_curve_dirac_derivative: sampled contour area is zero");
    }
    const bool counter_clockwise = twice_signed_area > 0.0;
    const double target_segment_length = segment_fraction * grid_spacing;

    // 形状导数：d chi / d p_(j,alpha)
    //          = integral_Gamma B_j(u) n_alpha delta_h(x-C(u)) ds。
    // B_j 是当前控制点对 B 样条曲线点的影响系数；这里只把边界积分
    // 离散化，不再重新生成 p+delta 和 p-delta 两张 mask。
    for (int edge = 0; edge < num_points; ++edge) {
        const int next = (edge + 1) % num_points;
        const double x0 = curve(edge, 1);
        const double y0 = curve(edge, 0);
        const double dx = curve(next, 1) - x0;
        const double dy = curve(next, 0) - y0;
        const double edge_length = std::hypot(dx, dy);
        if (edge_length <= 1e-14) continue;

        const int segment_count = std::max(
            1,
            static_cast<int>(std::ceil(edge_length / target_segment_length)));
        const double segment_length = edge_length / segment_count;
        const double tangent_x = dx / edge_length;
        const double tangent_y = dy / edge_length;
        const double normal_x = counter_clockwise ? tangent_y : -tangent_y;
        const double normal_y = counter_clockwise ? -tangent_x : tangent_x;
        const double scale =
            segment_length / (grid_spacing * grid_spacing);

        for (int segment = 0; segment < segment_count; ++segment) {
            const double fraction =
                (static_cast<double>(segment) + 0.5) / segment_count;
            const double basis =
                (1.0 - fraction) * influence[edge] +
                fraction * influence[next];
            if (std::abs(basis) <= 1e-15) continue;

            const double sample_x = x0 + fraction * dx;
            const double sample_y = y0 + fraction * dy;
            const double ux = sample_x / grid_spacing;
            const double uy = sample_y / grid_spacing;
            const int ix0 = static_cast<int>(std::floor(ux));
            const int iy0 = static_cast<int>(std::floor(uy));

            for (int row : {iy0, iy0 + 1}) {
                if (row < 0 || row >= height) continue;
                const double wy = tent_kernel_value(uy - row);
                if (wy == 0.0) continue;
                for (int col : {ix0, ix0 + 1}) {
                    if (col < 0 || col >= width) continue;
                    const double weight = tent_kernel_value(ux - col) * wy;
                    result.dx(row, col) +=
                        scale * basis * normal_x * weight;
                    result.dy(row, col) +=
                        scale * basis * normal_y * weight;
                }
            }
        }
    }

    return result;
}

ParametricDemo::ParametricDemo(const std::string& curve_type,
                               const Eigen::MatrixXd& mask_template,
                               int msaa_level,
                               const std::string& rasterizer)
    : _curve_type(curve_type),
      _rasterizer(rasterizer),
      _mask_template(mask_template),
      _renderer(msaa_level)
{
    if (_rasterizer != "msaa" && _rasterizer != "dirac") {
        throw std::invalid_argument(
            "ParametricDemo: rasterizer must be msaa or dirac");
    }
}

// ── render_curve ──────────────────────────────────────────────────────────
Eigen::MatrixXd ParametricDemo::render_curve(const Polygons& cps,
                                             int num_points) const {
    Polygons pts = get_curve_points(cps, num_points);
    if (_rasterizer == "dirac") {
        return _renderer.rasterize_dirac_indicator(
            pts, _mask_template, "gray", 1.0, 0.25);
    }
    return _renderer.MSAA(pts, _mask_template, "gray");
}

Eigen::MatrixXd ParametricDemo::render_curve_dirac(
    const Polygons& cps,
    int num_points,
    double grid_spacing,
    double segment_fraction) const
{
    Polygons pts = get_curve_points(cps, num_points);
    return _renderer.rasterize_dirac_indicator(
        pts,
        _mask_template,
        "gray",
        grid_spacing,
        segment_fraction);
}

// ── get_curve_points ──────────────────────────────────────────────────────
Polygons ParametricDemo::get_curve_points(const Polygons& cps,
                                           int num_points) const {
    if (_curve_type == "OA") return cps;
    if (_curve_type == "BZ") return compute_BZ_cps_and_points(cps);
    if (_curve_type == "BS") return b_spline(cps, num_points);
    return cps;
}

// ── 周期均匀 B 样条 ──────────────────────────────────────────────────────
// 主图形轮廓是闭合的；输入点直接作为控制点，中间控制点不要求位于曲线上。
Polygons ParametricDemo::b_spline(const Polygons& contours,
                                   int num_points) const {
    Polygons result;
    for (const auto& contour : contours) {
        const int point_count = static_cast<int>(contour.rows());
        if (point_count < 2) {
            result.push_back(contour);
            continue;
        }

        Eigen::MatrixXd fitted(num_points, 2);
        const int degree = std::min(3, point_count - 1);

        for (int k = 0; k < num_points; ++k) {
            const double global_t =
                static_cast<double>(k) / num_points * point_count;
            const int segment = static_cast<int>(std::floor(global_t));
            const double local_t = global_t - segment;
            fitted.row(k) = periodic_b_spline_point(
                contour, degree, segment, local_t).transpose();
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
