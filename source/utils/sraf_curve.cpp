#include "sraf_curve.h"

#include <unsupported/Eigen/Splines>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace litho {
namespace {

/**
 * @brief 计算周期均匀 B 样条上指定参数位置的坐标。
 *
 * 根据控制点数量自动支持一次、二次和三次 B 样条。控制点下标采用周期循环，
 * 因此最后一段能够平滑连接回第一段。
 *
 * @param controls 按骨架方向排列的 B 样条控制点，坐标格式为 (x,y)。
 * @param degree B 样条次数，当前支持 1、2、3。
 * @param segment 当前曲线段编号。
 * @param t 当前曲线段内的局部参数，范围为 [0,1)。
 * @return 周期 B 样条在该参数位置的二维坐标。
 */
cv::Point2d periodic_b_spline_point(
    const std::vector<cv::Point2d>& controls,
    int degree,
    int segment,
    double t) {
    const int count = static_cast<int>(controls.size());
    const auto point_at = [&](int index) -> cv::Point2d {
        const int wrapped = (index % count + count) % count;
        return controls[static_cast<std::size_t>(wrapped)];
    };
    if (degree == 1) return (1.0 - t) * point_at(segment) + t * point_at(segment + 1);
    if (degree == 2) {
        const double b0 = 0.5 * (1.0 - t) * (1.0 - t);
        const double b1 = 0.5 * (-2.0 * t * t + 2.0 * t + 1.0);
        const double b2 = 0.5 * t * t;
        return b0 * point_at(segment - 1) + b1 * point_at(segment) + b2 * point_at(segment + 1);
    }

    const double t2 = t * t;
    const double t3 = t2 * t;
    const double b0 = (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0;
    const double b1 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0;
    const double b2 = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0;
    const double b3 = t3 / 6.0;
    return b0 * point_at(segment - 1) + b1 * point_at(segment) +
           b2 * point_at(segment + 1) + b3 * point_at(segment + 2);
}

/**
 * @brief 计算一个二维点到有限线段的最短欧氏距离。
 *
 * 先把点投影到线段所在直线，再把投影参数限制到 [0,1]。如果线段退化为
 * 一个点，则直接返回到该点的距离。
 *
 * @param point 等待计算距离的二维点。
 * @param begin 线段起点。
 * @param end 线段终点。
 * @return point 到线段 begin-end 的最短距离，单位为像素。
 */
double distance_to_segment(
    const cv::Point2d& point,
    const cv::Point2d& begin,
    const cv::Point2d& end) {
    const cv::Point2d direction = end - begin;
    const double length_squared = direction.dot(direction);
    if (length_squared <= 1e-12) return cv::norm(point - begin);

    const double projection = (point - begin).dot(direction) / length_squared;
    const double t = std::clamp(projection, 0.0, 1.0);
    return cv::norm(point - (begin + direction * t));
}

/**
 * @brief 计算一个二维点到整条 SRAF 中心曲线的最短距离。
 *
 * 曲线已经被密集采样为有序折线，因此逐段计算距离并取最小值。闭合曲线会
 * 额外计算最后一个采样点到第一个采样点之间的闭合线段。
 *
 * @param point 等待计算距离的像素或子像素坐标。
 * @param curve 拟合并密集采样后的 SRAF 参数曲线。
 * @return point 到曲线的最短距离，单位为像素。
 */
double distance_to_curve(
    const cv::Point2d& point,
    const SrafParametricCurve& curve) {
    if (curve.points.size() == 1) return cv::norm(point - curve.points.front());

    double minimum = std::numeric_limits<double>::infinity();
    for (std::size_t i = 1; i < curve.points.size(); ++i) {
        minimum = std::min(
            minimum,
            distance_to_segment(point, curve.points[i - 1], curve.points[i]));
    }
    if (curve.closed) {
        minimum = std::min(
            minimum,
            distance_to_segment(point, curve.points.back(), curve.points.front()));
    }
    return minimum;
}

}  // namespace

/**
 * @brief 用骨架间隔点直接构造一根 B 样条中心线。
 *
 * 输入点直接作为 B 样条控制点，不再求解另一组插值控制点。开放曲线使用
 * 夹持节点，因此经过首尾控制点；闭合曲线使用周期节点，首尾平滑连接。
 *
 * @param centerline 单根 SRAF 的有序骨架、开放/闭合标志和间隔控制点。
 * @param sample_spacing 相邻密集曲线点的大致间隔，单位为像素，必须大于 0。
 * @return B 样条次数、原始控制点和密集曲线点。
 */
SrafParametricCurve SrafCurve::fit(
    const SrafCenterlineGeometry& centerline,
    double sample_spacing) {
    if (!centerline.path.valid) {
        throw std::invalid_argument("SrafCurve::fit: centerline path is invalid");
    }
    if (sample_spacing <= 0.0) {
        throw std::invalid_argument("SrafCurve::fit: sample spacing must be positive");
    }
    if (centerline.control_points.empty()) {
        throw std::invalid_argument("SrafCurve::fit: control points are empty");
    }

    SrafParametricCurve curve;
    curve.component_id = centerline.component_id;
    curve.closed = centerline.path.closed;
    curve.spline_control_points = centerline.control_points;
    const auto& control_points = curve.spline_control_points;
    if (control_points.size() == 1) {
        curve.degree = 0;
        curve.points = control_points;
        return curve;
    }

    const int point_count = static_cast<int>(control_points.size());
    curve.degree = std::min(3, point_count - 1);
    if (!curve.closed) {
        using Spline2d = Eigen::Spline<double, 2>;
        Eigen::Matrix<double, 2, Eigen::Dynamic> controls(2, point_count);
        // 两行的控制点矩阵
        for (int i = 0; i < point_count; ++i) {
            controls(0, i) = control_points[static_cast<std::size_t>(i)].x;
            controls(1, i) = control_points[static_cast<std::size_t>(i)].y;
        }

        // 夹持均匀节点：首尾各重复 degree+1 次，所以曲线经过首尾控制点。
        Spline2d::KnotVectorType knots(point_count + curve.degree + 1);
        knots.head(curve.degree + 1).setZero();
        knots.tail(curve.degree + 1).setOnes();
        for (int i = curve.degree + 1; i < point_count; ++i) {
            knots(i) = static_cast<double>(i - curve.degree) /
                        static_cast<double>(point_count - curve.degree);
        }
        

        const Spline2d spline(knots, controls);

        double control_polygon_length = 0.0;
        for (int i = 1; i < point_count; ++i) {
            control_polygon_length += cv::norm(control_points[static_cast<std::size_t>(i)] -
                                               control_points[static_cast<std::size_t>(i - 1)]);
        }

        const int sample_count = std::max(2, static_cast<int>(std::ceil(control_polygon_length / sample_spacing)) + 1);
        for (int i = 0; i < sample_count; ++i) {
            const double u = static_cast<double>(i) / static_cast<double>(sample_count - 1);
            const auto point = spline(u);
            curve.points.emplace_back(point(0), point(1));
        }
        return curve;
    }

    // 周期均匀 B 样条直接循环使用首尾控制点，不重复保存第一个曲线点。
    for (int segment = 0; segment < point_count; ++segment) {
        const cv::Point2d begin = control_points[static_cast<std::size_t>(segment)];
        const cv::Point2d end = control_points[static_cast<std::size_t>((segment + 1) % point_count)];
        const int steps = std::max(1, static_cast<int>(std::ceil(cv::norm(end - begin) / sample_spacing)));
        for (int step = 0; step < steps; ++step) {
            const double t = static_cast<double>(step) / static_cast<double>(steps);
            curve.points.push_back(periodic_b_spline_point(control_points, curve.degree, segment, t));
        }
    }
    return curve;
}

/**
 * @brief 缓存 ROI 内所有子像素到曲线的最短距离。
 *
 * ROI 只覆盖曲线周围 max_half_width 加一个像素的区域。之后宽度优化不再
 * 重复计算曲线距离，只需比较 distance <= half_width。
 *
 * @param curve 已经拟合并密集采样的单根 SRAF 中心曲线。
 * @param mask_size 输出 SRAF 画布大小，格式为 (width,height)。
 * @param max_half_width 后续允许使用的最大半宽，单位为像素。
 * @param samples_per_axis 每个像素沿 x、y 方向的 MSAA 采样数量。
 * @return 局部 ROI、采样设置和所有子像素距离组成的缓存。
 */
SrafDistanceCache SrafCurve::build_distance_cache(
    const SrafParametricCurve& curve,
    cv::Size mask_size,
    double max_half_width,
    int samples_per_axis) {
    if (curve.points.empty()) {
        throw std::invalid_argument("SrafCurve::build_distance_cache: curve is empty");
    }
    if (mask_size.width <= 0 || mask_size.height <= 0) {
        throw std::invalid_argument("SrafCurve::build_distance_cache: mask size is invalid");
    }
    if (max_half_width < 0.0 || samples_per_axis < 1) {
        throw std::invalid_argument("SrafCurve::build_distance_cache: invalid render settings");
    }

    double min_x = curve.points.front().x;
    double max_x = min_x;
    double min_y = curve.points.front().y;
    double max_y = min_y;
    for (const auto& point : curve.points) {
        min_x = std::min(min_x, point.x);
        max_x = std::max(max_x, point.x);
        min_y = std::min(min_y, point.y);
        max_y = std::max(max_y, point.y);
    }

    const double padding = max_half_width + 1.0;
    const int left = std::max(0, static_cast<int>(std::floor(min_x - padding)));
    const int top = std::max(0, static_cast<int>(std::floor(min_y - padding)));
    const int right = std::min(mask_size.width - 1, static_cast<int>(std::ceil(max_x + padding)));
    const int bottom = std::min(mask_size.height - 1, static_cast<int>(std::ceil(max_y + padding)));

    SrafDistanceCache cache;
    cache.component_id = curve.component_id;
    cache.mask_size = mask_size;
    cache.roi = cv::Rect(left, top, right - left + 1, bottom - top + 1);
    cache.samples_per_axis = samples_per_axis;
    cache.max_half_width = max_half_width;

    const std::size_t sample_count = static_cast<std::size_t>(samples_per_axis * samples_per_axis);
    const std::size_t pixel_count = static_cast<std::size_t>(cache.roi.area());
    cache.distances.reserve(pixel_count * sample_count);

    for (int y = cache.roi.y; y < cache.roi.y + cache.roi.height; ++y) {
        for (int x = cache.roi.x; x < cache.roi.x + cache.roi.width; ++x) {
            for (int sy = 0; sy < samples_per_axis; ++sy) {
                for (int sx = 0; sx < samples_per_axis; ++sx) {
                    const double offset_x = (static_cast<double>(sx) + 0.5) / samples_per_axis - 0.5;
                    const double offset_y = (static_cast<double>(sy) + 0.5) / samples_per_axis - 0.5;
                    const cv::Point2d sample(x + offset_x, y + offset_y);
                    cache.distances.push_back(static_cast<float>(distance_to_curve(sample, curve)));
                }
            }
        }
    }
    return cache;
}

/**
 * @brief 用距离阈值和 MSAA 覆盖率恢复一根灰度 SRAF。
 *
 * 每个子像素到中心曲线的距离不超过 half_width 时记为内部，一个像素的
 * 灰度值等于内部子像素数除以总子像素数。
 *
 * @param cache build_distance_cache 预计算得到的子像素距离缓存。
 * @param half_width 当前 SRAF 半宽，必须位于 [0,max_half_width]。
 * @return 与原画布同尺寸的灰度 SRAF，数值范围为 [0,1]。
 */
Eigen::MatrixXd SrafCurve::render(
    const SrafDistanceCache& cache,
    double half_width) {
    if (half_width < 0.0 || half_width > cache.max_half_width + 1e-12) {
        throw std::invalid_argument("SrafCurve::render: half width is outside cache range");
    }

    Eigen::MatrixXd mask = Eigen::MatrixXd::Zero(cache.mask_size.height, cache.mask_size.width);
    const int sample_count = cache.samples_per_axis * cache.samples_per_axis;
    const std::size_t expected = static_cast<std::size_t>(cache.roi.area() * sample_count);
    if (cache.distances.size() != expected) {
        throw std::invalid_argument("SrafCurve::render: distance cache size is invalid");
    }

    std::size_t offset = 0;
    for (int y = cache.roi.y; y < cache.roi.y + cache.roi.height; ++y) {
        for (int x = cache.roi.x; x < cache.roi.x + cache.roi.width; ++x) {
            int inside_count = 0;
            for (int sample = 0; sample < sample_count; ++sample) {
                if (cache.distances[offset++] <= half_width) ++inside_count;
            }
            mask(y, x) = static_cast<double>(inside_count) / static_cast<double>(sample_count);
        }
    }
    return mask;
}

/**
 * @brief 恢复并合并多根灰度 SRAF，重叠区域取最大覆盖率。
 *
 * half_widths 只有一个元素时所有 SRAF 共用该半宽；否则每根 SRAF 使用
 * 相同下标处的独立半宽。
 *
 * @param caches 每根固定中心曲线对应的距离缓存。
 * @param half_widths 一个共享半宽，或与 caches 等长的独立半宽数组。
 * @return 所有 SRAF 按像素最大覆盖率合并后的灰度图。
 */
Eigen::MatrixXd SrafCurve::render_all(
    const std::vector<SrafDistanceCache>& caches,
    const std::vector<double>& half_widths) {
    if (caches.empty()) return {};
    if (half_widths.size() != 1 && half_widths.size() != caches.size()) {
        throw std::invalid_argument("SrafCurve::render_all: half width count does not match curves");
    }

    const cv::Size mask_size = caches.front().mask_size;
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(mask_size.height, mask_size.width);
    for (std::size_t i = 0; i < caches.size(); ++i) {
        if (caches[i].mask_size != mask_size) {
            throw std::invalid_argument("SrafCurve::render_all: mask sizes do not match");
        }
        const double half_width = half_widths.size() == 1 ? half_widths.front() : half_widths[i];
        result = result.cwiseMax(render(caches[i], half_width));
    }
    return result;
}

}  // namespace litho
