#pragma once

#include "sraf_geometry.h"

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <vector>

namespace litho {

// 单根 SRAF 的 B 样条。spline_control_points 就是从骨架间隔取得的控制点，
// points 是按曲线顺序密集采样的 (x,y) 坐标。
struct SrafParametricCurve {
    int component_id = 0;
    bool closed = false;
    int degree = 0;
    std::vector<cv::Point2d> spline_control_points;
    std::vector<cv::Point2d> points;
};

// 固定中心曲线的子像素距离缓存。宽度优化时只需改变距离阈值。
struct SrafDistanceCache {
    int component_id = 0;
    cv::Size mask_size;
    cv::Rect roi;
    int samples_per_axis = 4;
    double max_half_width = 0.0;
    std::vector<float> distances;
};

class SrafCurve {
public:
    // 直接把骨架取样点作为 B 样条控制点，不要求曲线经过中间控制点。
    // 开放曲线使用夹持均匀 B 样条，闭合曲线使用周期均匀 B 样条。
    // sample_spacing 控制拟合曲线相邻密集点的大致距离，单位为像素。
    static SrafParametricCurve fit(
        const SrafCenterlineGeometry& centerline,
        double sample_spacing = 0.25);

    // 预计算 ROI 内每个子像素到曲线折线段的最短距离。
    static SrafDistanceCache build_distance_cache(
        const SrafParametricCurve& curve,
        cv::Size mask_size,
        double max_half_width,
        int samples_per_axis = 4);

    // 根据半宽和子像素覆盖率恢复单根灰度 SRAF，返回值范围为 [0,1]。
    static Eigen::MatrixXd render(
        const SrafDistanceCache& cache,
        double half_width);

    // 合并多根 SRAF。一个半宽表示全部共用，否则必须与缓存数量相同。
    static Eigen::MatrixXd render_all(
        const std::vector<SrafDistanceCache>& caches,
        const std::vector<double>& half_widths);
};

}  // namespace litho
