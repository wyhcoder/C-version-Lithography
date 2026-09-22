#pragma once

#include <Eigen/Dense>
#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace litho {

// 完整掩模拆分结果。灰度值直接从输入 full_mask 复制，不在这里二值化。
struct SrafSplitResult {
    Eigen::MatrixXd main_mask;
    Eigen::MatrixXd sraf_mask;
};

// 一个骨架连通域排序后的路径。坐标统一采用 OpenCV 约定 (x, y)。
struct SkeletonPathResult {
    std::vector<cv::Point2d> points;
    // 分叉骨架按端点/分叉点拆出的无分叉边；边内保持原始骨架像素顺序。
    std::vector<std::vector<cv::Point2d>> edges;
    bool closed = false;
    bool valid = false;
    int endpoint_count = 0;
    int branchpoint_count = 0;
    std::string diagnostic;
};

// 单根 SRAF 的几何信息。参数化样条会在后续 centerline_spline 模块生成；
// 本模块只负责得到可靠、有序的中心路径和按固定间隔选取的控制点。
struct SrafCenterlineGeometry {
    int component_id = 0;
    int area = 0;
    cv::Rect bounding_box;
    cv::Point2d centroid;  // (x, y)

    cv::Mat skeleton_mask;  // CV_8UC1，尺寸与完整版图相同，取值 0/255
    SkeletonPathResult path;
    std::vector<cv::Point2d> control_points;
    // 分叉图每条边的间隔取点，供逐边开放式插值曲线拟合。
    std::vector<std::vector<cv::Point2d>> edge_control_points;
};

struct SrafGeometryConfig {
    // full_mask > foreground_threshold 视为前景。
    double foreground_threshold = 1e-6;
    // target_mask > target_threshold 视为主图形参考区域。
    double target_threshold = 0.0;
    // 连通域与 target 的重叠像素数至少达到该面积比例，才判为主图形。
    double overlap_ratio = 0.05;
    // 完全找不到与 target 重叠的主图形时，用 target 膨胀区域进行兜底拆分。
    int fallback_dilate_radius = 3;

    // SRAF 骨架化前的可选开运算半径；0 表示不做开运算。
    int opening_radius = 0;
    // 小于该面积的 SRAF 连通域会在骨架化前删除。
    int minimum_component_area = 3;
    // 沿有序骨架每隔多少个骨架像素取一个控制点。
    int control_point_interval = 10;
};

struct SrafGeometryResult {
    SrafSplitResult split;
    cv::Mat cleaned_sraf_mask;  // CV_8UC1，0/255
    cv::Mat skeleton_mask;      // 所有 SRAF 骨架的并集，0/255
    std::vector<SrafCenterlineGeometry> centerlines;
};

class SrafGeometry {
public:
    // 按“8 连通域与 target 的真实重叠比例”拆分主图形与 SRAF；若没有任何
    // 连通域匹配 target，则回退到 target 膨胀区域拆分。
    static SrafSplitResult split_main_and_sraf(
        const Eigen::MatrixXd& full_mask,
        const Eigen::MatrixXd& target_mask,
        double foreground_threshold = 1e-6,
        double target_threshold = 0.0,
        double overlap_ratio = 0.05,
        int fallback_dilate_radius = 3);

    // 转换为 CV_8UC1 二值图，前景为 255、背景为 0。
    static cv::Mat to_binary_mask(
        const Eigen::MatrixXd& mask,
        double threshold = 1e-6);

    // 可选开运算并删除小连通域。opening_radius=0 时跳过开运算。
    static cv::Mat clean_binary_sraf(
        const cv::Mat& binary_sraf,
        int opening_radius = 0,
        int minimum_component_area = 3);

    // 通过 OpenCV contrib 的 Zhang-Suen 细化将二值 SRAF 变为一像素宽骨架，
    // 输出取值 0/255。
    static cv::Mat skeletonize(const cv::Mat& binary_sraf);

    // 无分叉骨架排序为一条路径；分叉骨架按端点/分叉点拆成多条边。
    // 单点、开放路径和闭环继续使用 points；分叉使用 edges。
    static SkeletonPathResult order_skeleton_path(
        const cv::Mat& single_component_skeleton);

    // 沿有序骨架按下标间隔直接取点。所有控制点都来自原始整数骨架；
    // 开放路径额外保留末端点，闭合路径不重复首尾点。
    static std::vector<cv::Point2d> sample_skeleton_points(
        const std::vector<cv::Point2d>& ordered_path,
        int interval,
        bool closed);

    // 完整入口：拆分 -> 清理 -> 连通域 -> 骨架 -> 路径排序 -> 控制点采样。
    static SrafGeometryResult extract(
        const Eigen::MatrixXd& full_mask,
        const Eigen::MatrixXd& target_mask,
        const SrafGeometryConfig& config = {});
};

}  // namespace litho
