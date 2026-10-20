#pragma once

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <vector>
#include <array>
#include <string>

namespace litho {

// ── 统一数据类型（与 msaa.h / parametric.h 一致）──────────────────────
using Contour       = Eigen::MatrixXd;        // [N x 2] (y, x) double
using ControlPoints = std::vector<Contour>;   // 多轮廓

// ── 整型坐标（Bresenham 等整数运算，供 .cpp 内部使用）─────────────────
using IPoint  = std::array<int, 2>;
using IPoints = std::vector<IPoint>;

// ── EP 采样结果 ──────────────────────────────────────────────────────────
struct EpsResult {
    Eigen::MatrixXd    eps;            // [M x 2] 每行 (y, x) double
    Eigen::RowVectorXd weight_epe;     // [1 x M] EPE 权重（0 或 1）
    Eigen::RowVectorXd weight_meef;    // [1 x M] MEEF 权重
};

class EpSelect {
public:
    EpSelect(const Eigen::MatrixXd& target_mask,
             double mid_weight,
             double other_weight,
             const std::string& pattern_name = "");

    // ── 轮廓控制点 ──────────────────────────────────────────────────────
    ControlPoints fix_cps(const Eigen::MatrixXd& ls_mask, int k,
                          const std::string& symmetry = "left-right") const;

    ControlPoints fix_OPC_cps(const Eigen::MatrixXd& ls_mask, int k) const;

    ControlPoints extract_mask_control_points(int k,
                                              const std::string& symmetry = "") const;

    // ── EP 评价点 ────────────────────────────────────────────────────────
    EpsResult select_eps_others(int interval_line = 6,
                                int interval_corner = 2) const;

    EpsResult select_eps_via(int r = 1) const;

    // 为全芯片 ILT 提取亚像素 EP。输入间隔单位为像素。
    // 输出坐标为局部图像零基 (y,x)，可包含半像素位置。
    // 候选点沿 target>0.5 的 0.5 等值线按弧长采样。
    // weight_epe=1 表示 active，0 表示仅保留用于可视化的 inactive。
    EpsResult select_eps_subpixel(double interval_line = 10.0, double interval_corner = 2.0) const;

    // 对齐 edge_sampling.py：扫描 target>=threshold 的水平/垂直边界段，在去除两端 corner_margin_px 后等距采样。
    // 输出为局部零基 (y,x) 浮点像素坐标；每个输出点都是有效 EP，weight_epe 和 weight_meef 均为 1。
    EpsResult select_eps_edge_sampling(double sample_step_px = 1.0, double corner_margin_px = 0.5,
                                       bool include_image_boundary = true, double threshold = 0.5) const;

    // ── 辅助工具 ─────────────────────────────────────────────────────────
    static ControlPoints remove_adjacent_close_points(const ControlPoints& contours,
                                                      double threshold = 2.0);

private:
    static IPoints     bresenham_line(int y1, int x1, int y2, int x2);
    static IPoints     sample_elements(const IPoints& lst, int k,
                                       const std::string& mode = "skip");
    static std::string segment_orientation(const IPoint& s, const IPoint& e);
    static bool        is_short_corner_connector(const IPoints& pts,
                                                  int seg_idx, int seg_len,
                                                  int left_lim, int right_lim);

    static cv::Mat  to_cv8u(const Eigen::MatrixXd& M);
    static IPoints  cv_to_ipoints(const std::vector<cv::Point>& c);
    static Contour  ipoints_to_contour(const IPoints& pts);

    Eigen::MatrixXd _target;
    double          _mid_weight;
    double          _other_weight;
    std::string     _pattern_name;
};

}  // namespace litho
