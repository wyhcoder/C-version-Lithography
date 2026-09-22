#pragma once

#include "ep_select.h"
#include "litho_prepare.h"
#include "lithography_simulator.h"
#include "sraf_curve.h"
#include "sraf_geometry.h"
#include "pv_band_computer.h"
#include <Eigen/Dense>
#include <opencv2/core/mat.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace litho {

struct SRAFConfig {
    std::string pattern_name;
    std::string ls_mask_path;
    std::string save_file_path;
    // MEEF best_wepe/control_points.txt，文件内每行坐标顺序为 (y,x)。
    std::string main_cps_path;

    // 主图形控制点的曲线类型：OA 折线、BZ Bezier、BS B 样条。
    std::string curve_type = "BS";
    // false：所有 SRAF 共用一个半宽；true：每根 SRAF 有独立半宽。
    bool independent_sraf_widths = true;

    // SRAF 几何提取参数。
    double foreground_threshold = 1e-6;
    double target_threshold = 0.0;
    double overlap_ratio = 0.05;
    int fallback_dilate_radius = 3;
    int opening_radius = 0;
    int minimum_component_area = 3;
    int control_point_interval = 10;
    double initial_half_width = 6.0;
    // 距离缓存覆盖的最大半宽；未来优化得到的半宽不能超过它。
    double maximum_half_width = 8.0;
    int samples_per_axis = 4;

    // 所有 SRAF 共用一个半宽时的一维搜索参数。
    double shared_width_min = 1.0;
    double shared_width_max = 4.0;
    double shared_width_coarse_step = 0.5;
    double shared_width_fine_step = 0.1;

    // 每根 SRAF 独立宽度时使用有边界 CMA-ES。
    double independent_width_min = 1.0;
    double independent_width_max = 4.0;
    double cma_initial_sigma = 0.375;
    int cma_population_size = 12;
    int cma_max_evaluations = 200;
    double cma_tolerance_x = 1e-2;
    double cma_tolerance_fun = 1e-3;
    // libcmaes 中 0 表示随机种子；使用正数可复现实验。
    unsigned int cma_seed = 1;

    double weight_pvband = 0.7;
    double weight_wepe = 0.25;
    double weight_pe = 0.05;

    // 为后续 PV Band 宽度优化预留；当前初始化阶段不使用。
    double mid_weight = 4.0;
    double other_weight = 1.0;
    int msaa_level = 16;
    double dose_margin = 0.05;
    double defocus_range = 200.0;
    int defocus_step = 3;
    // EPE 选取。
    int interval_line = 5;
    int interval_corner = 2;
    std::string mode = "full";
};

class SRAF_Optimizer {
public:
    SRAF_Optimizer(
        const LithographySimulator& simulator,
        const ImagingCache& cache,
        const SRAFConfig& config);

    // 共享宽度使用粗到细的一维搜索，独立宽度使用 CMA-ES。
    void optimize();

    // SRAF 几何中的 cv::Point2d 为 (x,y)；main_control_points 为 [y,x]。
    const SrafGeometryResult& geometry_result() const noexcept {
        return _sraf_geometry_result;
    }
    const ControlPoints& main_control_points() const noexcept {
        return _main_cps;
    }
    const Eigen::MatrixXd& rendered_main_mask() const noexcept {
        return _rendered_main_mask;
    }
    const Eigen::MatrixXd& rendered_sraf_mask() const noexcept {
        return _rendered_sraf_mask;
    }
    const Eigen::MatrixXd& initial_mask() const noexcept {
        return _initial_mask;
    }
    const std::vector<double>& initial_half_widths() const noexcept {
        return _initial_half_widths;
    }
    // 共享宽度模式的结果；独立模式使用 optimized_half_widths()。
    double optimized_half_width() const noexcept {
        return _optimized_half_width;
    }
    const std::vector<double>& optimized_half_widths() const noexcept {
        return _optimized_half_widths;
    }
    const Eigen::MatrixXd& optimized_mask() const noexcept {
        return _optimized_mask;
    }
    int main_pixel_count() const noexcept { return _main_pixels; }
    int sraf_pixel_count() const noexcept { return _sraf_pixels; }
    std::size_t main_control_point_count() const noexcept {
        return _main_control_point_count;
    }
    std::size_t control_point_count() const noexcept {
        return _control_point_count;
    }
    // 因路径不连续或控制点为空而从宽度优化中删除的原始 component id；分叉骨架正常保留。
    const std::vector<int>& dropped_component_ids() const noexcept {
        return _dropped_component_ids;
    }

    static cv::Mat eigen_mask_to_u8(const Eigen::MatrixXd& mask);

private:
    static SrafGeometryConfig make_geometry_config(const SRAFConfig& config);
    static ControlPoints load_main_control_points_txt(
        const std::string& path,
        int image_rows,
        int image_cols);
    void drop_invalid_geometry_components();
    void validate_geometry_result() const;
    void render_initial_masks();
    void save_control_points() const;
    void save_main_control_points() const;
    void save_initial_masks() const;

    LithographySimulator _simulator;
    ImagingCache _cache;
    SRAFConfig _sraf_config;

    Eigen::MatrixXd _lsm_mask;
    Eigen::MatrixXd _target_mask;
    SrafGeometryResult _sraf_geometry_result;
    ControlPoints _main_cps;
    EpsResult _eps_result;
    std::vector<SrafParametricCurve> _sraf_curves;
    std::vector<SrafDistanceCache> _sraf_distance_caches;
    std::vector<double> _initial_half_widths;
    std::vector<int> _dropped_component_ids;
    Eigen::MatrixXd _rendered_main_mask;
    Eigen::MatrixXd _rendered_sraf_mask;
    Eigen::MatrixXd _initial_mask;
    double _optimized_half_width = 0.0;
    std::vector<double> _optimized_half_widths;
    Eigen::MatrixXd _optimized_mask;
    int _main_pixels = 0;
    int _sraf_pixels = 0;
    std::size_t _main_control_point_count = 0;
    std::size_t _control_point_count = 0;

};

}  // namespace litho
