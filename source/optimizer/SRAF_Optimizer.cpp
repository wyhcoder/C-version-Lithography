#include "SRAF_Optimizer.h"
#include "ep_select.h"
#include "save_txt.h"
#include "imaging.h"
#include "parametric.h"
#include "loss.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "sraf_curve.h"
#include "sraf_geometry.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace litho {

cv::Mat SRAF_Optimizer::eigen_mask_to_u8(const Eigen::MatrixXd& mask) {
    cv::Mat image(
        static_cast<int>(mask.rows()),
        static_cast<int>(mask.cols()),
        CV_8UC1,
        cv::Scalar(0));
    for (Eigen::Index y = 0; y < mask.rows(); ++y) {
        for (Eigen::Index x = 0; x < mask.cols(); ++x) {
            const double value = std::clamp(mask(y, x), 0.0, 1.0);
            image.at<uchar>(static_cast<int>(y), static_cast<int>(x)) =
                static_cast<uchar>(std::lround(value * 255.0));
        }
    }
    return image;
}

SrafGeometryConfig SRAF_Optimizer::make_geometry_config(
    const SRAFConfig& config) {
    SrafGeometryConfig geometry_config;
    geometry_config.foreground_threshold = config.foreground_threshold;
    geometry_config.target_threshold = config.target_threshold;
    geometry_config.overlap_ratio = config.overlap_ratio;
    geometry_config.fallback_dilate_radius = config.fallback_dilate_radius;
    geometry_config.opening_radius = config.opening_radius;
    geometry_config.minimum_component_area = config.minimum_component_area;
    geometry_config.control_point_interval = config.control_point_interval;
    return geometry_config;
}

ControlPoints SRAF_Optimizer::load_main_control_points_txt(
    const std::string& path,
    int image_rows,
    int image_cols) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error(
            "SRAF_Optimizer: cannot open main control-points file: " + path);
    }

    std::vector<std::vector<Eigen::Vector2d>> parsed_contours;
    std::vector<Eigen::Vector2d> current_contour;
    auto finish_contour = [&]() {
        if (!current_contour.empty()) {
            parsed_contours.push_back(std::move(current_contour));
            current_contour.clear();
        }
    };

    std::string line;
    int line_number = 0;
    while (std::getline(input, line)) {
        ++line_number;
        const std::size_t first = line.find_first_not_of(" \t\r\n");
        if (first == std::string::npos) {
            finish_contour();
            continue;
        }
        if (line[first] == '#') {
            if (line.find("contour", first) != std::string::npos) {
                finish_contour();
            }
            continue;
        }

        std::istringstream values(line);
        double y = 0.0;
        double x = 0.0;
        std::string extra;
        if (!(values >> y >> x) || (values >> extra)) { // 读取不出或者是尝试读取下一个数字
            throw std::runtime_error(
                "SRAF_Optimizer: expected exactly 'y x' at " + path +
                ":" + std::to_string(line_number));
        }
        if (!std::isfinite(y) || !std::isfinite(x) ||
            y < 0.0 || y > image_rows - 1.0 ||
            x < 0.0 || x > image_cols - 1.0) {
            throw std::runtime_error(
                "SRAF_Optimizer: invalid/out-of-bounds main control point at " +
                path + ":" + std::to_string(line_number));
        }
        current_contour.emplace_back(y, x);
    }
    finish_contour();

    if (parsed_contours.empty()) {
        throw std::runtime_error(
            "SRAF_Optimizer: no main control-point contours in " + path);
    }

    ControlPoints result;
    result.reserve(parsed_contours.size());
    for (std::size_t contour_index = 0;
         contour_index < parsed_contours.size();
         ++contour_index) {
        const auto& points = parsed_contours[contour_index];
        if (points.size() < 2) {
            throw std::runtime_error(
                "SRAF_Optimizer: main contour " +
                std::to_string(contour_index) +
                " has fewer than 2 points in " + path);
        }

        Contour contour(static_cast<Eigen::Index>(points.size()), 2);
        for (Eigen::Index row = 0; row < contour.rows(); ++row) {
            contour.row(row) =
                points[static_cast<std::size_t>(row)].transpose();
        }
        result.push_back(std::move(contour));
    }
    return result;
}

SRAF_Optimizer::SRAF_Optimizer(
    const LithographySimulator& simulator,
    const ImagingCache& cache,
    const SRAFConfig& config)
    : _simulator(simulator),
      _cache(cache),
      _sraf_config(config) {
    if (_sraf_config.ls_mask_path.empty()) {
        throw std::invalid_argument(
            "SRAF_Optimizer: ls_mask_path must not be empty");
    }
    if (_sraf_config.save_file_path.empty()) {
        throw std::invalid_argument(
            "SRAF_Optimizer: save_file_path must not be empty");
    }
    if (_sraf_config.main_cps_path.empty()) {
        throw std::invalid_argument(
            "SRAF_Optimizer: main_cps_path must not be empty");
    }

    SaveTxt::load_txt(_sraf_config.ls_mask_path, _lsm_mask);
    _target_mask = _simulator._mask.data();
    _main_cps = load_main_control_points_txt(
        _sraf_config.main_cps_path,
        static_cast<int>(_target_mask.rows()),
        static_cast<int>(_target_mask.cols()));
    for (const auto& contour : _main_cps) {
        _main_control_point_count += static_cast<std::size_t>(contour.rows());
    }

    _sraf_geometry_result = SrafGeometry::extract(
        _lsm_mask,
        _target_mask,
        make_geometry_config(_sraf_config));

    _main_pixels = static_cast<int>(
        (_sraf_geometry_result.split.main_mask.array() >
         _sraf_config.foreground_threshold)
            .count());
    _sraf_pixels = static_cast<int>(
        (_sraf_geometry_result.split.sraf_mask.array() >
         _sraf_config.foreground_threshold)
            .count());
    for (const auto& centerline : _sraf_geometry_result.centerlines) {
        _control_point_count += centerline.control_points.size();
    }

    std::cout << "\nSRAF geometry initialization\n"
              << "  pattern           : " << _sraf_config.pattern_name << '\n'
              << "  mask size         : " << _lsm_mask.rows() << " x "
              << _lsm_mask.cols() << '\n'
              << "  main pixels       : " << _main_pixels << '\n'
              << "  raw SRAF pixels   : " << _sraf_pixels << '\n'
              << "  cleaned pixels    : "
              << cv::countNonZero(_sraf_geometry_result.cleaned_sraf_mask) << '\n'
              << "  skeleton pixels   : "
              << cv::countNonZero(_sraf_geometry_result.skeleton_mask) << '\n'
              << "  SRAF components   : "
              << _sraf_geometry_result.centerlines.size() << '\n'
              << "  main contours     : " << _main_cps.size() << '\n'
              << "  main control pts  : " << _main_control_point_count << '\n'
              << "  SRAF control pts  : " << _control_point_count << "\n\n";

    for (const auto& centerline : _sraf_geometry_result.centerlines) {
        std::cout << "  component " << std::setw(3) << centerline.component_id
                  << " | valid=" << (centerline.path.valid ? "yes" : "no")
                  << " | closed=" << (centerline.path.closed ? "yes" : "no")
                  << " | controls=" << centerline.control_points.size()
                  << " | " << centerline.path.diagnostic << '\n';
    }

    validate_geometry_result();
    save_main_control_points();
    save_control_points();
    EpSelect ep_select(_target_mask, _sraf_config.mid_weight, _sraf_config.other_weight);
    _eps_result = ep_select.select_eps_others(_sraf_config.interval_line, _sraf_config.interval_corner);
    render_initial_masks();
    save_initial_masks();
}

void SRAF_Optimizer::validate_geometry_result() const {
    if (_sraf_geometry_result.centerlines.empty()) {
        throw std::runtime_error(
            "SRAF_Optimizer: no SRAF component was extracted from the input mask");
    }

    std::ostringstream invalid_components;
    bool first = true;
    for (const auto& centerline : _sraf_geometry_result.centerlines) {
        if (centerline.path.valid && !centerline.control_points.empty()) {
            continue;
        }
        if (!first) invalid_components << ", ";
        invalid_components << centerline.component_id;
        first = false;
    }
    if (!first) {
        throw std::runtime_error(
            "SRAF_Optimizer: invalid SRAF centerline/control points in component(s): " +
            invalid_components.str());
    }
}

void SRAF_Optimizer::save_control_points() const {
    namespace fs = std::filesystem;
    const fs::path output_dir(_sraf_config.save_file_path);
    fs::create_directories(output_dir);
    const fs::path output_path = output_dir / "sraf_control_points_xy.txt";

    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error(
            "SRAF_Optimizer: cannot write control points to " +
            output_path.string());
    }

    output << "# SRAF control points\n"
           << "# coordinate_order=x y\n"
           << "# component_count=" << _sraf_geometry_result.centerlines.size()
           << " total_control_points=" << _control_point_count << '\n'
           << std::fixed << std::setprecision(6);
    for (const auto& centerline : _sraf_geometry_result.centerlines) {
        output << "# component=" << centerline.component_id
               << " closed=" << (centerline.path.closed ? 1 : 0)
               << " count=" << centerline.control_points.size() << '\n';
        for (const auto& point : centerline.control_points) {
            output << point.x << ' ' << point.y << '\n';
        }
    }
    if (!output) {
        throw std::runtime_error(
            "SRAF_Optimizer: failed while writing control points to " +
            output_path.string());
    }
    std::cout << "\n  control points saved: " << output_path << '\n';
}

void SRAF_Optimizer::save_main_control_points() const {
    namespace fs = std::filesystem;
    const fs::path output_dir(_sraf_config.save_file_path);
    fs::create_directories(output_dir);
    const fs::path output_path =
        output_dir / "main_control_points_loaded_yx.txt";

    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error(
            "SRAF_Optimizer: cannot write loaded main control points to " +
            output_path.string());
    }

    output << std::fixed << std::setprecision(6);
    for (std::size_t contour_index = 0;
         contour_index < _main_cps.size();
         ++contour_index) {
        const Contour& contour = _main_cps[contour_index];
        output << "# contour " << contour_index
               << " (num_points = " << contour.rows() << ")\n";
        for (Eigen::Index row = 0; row < contour.rows(); ++row) {
            output << contour(row, 0) << ' ' << contour(row, 1) << '\n';
        }
        output << '\n';
    }
    if (!output) {
        throw std::runtime_error(
            "SRAF_Optimizer: failed while writing loaded main control points to " +
            output_path.string());
    }
    std::cout << "  main control points loaded: "
              << output_path << '\n';
}

void SRAF_Optimizer::render_initial_masks() {
    if (_sraf_config.curve_type != "OA" &&
        _sraf_config.curve_type != "BZ" &&
        _sraf_config.curve_type != "BS") {
        throw std::invalid_argument(
            "SRAF_Optimizer: curve_type must be OA, BZ, or BS");
    }
    if (_sraf_config.initial_half_width < 0.0) {
        throw std::invalid_argument(
            "SRAF_Optimizer: initial_half_width must be non-negative");
    }
    if (!std::isfinite(_sraf_config.maximum_half_width) ||
        _sraf_config.maximum_half_width < _sraf_config.initial_half_width) {
        throw std::invalid_argument(
            "SRAF_Optimizer: maximum_half_width must be finite and not "
            "smaller than initial_half_width");
    }
    if (_sraf_config.samples_per_axis < 1) {
        throw std::invalid_argument(
            "SRAF_Optimizer: samples_per_axis must be at least 1");
    }

    _sraf_curves.clear();
    _sraf_distance_caches.clear();
    _sraf_curves.reserve(_sraf_geometry_result.centerlines.size());
    _sraf_distance_caches.reserve(_sraf_geometry_result.centerlines.size());

    // cv::Size 的顺序是 (width,height)，对应 Eigen 的 (cols,rows)。
    const cv::Size mask_size(
        static_cast<int>(_lsm_mask.cols()),
        static_cast<int>(_lsm_mask.rows()));
    for (const auto& centerline : _sraf_geometry_result.centerlines) {
        _sraf_curves.push_back(SrafCurve::fit(centerline));
        _sraf_distance_caches.push_back(SrafCurve::build_distance_cache(
            _sraf_curves.back(),
            mask_size,
            _sraf_config.maximum_half_width,
            _sraf_config.samples_per_axis));
    }

    const std::size_t width_count = _sraf_config.independent_sraf_widths
        ? _sraf_distance_caches.size(): 1;
    _initial_half_widths.assign( width_count, _sraf_config.initial_half_width);

    _rendered_sraf_mask = SrafCurve::render_all(_sraf_distance_caches,_initial_half_widths);

    ParametricDemo parametric(
        _sraf_config.curve_type,
        _target_mask,
        _sraf_config.msaa_level);
    _rendered_main_mask = parametric.render_curve(_main_cps);

   

    // 两张覆盖率 mask 做几何并集，保证整体 mask 始终位于 [0,1]。
    _initial_mask = _rendered_main_mask.cwiseMax(_rendered_sraf_mask);
}

void SRAF_Optimizer::save_initial_masks() const {
    namespace fs = std::filesystem;
    const fs::path output_dir(_sraf_config.save_file_path);
    fs::create_directories(output_dir);

    const auto save_png = [&](const Eigen::MatrixXd& mask,
                              const char* file_name) {
        const fs::path path = output_dir / file_name;
        if (!cv::imwrite(path.string(), eigen_mask_to_u8(mask))) {
            throw std::runtime_error(
                "SRAF_Optimizer: cannot write mask image to " +
                path.string());
        }
        std::cout << "  mask image saved: " << path << '\n';
    };

    save_png(_rendered_main_mask, "initial_main_mask.png");
    save_png(_rendered_sraf_mask, "initial_sraf_mask.png");
    save_png(_initial_mask, "initial_combined_mask.png");
    SaveTxt::save_mat(
        _initial_mask,
        (output_dir / "initial_combined_mask.txt").string(),
        6);

    std::cout << "  SRAF width mode   : "
              << (_sraf_config.independent_sraf_widths
                      ? "independent"
                      : "shared")
              << '\n'
              << "  width variables   : " << _initial_half_widths.size()
              << '\n'
              << "  initial half width: "
              << _sraf_config.initial_half_width << " px\n"
              << "  maximum half width: "
              << _sraf_config.maximum_half_width << " px\n";
}

void SRAF_Optimizer::optimize(){
    Imaging imaging(_cache); //创建成像对象
    // 成像模型的参数
    const double threshold = _simulator._params.resist.threshold;
    const double alpha = _simulator._params.resist.alpha;
    const double pixel_size = _simulator._params.system.pixel_size_nm;
    const Eigen::RowVectorXd& wepe_weight = _eps_result.weight_epe;

    struct EvaluateState{
        Eigen::MatrixXd mask;
        Imaging_Result imaging;
        double epe = 0.0;
        double wepe = 0.0;
        double pe = 0.0;
        Pvband_result pvBandResult;
        Eigen::RowVectorXd epe_vector;

    };

    PvbandComputer pvBandCompute(_simulator, _sraf_config.dose_margin, _sraf_config.defocus_range, _sraf_config.defocus_step, _sraf_config.mode);

    auto evalua = [&](const Eigen::MatrixXd& mask){
        EvaluateState state;
        state.mask = mask;
        state.imaging = imaging.compute(mask, threshold, alpha );
        EpeEvaluation epe = Loss::evaluate_epe(
                state.imaging.aerial_image,
                _eps_result.eps,
                threshold,
                pixel_size);
        state.epe = epe.total_epe;
        state.epe_vector = std::move(epe.epe_vector);
        state.wepe = Loss::weighted_epe(wepe_weight, state.epe_vector);
        state.pe = Loss::pe_loss(state.imaging.wafer_image, _target_mask);
        state.pvBandResult = pvBandCompute.compute_pvband(state.mask);
        return state;

    };
    auto print_metrics = [](const char* label, double mean_wepe, double mean_epe, double pe, double pvband){
        std::ostringstream line;
        line << "  " << std::left << std::setw(18) << label
                 << std::right << std::fixed << std::setprecision(6)
                 << " | mean_wEPE: " << std::setw(12) << mean_wepe
                 << " | mean_EPE: " << std::setw(12) << mean_epe
                 << " | PE: " << std::setw(12) << pe
                 << " | PV Band: " << std::setw(12) << pvband;
            std::cout << line.str() << '\n';
    };

    // 初始化 mask 已在构造阶段生成。后续宽度优化直接复用
    // _sraf_distance_caches，不需要重复拟合曲线和计算子像素距离。
    Eigen::MatrixXd current_mask = _initial_mask;
    EvaluateState init_eval = evalua(current_mask);

    const int ep_count = static_cast<int>(_eps_result.eps.rows());
    const int weighted_ep_count = static_cast<int>(
        (_eps_result.weight_epe.array() == 1.0).count());
    std::cout << "\nInitial mask evaluation\n";
    print_metrics(
        "initial mask",
        init_eval.wepe / std::max(1, weighted_ep_count),
        init_eval.epe / std::max(1, ep_count),
        init_eval.pe,
        init_eval.pvBandResult.pv_loss);
    std::cout << "  PV map            | min: "
              << init_eval.pvBandResult.pv_map.minCoeff()
              << " | max: " << init_eval.pvBandResult.pv_map.maxCoeff()
              << " | mean: " << init_eval.pvBandResult.pv_map.mean()
              << '\n';

    const std::filesystem::path pv_map_path =
        std::filesystem::path(_sraf_config.save_file_path) /
        "initial_pv_band_map.png";
    if (!cv::imwrite(
            pv_map_path.string(),
            eigen_mask_to_u8(init_eval.pvBandResult.pv_map))) {
        throw std::runtime_error(
            "SRAF_Optimizer: cannot write PV Band map to " +
            pv_map_path.string());
    }
    std::cout << "  PV map saved       : " << pv_map_path << '\n';











}

}  // namespace litho
