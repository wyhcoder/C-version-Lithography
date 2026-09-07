#include "SRAF_Optimizer.h"
#include "ep_select.h"
#include "save_txt.h"
#include "imaging.h"
#include "parametric.h"
#include "loss.h"
#include <libcmaes/cmaes.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
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
namespace {

// 保存 libcmaes 的运行统计。
struct CmaEsResult {
    int evaluations = 0;
    int generations = 0;
};

/**
 * @brief 调用 libcmaes 执行有边界的全协方差 CMA-ES。
 *
 * 每个参数下标始终对应同一根 SRAF，边界由 pwqBoundStrategy 处理。
 */
template <typename Objective>
CmaEsResult run_bounded_cma_es(
    const Eigen::VectorXd& initial_mean,
    double initial_sigma,
    double lower_width,
    double upper_width,
    int population_size,
    int max_evaluations,
    double tolerance_x,
    double tolerance_fun,
    unsigned int seed,
    Objective&& objective) {
    const int dimension = static_cast<int>(initial_mean.size());
    std::vector<double> x0(
        initial_mean.data(), initial_mean.data() + initial_mean.size());
    std::vector<double> lower_bounds(dimension, lower_width);
    std::vector<double> upper_bounds(dimension, upper_width);

    using BoundTransform =
        libcmaes::GenoPheno<libcmaes::pwqBoundStrategy>;
    BoundTransform bounds(
        lower_bounds.data(), upper_bounds.data(), dimension);

    libcmaes::FitFunc fitness =
        [&](const double* values, const int& value_count) {
            const Eigen::Map<const Eigen::VectorXd> parameters(
                values, value_count);
            return objective(parameters);
        };

    libcmaes::CMAParameters<BoundTransform> parameters(
        x0,
        initial_sigma,
        population_size,
        seed,
        bounds);
    const int max_generations = max_evaluations / population_size;
    parameters.set_max_iter(max_generations);
    parameters.set_max_fevals(max_generations * population_size);
    parameters.set_xtolerance(tolerance_x);
    parameters.set_ftolerance(tolerance_fun);
    parameters.set_mt_feval(false);
    parameters.set_quiet(true);

    libcmaes::ProgressFunc<
        libcmaes::CMAParameters<BoundTransform>,
        libcmaes::CMASolutions>
        progress = [](
            const libcmaes::CMAParameters<BoundTransform>&,
            const libcmaes::CMASolutions& solutions) {
            if (solutions.niter() == 0) return 0;
            std::cout << "  generation " << std::setw(3)
                      << solutions.niter()
                      << " | evaluations=" << std::setw(4)
                      << solutions.fevals()
                      << " | best=" << std::fixed << std::setprecision(6)
                      << solutions.get_best_seen_candidate().get_fvalue()
                      << " | sigma=" << solutions.sigma() << '\n';
            return 0;
        };

    const libcmaes::CMASolutions solutions =
        libcmaes::cmaes<BoundTransform>(
            fitness, parameters, progress);

    CmaEsResult result;
    result.evaluations = solutions.nevals();
    result.generations = solutions.niter();
    return result;
}

}  // namespace

// 将 [0,1] 的 Eigen 灰度 mask 转为 OpenCV 8 位图像，用于保存 PNG。
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

// 从 SRAFConfig 中提取几何参数，生成 SrafGeometryConfig。
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

// 读取主图形控制点 TXT；空行或 contour 注释用于分隔轮廓。
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
    // 完成当前轮廓，然后继续读取下一条轮廓。
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

// 完成优化器初始化：读取输入、提取 SRAF、选择 EP 并生成初始 mask。
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

    // 读取 LSM mask、目标图形以及主图形优化后的控制点。
    SaveTxt::load_txt(_sraf_config.ls_mask_path, _lsm_mask);
    _target_mask = _simulator._mask.data();
    _main_cps = load_main_control_points_txt(
        _sraf_config.main_cps_path,
        static_cast<int>(_target_mask.rows()),
        static_cast<int>(_target_mask.cols()));
    for (const auto& contour : _main_cps) {
        _main_control_point_count += static_cast<std::size_t>(contour.rows());
    }

    // 从 LSM mask 中分离并提取每根 SRAF 的骨架和控制点。
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

    // 保存初始化结果，并为后续 EPE 与宽度优化准备数据。
    validate_geometry_result();
    save_main_control_points();
    save_control_points();
    EpSelect ep_select(_target_mask, _sraf_config.mid_weight, _sraf_config.other_weight);
    _eps_result = ep_select.select_eps_others(_sraf_config.interval_line, _sraf_config.interval_corner);
    render_initial_masks();
    save_initial_masks();
}

// 检查是否提取到 SRAF，以及每个分量是否具有有效骨架和控制点。
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

// 将每根 SRAF 的 B 样条控制点按 (x,y) 顺序保存到 TXT。
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

// 将主图形控制点按 (y,x) 顺序保存，便于核对读取结果。
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

// 拟合 SRAF 曲线、建立距离缓存，并渲染主图形和初始 SRAF mask。
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

    // 每根 SRAF 只拟合一次，并建立优化过程中重复使用的距离缓存。
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

    // 共享模式保存一个宽度，独立模式为每根 SRAF 保存一个宽度。
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

// 保存主图形、SRAF 和二者合并后的初始 mask。
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

// 优化 SRAF 半宽：独立模式使用 CMA-ES，共享模式使用粗细两级搜索。
void SRAF_Optimizer::optimize() {
    // nominal 成像对象在所有候选评价之间重复使用。
    Imaging imaging(_cache);
    // 成像模型的参数
    const double threshold = _simulator._params.resist.threshold;
    const double alpha = _simulator._params.resist.alpha;
    const double pixel_size = _simulator._params.system.pixel_size_nm;
    const Eigen::RowVectorXd& wepe_weight = _eps_result.weight_epe;

    // 保存一个候选 mask 对应的成像结果和三类误差。
    struct EvaluateState {
        Eigen::MatrixXd mask;
        Imaging_Result imaging;
        double epe = 0.0;
        double wepe = 0.0;
        double pe = 0.0;
        Pvband_result pvBandResult;
        Eigen::RowVectorXd epe_vector;

    };

    // 构造时已经预计算不同离焦量的 SOCS cache。
    PvbandComputer pvBandCompute(_simulator, _sraf_config.dose_margin, _sraf_config.defocus_range, _sraf_config.defocus_step, _sraf_config.mode);

    // 对完整 mask 统一计算 EPE、加权 EPE、PE 和 PV Band。
    auto evaluate = [&](const Eigen::MatrixXd& mask) {
        EvaluateState state;
        state.mask = mask;
        state.imaging = imaging.compute(mask, threshold, alpha);
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
    // 初始化 mask 已在构造阶段生成。后续宽度优化直接复用
    // _sraf_distance_caches，不需要重复拟合曲线和计算子像素距离。
    EvaluateState init_eval = evaluate(_initial_mask);

    // 把初始和优化后的 PV Band 用相同色标并排保存，便于直接比较。
    const auto save_pvband_comparison = [&init_eval](
        const EvaluateState& optimized,
        const std::filesystem::path& output_dir,
        const std::string& optimized_file_name) {
        const cv::Mat initial_gray = eigen_mask_to_u8(
            init_eval.pvBandResult.pv_map);
        const cv::Mat optimized_gray = eigen_mask_to_u8(
            optimized.pvBandResult.pv_map);

        cv::imwrite(
            (output_dir / "initial_pv_band_map.png").string(),
            initial_gray);
        cv::imwrite(
            (output_dir / optimized_file_name).string(),
            optimized_gray);

        cv::Mat initial_color;
        cv::Mat optimized_color;
        cv::applyColorMap(initial_gray, initial_color, cv::COLORMAP_TURBO);
        cv::applyColorMap(optimized_gray, optimized_color, cv::COLORMAP_TURBO);

        constexpr int title_height = 36;
        const auto make_panel = [&](const cv::Mat& image,
                                    const std::string& title) {
            cv::Mat panel(
                image.rows + title_height,
                image.cols,
                CV_8UC3,
                cv::Scalar(255, 255, 255));
            image.copyTo(panel(cv::Rect(
                0, title_height, image.cols, image.rows)));
            cv::putText(
                panel,
                title,
                cv::Point(8, 24),
                cv::FONT_HERSHEY_SIMPLEX,
                0.45,
                cv::Scalar(0, 0, 0),
                1,
                cv::LINE_AA);
            return panel;
        };

        std::ostringstream initial_title;
        initial_title << "Initial | PV loss=" << std::fixed
                      << std::setprecision(3)
                      << init_eval.pvBandResult.pv_loss;
        std::ostringstream optimized_title;
        optimized_title << "Optimized | PV loss=" << std::fixed
                        << std::setprecision(3)
                        << optimized.pvBandResult.pv_loss;

        cv::Mat comparison;
        cv::hconcat(
            make_panel(initial_color, initial_title.str()),
            make_panel(optimized_color, optimized_title.str()),
            comparison);
        cv::imwrite(
            (output_dir / "pv_band_comparison.png").string(),
            comparison);
    };

    // 用于损失函数归一化
    const int ep_count = static_cast<int>(_eps_result.eps.rows());
    const int weighted_ep_count = static_cast<int>(
        (_eps_result.weight_epe.array() == 1.0).count());
    const double ref_mean_wepe =
        init_eval.wepe / std::max(1, weighted_ep_count);
    const double ref_pe = init_eval.pe;
    const double ref_pvband = init_eval.pvBandResult.pv_loss;

    // 三项先除以初始值归一化，再按照配置权重组成最终 cost。
    auto joint_cost = [&](const EvaluateState& state) {
        const double mean_wepe =
            state.wepe / std::max(1, weighted_ep_count);
        return _sraf_config.weight_pvband *
                   state.pvBandResult.pv_loss / ref_pvband +
               _sraf_config.weight_wepe * mean_wepe / ref_mean_wepe +
               _sraf_config.weight_pe * state.pe / ref_pe;
    };

    if (_sraf_config.independent_sraf_widths) {
        if (_sraf_config.independent_width_min >=
                _sraf_config.independent_width_max ||
            _sraf_config.independent_width_max >
                _sraf_config.maximum_half_width ||
            _sraf_config.cma_initial_sigma <= 0.0 ||
            _sraf_config.cma_population_size < 4 ||
            _sraf_config.cma_max_evaluations <
                _sraf_config.cma_population_size) {
            throw std::invalid_argument(
                "SRAF_Optimizer: invalid independent-width CMA-ES parameters");
        }

        struct IndependentWidthRecord {
            int evaluation;
            double cost;
            double normalized_wepe;
            double normalized_pe;
            double normalized_pvband;
            double mean_wepe;
            double pe;
            double pvband;
            std::vector<double> widths;
        };

        std::vector<IndependentWidthRecord> history;
        std::vector<double> best_widths = _initial_half_widths;
        double best_cost = joint_cost(init_eval);
        EvaluateState best_state = init_eval;
        Eigen::MatrixXd best_sraf = _rendered_sraf_mask;
        int evaluation = 0;

        // 记录每次评价的原始指标、归一化指标和完整宽度向量。
        const auto make_record = [&](
            int evaluation_id,
            const std::vector<double>& widths,
            const EvaluateState& state,
            double cost) {
            const double mean_wepe =
                state.wepe / std::max(1, weighted_ep_count);
            return IndependentWidthRecord{
                evaluation_id,
                cost,
                mean_wepe / ref_mean_wepe,
                state.pe / ref_pe,
                state.pvBandResult.pv_loss / ref_pvband,
                mean_wepe,
                state.pe,
                state.pvBandResult.pv_loss,
                widths};
        };

        history.push_back(
            make_record(0, best_widths, init_eval, best_cost));

        std::cout << "\nIndependent-width CMA-ES optimization\n"
                  << "  dimensions      : " << best_widths.size() << '\n'
                  << "  width bounds    : ["
                  << _sraf_config.independent_width_min << ", "
                  << _sraf_config.independent_width_max << "] px\n"
                  << "  population size : "
                  << _sraf_config.cma_population_size << '\n'
                  << "  max evaluations : "
                  << _sraf_config.cma_max_evaluations << '\n'
                  << "  initial cost    : " << best_cost << '\n';

        // CMA-ES objective：宽度向量 -> SRAF mask -> 完整 mask -> 联合 cost。
        auto objective = [&](const Eigen::VectorXd& parameters) {
            std::vector<double> widths(
                parameters.data(), parameters.data() + parameters.size());
            Eigen::MatrixXd sraf = SrafCurve::render_all(
                _sraf_distance_caches, widths);
            Eigen::MatrixXd mask = _rendered_main_mask.cwiseMax(sraf);
            EvaluateState state = evaluate(mask);
            const double cost = joint_cost(state);

            ++evaluation;
            history.push_back(
                make_record(evaluation, widths, state, cost));

            if (cost < best_cost) {
                best_cost = cost;
                best_widths = widths;
                best_sraf = std::move(sraf);
                best_state = std::move(state);
                std::cout << "  new best eval " << std::setw(4) << evaluation
                          << " | cost=" << std::fixed << std::setprecision(6)
                          << best_cost << '\n';
            }
            return cost;
        };

        // 与 Python 版本一致，从宽度上下界的中心开始搜索。
        const double initial_mean_value =
            (_sraf_config.independent_width_min +
             _sraf_config.independent_width_max) /
            2.0;
        const Eigen::VectorXd initial_mean = Eigen::VectorXd::Constant(
            static_cast<Eigen::Index>(_initial_half_widths.size()),
            initial_mean_value);

        const CmaEsResult cma_result = run_bounded_cma_es(
            initial_mean,
            _sraf_config.cma_initial_sigma,
            _sraf_config.independent_width_min,
            _sraf_config.independent_width_max,
            _sraf_config.cma_population_size,
            _sraf_config.cma_max_evaluations,
            _sraf_config.cma_tolerance_x,
            _sraf_config.cma_tolerance_fun,
            _sraf_config.cma_seed,
            objective);

        // 保存搜索过程中找到的全局最优宽度和对应 mask。
        _optimized_half_widths = best_widths;
        _optimized_mask = best_state.mask;

        const std::filesystem::path output_dir(
            _sraf_config.save_file_path);
        std::ofstream history_file(
            output_dir / "independent_width_history.csv");
        history_file
            << "evaluation,cost,norm_wepe,norm_pe,norm_pvband,"
               "mean_wepe,pe,pvband";
        for (std::size_t i = 0; i < best_widths.size(); ++i) {
            history_file << ",w_" << i + 1;
        }
        history_file << '\n' << std::fixed << std::setprecision(6);
        for (const auto& record : history) {
            history_file << record.evaluation << ','
                         << record.cost << ','
                         << record.normalized_wepe << ','
                         << record.normalized_pe << ','
                         << record.normalized_pvband << ','
                         << record.mean_wepe << ','
                         << record.pe << ','
                         << record.pvband;
            for (double width : record.widths) {
                history_file << ',' << width;
            }
            history_file << '\n';
        }

        std::ofstream widths_file(
            output_dir / "optimized_independent_widths.csv");
        widths_file
            << "sraf_index,component_id,initial_half_width,"
               "optimized_half_width\n"
            << std::fixed << std::setprecision(6);
        for (std::size_t i = 0; i < best_widths.size(); ++i) {
            widths_file << i << ','
                        << _sraf_geometry_result.centerlines[i].component_id
                        << ',' << _initial_half_widths[i] << ','
                        << best_widths[i] << '\n';
        }

        cv::imwrite(
            (output_dir / "optimized_independent_sraf_mask.png").string(),
            eigen_mask_to_u8(best_sraf));
        cv::imwrite(
            (output_dir / "optimized_independent_mask.png").string(),
            eigen_mask_to_u8(best_state.mask));
        save_pvband_comparison(
            best_state,
            output_dir,
            "optimized_independent_pv_band_map.png");

        const auto [minimum_width, maximum_width] = std::minmax_element(
            best_widths.begin(), best_widths.end());
        std::cout << "\nBest independent widths\n"
                  << "  cost        : " << best_cost << '\n'
                  << "  width range : [" << *minimum_width << ", "
                  << *maximum_width << "] px\n"
                  << "  evaluations : " << cma_result.evaluations << '\n'
                  << "  generations : " << cma_result.generations << '\n';
        for (std::size_t i = 0; i < best_widths.size(); ++i) {
            std::cout << "  SRAF " << std::setw(2) << i + 1
                      << " (component "
                      << _sraf_geometry_result.centerlines[i].component_id
                      << ") : " << best_widths[i] << " px\n";
        }
        std::cout << "  history     : "
                  << (output_dir / "independent_width_history.csv")
                  << '\n';
        return;
    }

    // 以下为所有 SRAF 共用一个半宽时的一维搜索记录。
    struct WidthRecord {
        std::string stage;
        double width;
        double cost;
        double mean_wepe;
        double mean_epe;
        double pe;
        double pvband;
    };
    std::vector<WidthRecord> history;

    double best_width = _sraf_config.initial_half_width;
    double best_cost = joint_cost(init_eval);
    EvaluateState best_state = init_eval;
    Eigen::MatrixXd best_sraf = _rendered_sraf_mask;
    history.push_back({
        "initial",
        best_width,
        best_cost,
        ref_mean_wepe,
        init_eval.epe / std::max(1, ep_count),
        init_eval.pe,
        init_eval.pvBandResult.pv_loss});

    auto print_record = [](const WidthRecord& record) {
        std::cout << "  " << std::left << std::setw(7) << record.stage
                  << std::right << std::fixed << std::setprecision(6)
                  << " w=" << std::setw(5) << record.width
                  << " | cost=" << std::setw(10) << record.cost
                  << " | mean_wEPE=" << std::setw(10) << record.mean_wepe
                  << " | mean_EPE=" << std::setw(10) << record.mean_epe
                  << " | PE=" << std::setw(12) << record.pe
                  << " | PV=" << std::setw(12) << record.pvband << '\n';
    };

    std::cout << "\nShared-width optimization\n";
    print_record(history.front());

    // 避免粗搜索和细搜索重复评价同一个宽度。
    auto was_evaluated = [&](double width) {
        return std::any_of(
            history.begin(), history.end(),
            [&](const WidthRecord& record) {
                return std::abs(record.width - width) < 1e-9;
            });
    };

    // 渲染一个共享宽度并更新当前最优结果。
    auto evaluate_width = [&](double width, const std::string& stage) {
        if (was_evaluated(width)) return;

        Eigen::MatrixXd sraf = SrafCurve::render_all(
            _sraf_distance_caches,
            {width});
        Eigen::MatrixXd mask = _rendered_main_mask.cwiseMax(sraf);
        EvaluateState state = evaluate(mask);
        WidthRecord record{
            stage,
            width,
            joint_cost(state),
            state.wepe / std::max(1, weighted_ep_count),
            state.epe / std::max(1, ep_count),
            state.pe,
            state.pvBandResult.pv_loss};
        history.push_back(record);
        print_record(record);

        if (record.cost < best_cost) {
            best_width = width;
            best_cost = record.cost;
            best_sraf = std::move(sraf);
            best_state = std::move(state);
        }
    };

    // 第一阶段：在完整宽度区间内进行粗搜索。
    const int coarse_count = static_cast<int>(std::round(
        (_sraf_config.shared_width_max - _sraf_config.shared_width_min) /
        _sraf_config.shared_width_coarse_step));
    for (int i = 0; i <= coarse_count; ++i) {
        evaluate_width(
            _sraf_config.shared_width_min +
                i * _sraf_config.shared_width_coarse_step,
            "coarse");
    }

    // 第二阶段：在粗搜索最优值附近进行细搜索。
    const double fine_min = std::max(
        _sraf_config.shared_width_min,
        best_width - _sraf_config.shared_width_coarse_step);
    const double fine_max = std::min(
        _sraf_config.shared_width_max,
        best_width + _sraf_config.shared_width_coarse_step);
    const int fine_count = static_cast<int>(std::round(
        (fine_max - fine_min) / _sraf_config.shared_width_fine_step));
    for (int i = 0; i <= fine_count; ++i) {
        evaluate_width(
            fine_min + i * _sraf_config.shared_width_fine_step,
            "fine");
    }

    // 保存共享宽度模式的最优数值、mask 和搜索历史。
    _optimized_half_width = best_width;
    _optimized_half_widths = {best_width};
    _optimized_mask = best_state.mask;

    const std::filesystem::path output_dir(_sraf_config.save_file_path);
    std::ofstream history_file(output_dir / "shared_width_history.csv");
    history_file << "stage,half_width,cost,mean_wepe,mean_epe,pe,pvband\n"
                 << std::fixed << std::setprecision(6);
    for (const auto& record : history) {
        history_file << record.stage << ','
                     << record.width << ','
                     << record.cost << ','
                     << record.mean_wepe << ','
                     << record.mean_epe << ','
                     << record.pe << ','
                     << record.pvband << '\n';
    }

    cv::imwrite(
        (output_dir / "optimized_shared_sraf_mask.png").string(),
        eigen_mask_to_u8(best_sraf));
    cv::imwrite(
        (output_dir / "optimized_shared_mask.png").string(),
        eigen_mask_to_u8(best_state.mask));
    save_pvband_comparison(
        best_state,
        output_dir,
        "optimized_shared_pv_band_map.png");

    std::cout << "\nBest shared width\n"
              << "  half width : " << best_width << " px\n"
              << "  full width : " << 2.0 * best_width << " px\n"
              << "  cost       : " << best_cost << '\n'
              << "  history    : "
              << (output_dir / "shared_width_history.csv") << '\n';
}

}  // namespace litho
