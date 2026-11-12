#include "sraf_printing_repair.h"
#include "console_log.h"
#include "imaging.h"
#include "lithography_simulator.h"
#include "loss.h"
#include "save_txt.h"
#include <yaml-cpp/yaml.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;
using namespace litho;

namespace {

// 将完整路径作为一个 POSIX shell 参数传递，保护空格、单引号等字符，避免改变命令结构。
std::string shell_quote(const fs::path& path) {
    std::string quoted = "'";
    for (char character : path.string()) {
        if (character == '\'') quoted += "'\"'\"'";
        else quoted += character;
    }
    return quoted + "'";
}

// 输入项目根目录和本次实际输出目录，调用已有脚本生成连续 WI 对比及其他诊断图。
// 优先使用项目 Python 环境；失败时保留数值结果并抛出错误，避免误报已经生成图片。
void save_comparison_figures(const fs::path& project_root, const fs::path& output_dir) {
    const fs::path script = project_root / "scripts/plot_sraf_printing_repair.py";
    if (!fs::is_regular_file(script)) throw std::runtime_error("Missing comparison script: " + script.string());
    const fs::path venv_python = project_root / ".venv/bin/python";
    const std::string python = fs::is_regular_file(venv_python) ? shell_quote(venv_python) : "python3";
    const fs::path plot_cache = project_root / ".cache/matplotlib";
    fs::create_directories(plot_cache);
    const std::string command = "MPLBACKEND=Agg MPLCONFIGDIR=" + shell_quote(plot_cache) + " " + python + " " +
                                shell_quote(script) + " " + shell_quote(output_dir);
    std::cout << "Generating WI comparison: " << command << '\n' << std::flush;
    if (std::system(command.c_str()) != 0) {
        throw std::runtime_error("Comparison plotting failed; numerical results are saved in " + output_dir.string());
    }
    const fs::path comparison = output_dir / "comparison.png";
    if (!fs::is_regular_file(comparison)) throw std::runtime_error("Comparison script did not create " + comparison.string());
    std::cout << "WI comparison saved: " << comparison << '\n';
}

struct RepairConfig {
    SrafShrinkConfig geometry; // 厚度筛选、剩余厚度目标及单点最大位移，长度单位均为 pixel。
    SrafPairGroupConfig pair_groups; // 每轮联合更新的组数和重复截面筛选参数。
    std::string mode = "single_sraf_multi_pair"; // 几何点对方法或法向有限差分梯度方法，每次均只处理一块。
    double fd_step_px = 0.05; // 法向中心差分的扰动距离，单位 pixel；不等于实际更新距离。
    double learning_rate = 1000.0; // 将损失对法向位移的导数转换为请求位移，再由 step_px 限幅。
    int max_iterations = 30; // 全局循环轮数上限，包含选块和结束块的检查轮次。
    int max_iterations_per_sraf = 30; // 每块允许接受的更新轮数；结束后才处理下一块。
    int max_backtracks = 6; // 全局试探和每组几何筛选各自允许将位移减半的次数。
    int target_guard_px = 3; // 主图形边缘排除带的正方形膨胀半径，单位 pixel。
    double intensity_margin = 0.01; // 本块 AI 停止目标相对源显影阈值降低的余量，与 AI 使用同一归一化强度。
    double max_mean_epe_increase_nm = 0.05; // 相对初始快照的平均绝对 EPE 增量上限，单位 nm。
    double max_max_epe_increase_nm = 0.25; // 相对初始快照的最大绝对 EPE 增量上限，单位 nm。
    double max_added_sraf_area_px = 0.25; // 相对初始轮廓新增的灰度覆盖面积上限，单位 pixel²。
    double snapshot_tolerance = 2e-5; // 重绘 mask 和重新成像与源快照的最大绝对差值容差。
};

struct RepairState {
    Eigen::MatrixXd sraf_mask; // N×N 的 SRAF 灰度覆盖率，N 为源快照边长像素数。
    Eigen::MatrixXd mask; // N×N 的完整掩模，等于固定主图形加本轮 SRAF。
    Imaging_Result image; // N×N 空间像强度及 sigmoid 显影图。
    Eigen::RowVectorXd epe_nm; // 源快照各 EP 的绝对 EPE，单位 nm。
    cv::Mat hotspots; // 当前达到/超过 AI 安全目标的背景像素，0/255；坐标为整图零基像素索引。
    cv::Mat printing; // AI>源显影阈值的 0/255 统计标记，不由 WI 二值化生成。
    cv::Mat background_window; // 0/255 背景检测范围：Guard 外及 Guard 内的独立超阈值区域，不表示是否超阈值。
    double background_peak = 0.0; // 主图形 Guard 外或独立额外显影连通域内的最大空间像强度。
    double background_wi_peak = 0.0; // 同一背景检测范围内的最大连续 WI，仅用于输出和核查。
    double mean_epe_nm = 0.0; // 源快照全部 EP 上绝对 EPE 的平均值，单位 nm。
    double max_epe_nm = 0.0; // 同一组 EP 的最大绝对 EPE，单位 nm。
    double sraf_area_px = 0.0; // MSAA 灰度覆盖之和，单位为像素面积。
    int background_printed_px = 0; // Guard 外 AI>源显影阈值的像素数。
    int raw_background_printed_px = 0; // target<=0.5 的全部像素中 AI 超阈值数量，含主图形边缘外溢。
    int detached_components = 0; // 与 target 没有任何重叠的显影连通域，8 邻接。
    int detached_pixels = 0; // 所有独立 AI 超阈值连通域包含的像素总数。
};

struct NormalGradientSample {
    SrafShrinkPoint point; // 本轮固定的控制点编号、单位内法线及材料厚度。
    double fd_step_px = 0.0; // 实际使用的双侧差分距离，单位 pixel。
    double loss_plus = std::numeric_limits<double>::quiet_NaN(); // 沿内法线正向扰动后的损失。
    double loss_minus = std::numeric_limits<double>::quiet_NaN(); // 沿内法线负向扰动后的损失。
    double gradient = 0.0; // (loss_plus-loss_minus)/(2*fd_step_px)，对内向位移求导。
    double requested_step_px = 0.0; // 投影为只向内后的请求位移，失败回退时整体减半。
    bool valid = false; // 双侧扰动均有有效曲线和有限损失时才参与更新。
};

// 从注释分组的 TXT 读取完整控制点，输出保留原轮廓顺序的 (y,x) 浮点像素坐标。
// 空行或 # 行结束当前轮廓，文件末尾也结束一次，每条 CR 轮廓至少需要四个控制点。
ControlPoints load_controls(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open controls: " + path.string());
    ControlPoints contours;
    std::vector<Eigen::Vector2d> points;
    auto finish_contour = [&]() {
        if (points.empty()) return;
        if (points.size() < 4) throw std::runtime_error("CR contour requires at least four controls");
        Polygon contour(points.size(), 2);
        for (int i = 0; i < points.size(); ++i) contour.row(i) = points[i].transpose();
        contours.push_back(contour);
        points.clear();
    };
    std::string line;
    while (std::getline(input, line)) {
        const auto first_content = line.find_first_not_of(" \t\r");
        if (first_content == std::string::npos || line[first_content] == '#') {
            finish_contour();
            continue;
        }
        std::istringstream values(line);
        double y, x;
        std::string extra;
        if (!(values >> y >> x) || (values >> extra) || !std::isfinite(y) || !std::isfinite(x)) {
            throw std::runtime_error("Invalid control coordinate in " + path.string());
        }
        points.emplace_back(y, x);
    }
    finish_contour();
    return contours;
}

// 将候选 CR 控制点写成注释分组的 (y,x) 像素坐标 TXT，保留轮廓及点编号。
void save_controls(const fs::path& path, const ControlPoints& sraf_controls) {
    std::ofstream output_file(path);
    output_file << std::fixed << std::setprecision(10);
    for (int contour_index = 0; contour_index < sraf_controls.size(); ++contour_index) {
        output_file << "# sraf block " << contour_index << '\n' << sraf_controls[contour_index] << '\n';
    }
    if (!output_file) throw std::runtime_error("Cannot write controls: " + path.string());
}

// 输入任意尺寸实数矩阵，按原行列顺序写 TXT；矩阵的单位由调用位置和文件名确定。
void save_matrix(const fs::path& path, const Eigen::MatrixXd& matrix) {
    std::ofstream output_file(path);
    output_file << std::fixed << std::setprecision(10) << matrix << '\n';
    if (!output_file) throw std::runtime_error("Cannot write matrix: " + path.string());
}

// 将灰度 mask 的 >0.5 区域变成同尺寸 0/255 图，用于连通域及边缘排除带计算。
cv::Mat foreground(const Eigen::MatrixXd& mask) {
    cv::Mat result(mask.rows(), mask.cols(), CV_8UC1);
    for (int y = 0; y < mask.rows(); ++y) {
        for (int x = 0; x < mask.cols(); ++x) result.at<unsigned char>(y, x) = mask(y, x) > 0.5 ? 255 : 0;
    }
    return result;
}

// 从独立实验 YAML 的 repair 段读取几何参数和接受容差，迭代前校验统计阈值。
RepairConfig load_config(const YAML::Node& root) {
    const auto repair_node = root["repair"];
    RepairConfig config;
    // 先读取并验证方法，再依次读取几何、迭代、成像容差，可省略字段沿用结构体中的默认值。
    if (repair_node["mode"]) config.mode = repair_node["mode"].as<std::string>();
    if (config.mode != "single_sraf_multi_pair" && config.mode != "single_sraf_normal_gradient") {
        throw std::invalid_argument("Unknown SRAF repair mode");
    }
    if (repair_node["normal_gradient"]) {
        const auto gradient_node = repair_node["normal_gradient"];
        if (gradient_node["fd_step_px"]) config.fd_step_px = gradient_node["fd_step_px"].as<double>();
        if (gradient_node["learning_rate"]) config.learning_rate = gradient_node["learning_rate"].as<double>();
    }
    if (repair_node["pair_groups"]) {
        config.pair_groups.maximum_pairs = repair_node["pair_groups"]["maximum_pairs"].as<int>();
        config.pair_groups.minimum_center_spacing_px = repair_node["pair_groups"]["minimum_center_spacing_px"].as<double>();
    }
    config.geometry.step_px = repair_node["step_px"].as<double>();
    config.geometry.width_threshold_px = repair_node["width_threshold_px"].as<double>();
    config.geometry.minimum_width_px = repair_node["minimum_width_px"].as<double>();
    config.max_iterations = repair_node["max_iterations"].as<int>();
    config.max_iterations_per_sraf = config.max_iterations;
    if (repair_node["max_iterations_per_sraf"]) config.max_iterations_per_sraf = repair_node["max_iterations_per_sraf"].as<int>();
    config.max_backtracks = repair_node["max_backtracks"].as<int>();
    config.target_guard_px = repair_node["target_guard_px"].as<int>();
    if (repair_node["intensity_margin"]) config.intensity_margin = repair_node["intensity_margin"].as<double>();
    config.max_mean_epe_increase_nm = repair_node["max_mean_epe_increase_nm"].as<double>();
    config.max_max_epe_increase_nm = repair_node["max_max_epe_increase_nm"].as<double>();
    config.max_added_sraf_area_px = repair_node["max_added_sraf_area_px"].as<double>();
    config.snapshot_tolerance = repair_node["snapshot_tolerance"].as<double>();
    if (config.max_iterations < 1 || config.max_backtracks < 0 || config.target_guard_px < 0) {
        throw std::invalid_argument("Invalid iteration/Guard count");
    }
    if (config.max_iterations_per_sraf < 1) throw std::invalid_argument("Invalid per-SRAF iteration limit");
    if (!std::isfinite(config.fd_step_px) || config.fd_step_px <= 0.0 ||
        !std::isfinite(config.learning_rate) || config.learning_rate <= 0.0) {
        throw std::invalid_argument("Normal-gradient step and learning rate must be finite and positive");
    }
    for (double value : {config.geometry.step_px, config.geometry.width_threshold_px, config.geometry.minimum_width_px}) {
        if (!std::isfinite(value) || value <= 0.0) throw std::invalid_argument("Pair geometry values must be finite and positive");
    }
    if (config.geometry.width_threshold_px < config.geometry.minimum_width_px) throw std::invalid_argument("Invalid pair width limits");
    if (config.pair_groups.maximum_pairs < 1 || !std::isfinite(config.pair_groups.minimum_center_spacing_px) ||
        config.pair_groups.minimum_center_spacing_px < 0.0) throw std::invalid_argument("Invalid pair-group limits");
    for (double value : {config.intensity_margin, config.max_mean_epe_increase_nm, config.max_max_epe_increase_nm,
                         config.max_added_sraf_area_px}) {
        if (!std::isfinite(value) || value < 0.0) throw std::invalid_argument("Repair tolerances must be finite and nonnegative");
    }
    if (!std::isfinite(config.snapshot_tolerance) || config.snapshot_tolerance <= 0.0) {
        throw std::invalid_argument("Invalid snapshot tolerance");
    }
    return config;
}

// 输入已完成成像且统计量为默认值的 state、同尺寸 target/Guard 和强度阈值，填充背景检测范围与统计量。
// 所有矩阵使用整图零基 (y,x) 像素索引，AI 和阈值均为源成像模型的归一化强度。
void update_printing_statistics(RepairState& state, const Eigen::MatrixXd& target_mask, const cv::Mat& target_guard,
                                double threshold, double safe_threshold) {
    state.hotspots = cv::Mat(target_mask.rows(), target_mask.cols(), CV_8UC1, cv::Scalar(0));
    state.printing = cv::Mat(target_mask.rows(), target_mask.cols(), CV_8UC1, cv::Scalar(0));
    cv::bitwise_not(target_guard, state.background_window);
    cv::Mat above_threshold(target_mask.rows(), target_mask.cols(), CV_8UC1, cv::Scalar(0)), component_labels;
    // printing 直接比较 AI 与源显影阈值；较低的安全目标只用于本块停止及移动日志热点。
    for (int y = 0; y < target_mask.rows(); ++y) {
        for (int x = 0; x < target_mask.cols(); ++x) {
            const double intensity = state.image.aerial_image(y, x);
            const bool above_printing_threshold = intensity > threshold;
            if (above_printing_threshold) above_threshold.at<unsigned char>(y, x) = 255;
            if (target_mask(y, x) <= 0.5 && above_printing_threshold) ++state.raw_background_printed_px;
            if (target_guard.at<unsigned char>(y, x) != 0) continue;
            state.background_peak = std::max(state.background_peak, intensity);
            state.background_wi_peak = std::max(state.background_wi_peak, state.image.wafer_image(y, x));
            if (above_printing_threshold) {
                ++state.background_printed_px;
                state.printing.at<unsigned char>(y, x) = 255;
            }
            if (intensity >= safe_threshold) state.hotspots.at<unsigned char>(y, x) = 255;
        }
    }
    // 连通域统计以 AI 超阈值位置为输入，标签 0 是背景，其他标签按是否接触 target 区分主图形和独立显影。
    const int label_count = cv::connectedComponents(above_threshold, component_labels, 8);
    std::vector<bool> overlaps_target(label_count, false);
    for (int y = 0; y < target_mask.rows(); ++y) {
        for (int x = 0; x < target_mask.cols(); ++x) {
            if (target_mask(y, x) > 0.5) overlaps_target[component_labels.at<int>(y, x)] = true;
        }
    }
    for (int label = 1; label < label_count; ++label) if (!overlaps_target[label]) ++state.detached_components;
    // 将独立显影补入背景范围，包括处在 Guard 内的部分，使靠近主图形的独立 printing 仍可检测。
    for (int y = 0; y < target_mask.rows(); ++y) {
        for (int x = 0; x < target_mask.cols(); ++x) {
            const int label = component_labels.at<int>(y, x);
            if (label == 0 || overlaps_target[label]) continue;
            ++state.detached_pixels;
            state.background_window.at<unsigned char>(y, x) = 255;
            state.printing.at<unsigned char>(y, x) = 255;
            state.hotspots.at<unsigned char>(y, x) = 255;
            state.background_peak = std::max(state.background_peak, state.image.aerial_image(y, x));
            state.background_wi_peak = std::max(state.background_wi_peak, state.image.wafer_image(y, x));
        }
    }
}

// 输入本块初始覆盖、锁定时完整 AI 和主图形 Guard，返回同尺寸且在本阶段固定的评价窗口。
// 外扩半径为 3 pixel，每轴 [-3,3] 包含端点，孔洞和轮廓编号均沿用 region 的原始归属。
cv::Mat make_sraf_evaluation_window(const SrafRegion& region, const RepairState& state, const cv::Mat& target_guard,
                                    double threshold) {
    cv::Mat window = foreground(region.initial_mask);
    // 本块初始材料外扩 3 pixel，覆盖每轴 [-3,3] 的邻域；只限制评价窗口，不裁切成像。
    constexpr int window_radius_px = 3;
    constexpr int window_diameter = 2 * window_radius_px + 1;
    cv::dilate(window, window, cv::getStructuringElement(cv::MORPH_RECT, {window_diameter, window_diameter}));
    window.setTo(0, target_guard);
    // 独立显影即使位于主图形 Guard 内，也保留在本块的评价窗口内。
    for (int y = 0; y < state.image.aerial_image.rows(); ++y) {
        for (int x = 0; x < state.image.aerial_image.cols(); ++x) {
            const bool in_region = region.initial_mask(y, x) > 0.0;
            const bool in_background = state.background_window.at<unsigned char>(y, x) != 0;
            if (in_region && in_background && state.image.aerial_image(y, x) > threshold) {
                window.at<unsigned char>(y, x) = 255;
            }
        }
    }
    return window;
}

// 对完整成像结果检查 EPE 和全背景 printing 数量；几何方法还要求本块 AI 峰值下降。
// 梯度方法单独检查损失下降，停止条件仍检查 AI 峰值，避免把低平均损失误当成所有像素达标。
// 输出首个拒绝原因，空字符串表示通过，EPE 上限固定参考初始状态，强度改善参考当前状态。
std::string imaging_rejection(const RepairState& candidate_state, const RepairState& current_state, const RepairState& initial_state,
                              const RepairConfig& config, double current_peak, double next_peak) {
    if (!candidate_state.image.aerial_image.allFinite() || !candidate_state.epe_nm.allFinite()) return "nonfinite_imaging_or_epe";
    if (candidate_state.mean_epe_nm > initial_state.mean_epe_nm + config.max_mean_epe_increase_nm) return "mean_epe_limit";
    if (candidate_state.max_epe_nm > initial_state.max_epe_nm + config.max_max_epe_increase_nm) return "max_epe_limit";
    if (candidate_state.detached_components > current_state.detached_components ||
        candidate_state.background_printed_px > current_state.background_printed_px) {
        return "printing_increased";
    }
    if (config.mode == "single_sraf_multi_pair" && next_peak >= current_peak - 1e-7) return "peak_did_not_decrease";
    return {};
}

// 将一个已评估状态的图像、控制点及统计值写入指定目录，供前后对照和独立核查。
void save_state(const fs::path& directory, const RepairState& state, const ControlPoints& sraf_controls, const ParametricDemo& renderer) {
    fs::create_directories(directory);
    save_controls(directory / "sraf_cps.txt", sraf_controls);
    // 保存 C++ 实际使用的曲线采样点，并显式重复首点闭合；绘图不用灰度等值线代替拟合曲线。
    std::ofstream curves_file(directory / "sraf_curves.txt");
    const auto curves = renderer.get_curve_points(sraf_controls);
    curves_file << std::fixed << std::setprecision(10);
    for (int contour = 0; contour < curves.size(); ++contour) {
        curves_file << "# sraf curve " << contour << '\n' << curves[contour] << '\n' << curves[contour].row(0) << "\n\n";
    }
    if (!curves_file) throw std::runtime_error("Cannot write fitted SRAF curves");
    save_matrix(directory / "sraf_mask.txt", state.sraf_mask);
    save_matrix(directory / "mask.txt", state.mask);
    save_matrix(directory / "aerial.txt", state.image.aerial_image);
    save_matrix(directory / "wafer.txt", state.image.wafer_image);
    save_matrix(directory / "epe_vector_nm.txt", state.epe_nm);
    std::ofstream metrics(directory / "metrics.txt");
    metrics << std::setprecision(12) << "background_peak " << state.background_peak
            << "\nbackground_wi_peak " << state.background_wi_peak
            << "\nbackground_printed_px " << state.background_printed_px
            << "\nraw_background_printed_px " << state.raw_background_printed_px
            << "\ndetached_components " << state.detached_components << "\ndetached_pixels " << state.detached_pixels
            << "\nmean_epe_nm " << state.mean_epe_nm << "\nmax_epe_nm " << state.max_epe_nm
            << "\nsraf_area_px " << state.sraf_area_px << '\n';
    if (!metrics) throw std::runtime_error("Cannot write repair metrics");
}

// 在固定的本块评价窗口内取空间像峰值；窗口坐标与完整输入图像一致，不裁剪成像输入。
double peak_in_sraf_window(const Eigen::MatrixXd& aerial, const cv::Mat& window) {
    double peak = 0.0;
    for (int y = 0; y < aerial.rows(); ++y) {
        for (int x = 0; x < aerial.cols(); ++x) {
            if (window.at<unsigned char>(y, x) != 0) peak = std::max(peak, aerial(y, x));
        }
    }
    return peak;
}

} // namespace

// 独立实验入口：固定主图形，逐块处理 AI 超阈值 SRAF，可选宽点对内缩或法向有限差分梯度下降。
int main(int argc, char** argv) {
    try {
        // 相对路径统一基于项目根目录，严格使用 YAML 指定的输出目录，重复运行覆盖本次生成的同名文件。
        const fs::path project_root = fs::path(__FILE__).parent_path().parent_path();
        fs::path config_path = project_root / "config_sraf_printing_repair.yaml";
        if (argc == 3 && std::string(argv[1]) == "--config") config_path = fs::absolute(argv[2]);
        else if (argc != 1) throw std::invalid_argument("Usage: demo_sraf_printing_repair [--config file.yaml]");
        const YAML::Node root = YAML::LoadFile(config_path.string());
        const RepairConfig config = load_config(root);
        auto resolve = [&](const std::string& value) { return fs::path(value).is_absolute() ? fs::path(value) : project_root / value; };
        const fs::path source_dir = resolve(root["source_dir"].as<std::string>());
        const std::string snapshot = root["snapshot"].as<std::string>();
        if (snapshot != "best_wepe" && snapshot != "best_epe") throw std::invalid_argument("snapshot must be best_wepe or best_epe");
        const fs::path output_dir = resolve(root["output_dir"].as<std::string>());
        fs::create_directories(output_dir);
        ConsoleLog log(output_dir / "console_output.log");
        // 复制文件也显式允许覆盖，否则第二次运行会因旧配置快照已存在而报错。
        fs::copy_file(config_path, output_dir / "config_used.yaml", fs::copy_options::overwrite_existing);
        fs::copy_file(source_dir / "config_used.yaml", output_dir / "source_config_used.yaml", fs::copy_options::overwrite_existing);
        std::cout << "Source: " << source_dir << "\nOutput: " << output_dir << '\n';

        // 成像和曲线表示取自源快照配置，threshold 与 AI 使用同一强度归一化。
        const YAML::Node source_config = YAML::LoadFile((source_dir / "config_used.yaml").string());
        if (source_config["meef"]["sraf_curve_type"].as<std::string>() != "CR" ||
            source_config["meef"]["rasterizer"].as<std::string>() != "msaa" ||
            source_config["meef"]["epe_mode"].as<std::string>() != "pytorch") {
            throw std::invalid_argument("This experiment requires saved CR/MSAA geometry and pytorch EPE");
        }
        const SimulationParameters simulation_params = SimulationParameters::from_yaml((source_dir / "config_used.yaml").string());
        const double threshold = simulation_params.resist.threshold, alpha = simulation_params.resist.alpha;
        // 检测直接采用源 AI 阈值；固定强度余量只控制已锁定 SRAF 的停止目标。
        const double safe_threshold = threshold - config.intensity_margin;
        if (!std::isfinite(safe_threshold) || safe_threshold <= 0.0) {
            throw std::invalid_argument("intensity_margin must give a finite positive AI safety target");
        }
        std::cout << std::setprecision(12) << "AI printing threshold=" << threshold << ", safety target=" << safe_threshold << '\n';
        // 读取 OPC 优化后的快照数据，检查尺寸和数值有效性；SRAF 曲线必须至少有一个轮廓。
        Eigen::MatrixXd target_mask, saved_sraf_mask, saved_full_mask, saved_aerial_image, edge_points;
        SaveTxt::load_txt((source_dir / "target_mask.txt").string(), target_mask);
        SaveTxt::load_txt((source_dir / "sraf_mask.txt").string(), saved_sraf_mask);
        SaveTxt::load_txt((source_dir / snapshot / "mask.txt").string(), saved_full_mask);
        SaveTxt::load_txt((source_dir / snapshot / "aerial.txt").string(), saved_aerial_image);
        SaveTxt::load_txt((source_dir / "eps.txt").string(), edge_points);
        for (const auto* matrix : {&target_mask, &saved_sraf_mask, &saved_full_mask, &saved_aerial_image}) {
            const int expected_size = simulation_params.system.grid_size_pixel;
            if (matrix->rows() != expected_size || matrix->cols() != expected_size || !matrix->allFinite()) {
                throw std::invalid_argument("Snapshot dimensions/values do not match its saved optical configuration");
            }
        }
        // target 和各幅图像均为 N×N，edge_points 为 EP 数量×2 的整图 (y,x) 像素坐标。
        if (edge_points.cols() != 2 || edge_points.rows() == 0 || !edge_points.allFinite()) {
            throw std::invalid_argument("Invalid saved EP coordinates");
        }
        // 保存快照包含主图形+固定 SRAF；相减得到同一时刻主图形，后续所有试探均保持它不变。
        const Eigen::MatrixXd main_mask = (saved_full_mask - saved_sraf_mask).cwiseMax(0.0);
        ControlPoints sraf_controls = load_controls(source_dir / "sraf_cps.txt");
        if (sraf_controls.empty()) throw std::invalid_argument("No saved SRAF contours");
        ParametricDemo renderer("CR", target_mask, source_config["meef"]["msaa_level"].as<int>(), "msaa");
        const Eigen::MatrixXd initial_sraf = renderer.render_curve_even_odd(sraf_controls);
        const double render_error = (initial_sraf - saved_sraf_mask).cwiseAbs().maxCoeff();
        if (render_error > config.snapshot_tolerance) {
            throw std::runtime_error("SRAF rerender differs from snapshot: " + std::to_string(render_error));
        }
        // 构建整幅图像的 SOCS 成像系统，几何长度用 pixel，系统像素间距和波长用 nm。
        Grid grid(simulation_params.system.grid_size_pixel, simulation_params.system.pixel_size_nm,
                  simulation_params.optics.na, simulation_params.system.wavelength_nm);
        Pupil pupil(LithographySimulator::_make_pupil_params(simulation_params), grid.grid_coords().Fx_2d, grid.grid_coords().Fy_2d);
        Source source_model(LithographySimulator::_make_source_params(simulation_params, grid.grid_coords().Fx_1d));
        LithoPrepare prepare(grid, pupil, source_model, true);
        Imaging imaging(prepare.cache());
        // 主图形膨胀半径为 target_guard_px，核覆盖每轴 [-r,r]，包括两个端点。
        const cv::Mat target_binary = foreground(target_mask);
        cv::Mat guard;
        const int diameter = 2 * config.target_guard_px + 1;
        cv::dilate(target_binary, guard, cv::getStructuringElement(cv::MORPH_RECT, {diameter, diameter}));

        // 成像始终使用完整 mask。printing 比较连续 AI 与源显影阈值，WI 仅保存并统计峰值供核查。
        // Guard 只改变背景检测范围；其内部独立的 AI 超阈值区域也保留，避免漏掉靠近主图形的 printing。
        auto evaluate_candidate = [&](const Eigen::MatrixXd& candidate_sraf_mask) {
            // 每次都创建新的状态，先完整成像并评价源 EP，再从同一幅 AI 构造检测范围和统计量。
            RepairState state;
            state.sraf_mask = candidate_sraf_mask;
            state.mask = main_mask + candidate_sraf_mask;
            state.image = imaging.compute(state.mask, threshold, alpha);
            state.epe_nm = Loss::evaluate_pytorch_epe(state.image.aerial_image, edge_points, threshold,
                                                     simulation_params.system.pixel_size_nm).absolute_epe_nm;
            state.mean_epe_nm = state.epe_nm.mean();
            state.max_epe_nm = state.epe_nm.maxCoeff();
            state.sraf_area_px = candidate_sraf_mask.sum();
            update_printing_statistics(state, target_mask, guard, threshold, safe_threshold);
            return state;
        };

        // 初始状态作为 EPE 容差的固定参照，current_state 只在候选通过本方法的接受条件后提交。
        RepairState current_state = evaluate_candidate(initial_sraf);
        const RepairState initial_state = current_state;

        // 初试算的 mask 和 aerial 与快照一致，允许有微小数值差异，避免不同编译器或优化级别导致的浮点误差。
        const double aerial_error = (current_state.image.aerial_image - saved_aerial_image).cwiseAbs().maxCoeff();
        std::cout << "Snapshot checks: mask max error=" << render_error << ", aerial max error=" << aerial_error << '\n';
        if (aerial_error > config.snapshot_tolerance) throw std::runtime_error("Recomputed imaging differs from saved snapshot");
        save_matrix(output_dir / "target_mask.txt", target_mask);
        save_matrix(output_dir / "main_mask.txt", main_mask);
        fs::copy_file(source_dir / snapshot / "control_points.txt", output_dir / "main_cps.txt", fs::copy_options::overwrite_existing);
        save_state(output_dir / "initial", current_state, sraf_controls, renderer);

        // 分别记录每轮状态、全部步长试探和逐点位移，CSV 字段的长度仍按 pixel 或 nm 标注。
        std::ofstream history(output_dir / "history.csv"), trials(output_dir / "trials.csv"), movements(output_dir / "movements.csv");
        history << "iteration,step_px,moved_controls,background_peak,background_printed_px,detached_components,detached_pixels,"
                   "raw_background_printed_px,mean_epe_nm,max_epe_nm,sraf_area_px\n" << std::setprecision(12);
        trials << "iteration,backtrack,step_px,moved_controls,accepted,reason\n" << std::setprecision(12);
        movements << "iteration,contour,point,old_y_px,old_x_px,new_y_px,new_x_px,width_px,hotspot_distance_px,"
                     "normal_y,normal_x,movement_px\n" << std::setprecision(12);

        auto record_iteration = [&](int iteration, double nominal_step, int moved_controls, const RepairState& state) {
            history << iteration << ',' << nominal_step << ',' << moved_controls << ',' << state.background_peak
                    << ',' << state.background_printed_px
                    << ',' << state.detached_components << ',' << state.detached_pixels << ',' << state.raw_background_printed_px
                    << ',' << state.mean_epe_nm << ',' << state.max_epe_nm << ',' << state.sraf_area_px << '\n';
            history.flush();
            std::cout << "Iteration " << iteration << ": moved=" << moved_controls << ", peak=" << state.background_peak
                      << ", printed=" << state.background_printed_px << ", detached=" << state.detached_components
                      << ", mean EPE=" << state.mean_epe_nm << " nm, max EPE=" << state.max_epe_nm << " nm\n" << std::flush;
        };
        record_iteration(0, 0.0, 0, current_state);

        // 逐点记录同轮的更新前后坐标、曲线法线和实际位移，便于复核联合更新。
        auto record_control_movements = [&](int iteration, const SrafShrinkProposal& proposal) {
            for (const auto& point : proposal.points) {
                if (point.movement_px <= 0.0) continue;
                const auto old = sraf_controls[point.contour].row(point.point);
                const auto moved = proposal.control_points[point.contour].row(point.point);
                movements << iteration << ',' << point.contour << ',' << point.point << ',' << old(0) << ',' << old(1)
                          << ',' << moved(0) << ',' << moved(1) << ',' << point.width_px << ',' << point.hotspot_distance_px
                          << ',' << point.inward_yx(0) << ',' << point.inward_yx(1) << ',' << point.movement_px << '\n';
            }
            movements.flush();
        };

        // 两种方法的周期 CR 都按循环索引闭合，只对有变化的采样曲线检查自交，不检查 mask 栅格连通性。
        auto geometry_rejection = [&](const ControlPoints& candidate_controls, const Eigen::MatrixXd& candidate_sraf_mask) -> std::string {
            const auto candidate_curves = renderer.get_curve_points(candidate_controls);
            for (int contour_index = 0; contour_index < candidate_controls.size(); ++contour_index) {
                if ((candidate_controls[contour_index] - sraf_controls[contour_index]).cwiseAbs().maxCoeff() < 1e-12) continue;
                if (sraf_contour_self_intersects(candidate_curves[contour_index])) return "contour_self_intersection";
            }

            // 原几何方法保留灰度覆盖面积约束，梯度方法只用曲线自交作为几何拒绝条件。
            if (config.mode == "single_sraf_normal_gradient") return {};
            if (candidate_sraf_mask.sum() >= current_state.sraf_area_px) return "area_did_not_decrease";
            if ((candidate_sraf_mask - initial_sraf).cwiseMax(0.0).sum() > config.max_added_sraf_area_px) return "outward_area_exceeded";
            return {};
        };

        int accepted_iterations = 0;
        std::string stop_reason = "maximum_iterations";

        // 分块依据初始连续轮廓，块的全局编号和初始覆盖率在整次修复中固定。
        const auto regions = group_sraf_regions(sraf_controls, renderer);
        std::vector<bool> finished_regions(regions.size(), false);
        std::ofstream pair_history, stages, group_history, gradient_history, loss_history;
        int finite_difference_evaluations = 0;
        int active_region_index = -1, stage_number = 0, accepted_region_iterations = 0;
        cv::Mat active_window;
        double stage_initial_peak = 0.0;
        std::ofstream region_map(output_dir / "sraf_regions.csv");
        region_map << "sraf,contour\n";
        for (const auto& region : regions) {
            for (int contour : region.contour_indices) region_map << region.outer_contour << ',' << contour << '\n';
        }
        if (!region_map) throw std::runtime_error("Failed writing SRAF grouping");
        pair_history.open(output_dir / "pair_history.csv");
        stages.open(output_dir / "sraf_stages.csv");
        pair_history << "stage,iteration,sraf,first_contour,first_point,second_contour,second_point,width_px,"
                        "section_diameter_px,step_px,region_peak_before,region_peak_after\n" << std::setprecision(12);
        stages << "stage,sraf,accepted_iterations,initial_peak,final_peak,stop_reason\n" << std::setprecision(12);
        group_history.open(output_dir / "group_history.csv");
        group_history << "stage,iteration,sraf,group,first_contour,first_point,second_contour,second_point,width_px,"
                         "section_diameter_px,normal_width_px,requested_step_px,accepted_step_px,"
                         "region_peak_before,region_peak_after\n"
                      << std::setprecision(12);
        if (config.mode == "single_sraf_normal_gradient") {
            gradient_history.open(output_dir / "gradient_history.csv");
            loss_history.open(output_dir / "loss_history.csv");
            gradient_history << "stage,iteration,sraf,contour,point,normal_y,normal_x,fd_step_px,loss_plus,loss_minus,"
                                "gradient,requested_step_px,accepted_step_px,valid\n" << std::setprecision(12);
            loss_history << "stage,iteration,sraf,loss_before,loss_after,step_scale,moved_controls,accepted,"
                            "region_peak_before,region_peak_after,fd_evaluations,reason\n" << std::setprecision(12);
        }
        std::cout << "Repair mode: " << config.mode << ", SRAF regions: " << regions.size() << '\n';
        auto finish_active_sraf = [&](const std::string& reason) {
            // 先记完本块阶段结果，再将其标记为完成，下一轮才允许重新选择另一块。
            const double peak = peak_in_sraf_window(current_state.image.aerial_image, active_window);
            stages << stage_number << ',' << regions[active_region_index].outer_contour << ',' << accepted_region_iterations
                   << ',' << stage_initial_peak << ',' << peak << ',' << reason << '\n';
            stages.flush();
            std::cout << "Finish SRAF " << regions[active_region_index].outer_contour << ": " << reason << ", peak=" << peak << '\n';
            finished_regions[active_region_index] = true;
            active_region_index = -1;
        };
        for (int iteration = 1; iteration <= config.max_iterations; ++iteration) {
            // 外层锁定一块 AI 超阈值的 SRAF，直到本块 AI 峰值达到安全目标或触发停止条件。
            if (active_region_index >= 0 && peak_in_sraf_window(current_state.image.aerial_image, active_window) <= safe_threshold) {
                finish_active_sraf("region_safe_threshold_reached");
            }
            if (active_region_index < 0) {
                active_region_index = select_printing_sraf_from_aerial(regions, current_state.image.aerial_image, threshold,
                                                               current_state.background_window, finished_regions);
                if (active_region_index < 0) {
                    stop_reason = cv::countNonZero(current_state.printing) == 0 ? "printing_regions_repaired" : "unresolved_printing";
                    break;
                }
                ++stage_number;
                accepted_region_iterations = 0;
                active_window = make_sraf_evaluation_window(regions[active_region_index], current_state, guard, threshold);
                if (config.mode == "single_sraf_normal_gradient") {
                    // 导出本阶段固定的整图 0/1 损失窗口，便于独立复算损失；阶段编号与 CSV 一致，从 1 开始。
                    fs::create_directories(output_dir / "loss_windows");
                    Eigen::MatrixXd window(active_window.rows, active_window.cols);
                    for (int y = 0; y < active_window.rows; ++y) {
                        for (int x = 0; x < active_window.cols; ++x) window(y, x) = active_window.at<unsigned char>(y, x) != 0;
                    }
                    save_matrix(output_dir / "loss_windows" / ("stage_" + std::to_string(stage_number) + ".txt"), window);
                }
                stage_initial_peak = peak_in_sraf_window(current_state.image.aerial_image, active_window);
                std::cout << "Lock SRAF " << regions[active_region_index].outer_contour << ", initial peak=" << stage_initial_peak << '\n';
            }
            if (accepted_region_iterations >= config.max_iterations_per_sraf) {
                finish_active_sraf("per_sraf_iteration_limit");
                continue;
            }
            const SrafRegion& region = regions[active_region_index];
            if (config.mode == "single_sraf_normal_gradient") {
                // 1. 在当前控制点上一次性求法线，固定本块窗口；同轮所有扰动都从同一份坐标出发。
                const auto normal_points = sraf_control_normals(sraf_controls, renderer, region);
                const double current_loss = sraf_printing_loss(current_state.image.aerial_image, active_window, safe_threshold);
                const double current_peak = peak_in_sraf_window(current_state.image.aerial_image, active_window);
                cv::Mat local_hotspots;
                cv::bitwise_and(current_state.hotspots, active_window, local_hotspots);
                std::vector<cv::Point> hot_pixels;
                cv::findNonZero(local_hotspots, hot_pixels);
                std::vector<NormalGradientSample> samples; // 这个是每轮中控制点的一些属性包括梯度和请求步长，后续联合更新时使用。
                int round_evaluations = 0;
                // 差分时只检查被扰动曲线有效，不使用“面积必须减小”条件，否则向外扰动会始终被拒绝。
                auto perturbed_loss = [&](const SrafShrinkPoint& point, double displacement) {
                    ControlPoints perturbed = sraf_controls;
                    perturbed[point.contour].row(point.point) += displacement * point.inward_yx.transpose();
                    const auto curves = renderer.get_curve_points({perturbed[point.contour]});
                    if (sraf_contour_self_intersects(curves.front())) return std::numeric_limits<double>::quiet_NaN();
                    const auto mask = renderer.render_curve_even_odd(perturbed);
                    ++round_evaluations;
                    const auto image = imaging.compute(main_mask + mask, threshold, alpha);
                    return sraf_printing_loss(image.aerial_image, active_window, safe_threshold);
                };
                // 2. 每个控制点分别向内/向外扰动，中心差分求导；仅无效曲线时才缩小差分距离。
                double maximum_requested_step = 0.0;
                for (auto point : normal_points) {
                    const Eigen::Vector2d position = sraf_controls[point.contour].row(point.point).transpose();
                    point.hotspot_distance_px = std::numeric_limits<double>::infinity();
                    for (const auto& hot : hot_pixels) {
                        const double distance = (position - Eigen::Vector2d(hot.y, hot.x)).norm();
                        point.hotspot_distance_px = std::min(point.hotspot_distance_px, distance);
                    }
                    NormalGradientSample sample;
                    sample.point = point;
                    for (int backtrack = 0; backtrack <= config.max_backtracks; ++backtrack) {
                        sample.fd_step_px = config.fd_step_px * std::pow(0.5, backtrack);
                        sample.loss_plus = perturbed_loss(point, sample.fd_step_px);
                        sample.loss_minus = perturbed_loss(point, -sample.fd_step_px);
                        if (!std::isfinite(sample.loss_plus) || !std::isfinite(sample.loss_minus)) continue;
                        sample.gradient = (sample.loss_plus - sample.loss_minus) / (2.0 * sample.fd_step_px);
                        sample.valid = std::isfinite(sample.gradient);
                        break;
                    }
                    if (sample.valid) {
                        // 正位移朝内：只保留负梯度的内向分量，并按每轮最大位移限幅，不施加额外厚度限制。
                        const double descent = std::max(0.0, -config.learning_rate * sample.gradient);
                        sample.requested_step_px = std::min(descent, config.geometry.step_px);
                    }
                    maximum_requested_step = std::max(maximum_requested_step, sample.requested_step_px);
                    samples.push_back(sample);
                }
                finite_difference_evaluations += round_evaluations;
                if (iteration == 1) {
                    // 单独导出第一次求出的原始法向导数，不把学习率或回退系数混入梯度值。
                    std::ofstream first_gradient(output_dir / "first_gradient.csv");
                    first_gradient << "sraf,contour,point,normal_y,normal_x,fd_step_px,loss_plus,loss_minus,gradient,"
                                      "requested_step_px,valid\n" << std::setprecision(12);
                    std::cout << "First normal gradients (positive direction = inward):\n";
                    for (const auto& sample : samples) {
                        const auto& point = sample.point;
                        first_gradient << region.outer_contour << ',' << point.contour << ',' << point.point
                                       << ',' << point.inward_yx(0) << ',' << point.inward_yx(1) << ',' << sample.fd_step_px
                                       << ',' << sample.loss_plus << ',' << sample.loss_minus << ',' << sample.gradient
                                       << ',' << sample.requested_step_px << ',' << sample.valid << '\n';
                        std::cout << "  CP " << point.contour << '/' << point.point << ": gradient=" << sample.gradient
                                  << ", requested step=" << sample.requested_step_px << " pixel\n";
                    }
                    if (!first_gradient) throw std::runtime_error("Failed writing first gradients");
                    std::cout << std::flush;
                }
                // 3. 求完所有梯度后联合更新；只回退实际更新步长，不改变本轮已求出的法线和梯度。
                bool accepted = false;
                double accepted_scale = 0.0, next_loss = current_loss, next_peak = current_peak;
                int moved_count = 0;
                std::string reason = "no_feasible_inward_gradient";
                for (int backtrack = 0; maximum_requested_step > 0.0 && backtrack <= config.max_backtracks; ++backtrack) {
                    const double scale = std::pow(0.5, backtrack);
                    SrafShrinkProposal proposal;
                    proposal.control_points = sraf_controls;
                    for (const auto& sample : samples) {
                        if (sample.requested_step_px <= 0.0) continue;
                        auto point = sample.point;
                        point.movement_px = scale * sample.requested_step_px;
                        proposal.control_points[point.contour].row(point.point) += point.movement_px * point.inward_yx.transpose();
                        proposal.points.push_back(point);
                    }
                    proposal.moved_count = static_cast<int>(proposal.points.size());
                    const auto candidate_mask = renderer.render_curve_even_odd(proposal.control_points);
                    reason = geometry_rejection(proposal.control_points, candidate_mask);
                    RepairState candidate_state;
                    double candidate_loss = current_loss, candidate_peak = current_peak;
                    // 4. 曲线有效后完整成像，只要求固定窗口损失下降；EPE 和全局 printing 仅记录供比较。
                    if (reason.empty()) {
                        candidate_state = evaluate_candidate(candidate_mask);
                        candidate_peak = peak_in_sraf_window(candidate_state.image.aerial_image, active_window);
                        candidate_loss = sraf_printing_loss(candidate_state.image.aerial_image, active_window, safe_threshold);
                        if (candidate_loss >= current_loss) reason = "loss_did_not_decrease";
                    }
                    trials << iteration << ',' << backtrack << ',' << scale * maximum_requested_step << ',' << proposal.moved_count
                           << ',' << (reason.empty() ? 1 : 0) << ',' << (reason.empty() ? "accepted_gradient" : reason) << '\n';
                    trials.flush();
                    if (!reason.empty()) {
                        std::cout << "  Reject normal gradient scale " << scale << ": " << reason << '\n' << std::flush;
                        continue;
                    }
                    record_control_movements(iteration, proposal);
                    moved_count = proposal.moved_count;
                    sraf_controls = std::move(proposal.control_points);
                    current_state = std::move(candidate_state);
                    accepted_scale = scale;
                    next_loss = candidate_loss;
                    next_peak = candidate_peak;
                    ++accepted_iterations;
                    ++accepted_region_iterations;
                    record_iteration(iteration, scale * maximum_requested_step, moved_count, current_state);
                    std::cout << "  SRAF " << region.outer_contour << ": loss " << current_loss << " -> " << next_loss
                              << ", normal controls=" << samples.size() << ", FD images=" << round_evaluations << '\n' << std::flush;
                    accepted = true;
                    break;
                }
                for (const auto& sample : samples) {
                    const auto& point = sample.point;
                    gradient_history << stage_number << ',' << iteration << ',' << region.outer_contour << ',' << point.contour
                                     << ',' << point.point << ',' << point.inward_yx(0) << ',' << point.inward_yx(1)
                                     << ',' << sample.fd_step_px << ',' << sample.loss_plus << ',' << sample.loss_minus
                                     << ',' << sample.gradient << ',' << sample.requested_step_px
                                     << ',' << accepted_scale * sample.requested_step_px << ',' << sample.valid << '\n';
                }
                loss_history << stage_number << ',' << iteration << ',' << region.outer_contour << ',' << current_loss << ',' << next_loss
                             << ',' << accepted_scale << ',' << moved_count << ',' << accepted << ',' << current_peak << ',' << next_peak
                             << ',' << round_evaluations << ',' << (accepted ? "accepted" : reason) << '\n';
                gradient_history.flush();
                loss_history.flush();
                if (!accepted) finish_active_sraf(reason);
                continue;
            }
            // 每轮从当前实际控制点重新拟合和测宽，组内使用本轮曲线上的法线。
            auto pairs = find_sraf_pair_groups(sraf_controls, renderer, region, config.geometry, config.pair_groups);
            if (pairs.empty()) {
                finish_active_sraf("no_valid_wide_pairs");
                continue;
            }
            // 到热点的距离只写入移动记录，宽点对已由本块几何选出。
            cv::Mat local_hotspots;
            cv::bitwise_and(current_state.hotspots, active_window, local_hotspots);
            std::vector<cv::Point> hot_pixels;
            cv::findNonZero(local_hotspots, hot_pixels);
            for (auto& pair : pairs) {
                for (SrafShrinkPoint* point : {&pair.first, &pair.second}) {
                    const Eigen::Vector2d position = sraf_controls[point->contour].row(point->point).transpose();
                    point->hotspot_distance_px = std::numeric_limits<double>::infinity();
                    for (const auto& hot : hot_pixels) {
                        const double distance = (position - Eigen::Vector2d(hot.y, hot.x)).norm();
                        point->hotspot_distance_px = std::min(point->hotspot_distance_px, distance);
                    }
                }
            }
            const double current_peak = peak_in_sraf_window(current_state.image.aerial_image, active_window);
            bool accepted = false;
            std::string group_stop_reason = "group_step_rejected";
            // 回退编号取 [0,max_backtracks]，首轮不减半，后续每次将名义最大位移减半。
            for (int backtrack = 0; backtrack <= config.max_backtracks; ++backtrack) {
                const double nominal_step = config.geometry.step_px * std::pow(0.5, backtrack);
                std::vector<double> requested_group_steps;
                for (const auto& pair : pairs) {
                    // 两侧都可能内缩，每侧最多使用超出厚度目标部分的 45%，长度和步长单位均为 pixel。
                    const double width = std::min({pair.width_px, pair.first.width_px, pair.second.width_px});
                    requested_group_steps.push_back(std::min(nominal_step, 0.45 * (width - config.geometry.minimum_width_px)));
                }
                std::vector<double> candidate_group_steps = requested_group_steps;
                // 先尝试全部组选定的联合位移，全部候选均由同一份当前控制点构造。
                auto proposal = propose_sraf_group_shrink(sraf_controls, pairs, candidate_group_steps);
                Eigen::MatrixXd candidate_sraf_mask = renderer.render_curve_even_odd(proposal.control_points);
                std::string rejection_reason = geometry_rejection(proposal.control_points, candidate_sraf_mask);
                bool geometry_filtered = false;
                if (!rejection_reason.empty()) {
                    // 先检查完整联合位移。失败时以“整组”为单位加入，其余组保持已选位移，受限组独立减半或暂缓。
                    // 每次候选都从同一 controls 重建，避免把试探过程变成连续多次实际更新。
                    candidate_group_steps.assign(pairs.size(), 0.0);
                    candidate_sraf_mask = current_state.sraf_mask;
                    for (int group = 0; group < pairs.size(); ++group) {
                        for (int local_backtrack = 0; local_backtrack <= config.max_backtracks; ++local_backtrack) {
                            auto trial_group_steps = candidate_group_steps;
                            trial_group_steps[group] = requested_group_steps[group] * std::pow(0.5, local_backtrack);
                            const auto trial = propose_sraf_group_shrink(sraf_controls, pairs, trial_group_steps);
                            Eigen::MatrixXd trial_sraf_mask = renderer.render_curve_even_odd(trial.control_points);
                            if (!geometry_rejection(trial.control_points, trial_sraf_mask).empty()) continue;
                            candidate_group_steps = std::move(trial_group_steps);
                            candidate_sraf_mask = std::move(trial_sraf_mask);
                            break;
                        }
                    }
                    proposal = propose_sraf_group_shrink(sraf_controls, pairs, candidate_group_steps);
                    if (proposal.moved_count > 0) {
                        rejection_reason.clear();
                        geometry_filtered = true;
                    }
                }
                RepairState candidate_state;
                double candidate_peak = current_peak;
                // 几何通过后再完整成像，按当前块峰值和全图 EPE/printing 结果决定是否接受。
                if (rejection_reason.empty()) {
                    candidate_state = evaluate_candidate(candidate_sraf_mask);
                    candidate_peak = peak_in_sraf_window(candidate_state.image.aerial_image, active_window);
                    rejection_reason = imaging_rejection(candidate_state, current_state, initial_state, config,
                                                          current_peak, candidate_peak);
                }
                std::string trial_status = rejection_reason;
                if (rejection_reason.empty()) trial_status = geometry_filtered ? "accepted_groups_filtered" : "accepted_groups";
                trials << iteration << ',' << backtrack << ',' << nominal_step << ',' << proposal.moved_count
                       << ',' << (rejection_reason.empty() ? 1 : 0) << ',' << trial_status << '\n';
                trials.flush();
                if (!rejection_reason.empty()) {
                    group_stop_reason = rejection_reason;
                    std::cout << "  Reject joint step " << nominal_step << ": " << rejection_reason << '\n' << std::flush;
                    continue;
                }
                // 日志先读取更新前控制点，组记录写完后才一次性提交候选状态和全部位移。
                record_control_movements(iteration, proposal);
                int active_groups = 0;
                for (int group = 0; group < pairs.size(); ++group) {
                    const auto& pair = pairs[group];
                    group_history << stage_number << ',' << iteration << ',' << region.outer_contour << ',' << group
                                  << ',' << pair.first.contour << ',' << pair.first.point << ',' << pair.second.contour
                                  << ',' << pair.second.point << ',' << pair.width_px << ',' << pair.section_diameter_px
                                  << ',' << std::min(pair.first.width_px, pair.second.width_px) << ','
                                  << requested_group_steps[group] << ',' << candidate_group_steps[group] << ',' << current_peak
                                  << ',' << candidate_peak << '\n';
                    if (candidate_group_steps[group] == 0.0) continue;
                    ++active_groups;
                    pair_history << stage_number << ',' << iteration << ',' << region.outer_contour << ','
                                 << pair.first.contour << ',' << pair.first.point << ',' << pair.second.contour << ','
                                 << pair.second.point << ',' << pair.width_px << ',' << pair.section_diameter_px << ','
                                 << candidate_group_steps[group] << ',' << current_peak << ',' << candidate_peak << '\n';
                }
                group_history.flush();
                pair_history.flush();
                sraf_controls = std::move(proposal.control_points);
                current_state = std::move(candidate_state);
                ++accepted_iterations;
                ++accepted_region_iterations;
                record_iteration(iteration, nominal_step, proposal.moved_count, current_state);
                std::cout << "  SRAF " << region.outer_contour << ": joint groups " << active_groups << '/'
                          << pairs.size() << ", region peak=" << candidate_peak << '\n' << std::flush;
                accepted = true;
                break;
            }
            if (!accepted) finish_active_sraf(group_stop_reason);
        }
        if (active_region_index >= 0) {
            const bool region_safe = peak_in_sraf_window(current_state.image.aerial_image, active_window) <= safe_threshold;
            finish_active_sraf(region_safe ? "region_safe_threshold_reached" : "global_iteration_limit");
        }
        // 收尾时分别记录局部修复完成和全背景安全余量两个状态，供复跑与绘图读取。
        save_state(output_dir / "final", current_state, sraf_controls, renderer);
        YAML::Node summary;
        summary["source_dir"] = source_dir.string();
        summary["snapshot"] = snapshot;
        summary["threshold"] = threshold;
        summary["resist_alpha"] = alpha;
        summary["intensity_margin"] = config.intensity_margin;
        summary["printing_signal"] = "aerial_image";
        summary["printing_comparison"] = ">";
        summary["safe_threshold"] = safe_threshold;
        summary["target_guard_px"] = config.target_guard_px;
        summary["accepted_iterations"] = accepted_iterations;
        summary["stop_reason"] = stop_reason;
        summary["safe_threshold_reached"] = current_state.background_peak <= safe_threshold && current_state.detached_components == 0;
        summary["initial_background_wi_peak"] = initial_state.background_wi_peak;
        summary["final_background_wi_peak"] = current_state.background_wi_peak;
        summary["repair_mode"] = config.mode;
        if (config.mode == "single_sraf_normal_gradient") {
            summary["fd_step_px"] = config.fd_step_px;
            summary["learning_rate"] = config.learning_rate;
            summary["finite_difference_evaluations"] = finite_difference_evaluations;
            summary["printing_loss"] = "mean(max(AI-safe_threshold,0)^2) in the fixed SRAF window";
            summary["geometry_checks"] = "periodic CR closure and sampled self-intersection only";
            gradient_history.close();
            loss_history.close();
            if (!gradient_history || !loss_history) throw std::runtime_error("Failed writing gradient records");
        }
        summary["processed_sraf_count"] = stage_number;
        summary["printing_cleared"] = cv::countNonZero(current_state.printing) == 0;
        if (!pair_history || !stages) throw std::runtime_error("Failed writing pair/stage records");
        summary["maximum_pairs_per_iteration"] = config.pair_groups.maximum_pairs;
        if (!group_history) throw std::runtime_error("Failed writing group records");
        summary["snapshot_mask_max_error"] = render_error;
        summary["snapshot_aerial_max_error"] = aerial_error;
        summary["main_mask_changed"] = false;
        std::ofstream report(output_dir / "summary.yaml");
        report << summary << '\n';
        // Python 会立即读取 summary 和 CSV，先关闭所有记录文件，确保完整内容已写入磁盘。
        report.close();
        history.close();
        trials.close();
        movements.close();
        pair_history.close();
        stages.close();
        group_history.close();
        region_map.close();
        if (!report || !history || !trials || !movements || !pair_history || !stages || !group_history || !region_map) {
            throw std::runtime_error("Failed writing experiment records");
        }
        std::cout << "Stopped: " << stop_reason << "\nResults: " << output_dir << '\n' << std::flush;
        save_comparison_figures(project_root, output_dir);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "SRAF printing repair failed: " << error.what() << '\n';
        return 1;
    }
}
