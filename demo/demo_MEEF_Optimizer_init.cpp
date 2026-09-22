#include "MEEF_Optimizer.h"
#include "litho_prepare.h"
#include "lithography_simulator.h"
#include "simulation_parameters.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <streambuf>
#include <string>
#include <yaml-cpp/yaml.h>

using namespace litho;

// POSIX shell 的单引号转义，确保来自命令行的路径不会改变可视化命令结构。
static std::string shell_quote(const std::filesystem::path& path) {
    const std::string text = path.string();
    std::string quoted = "'";
    for (char ch : text) {
        if (ch == '\'') quoted += "'\\\"'\\\"'";
        else quoted += ch;
    }
    return quoted + "'";
}

template <typename T>
static T yaml_required(const YAML::Node& node, const char* key) {
    if (!node || !node[key]) {
        throw std::runtime_error("config.yaml 缺少 meef." + std::string(key));
    }
    return node[key].as<T>();
}

template <typename T>
static T yaml_optional(
    const YAML::Node& node, const char* key, const T& default_value)
{
    return (node && node[key]) ? node[key].as<T>() : default_value;
}

static std::filesystem::path config_relative_path(
    const std::filesystem::path& project_root,
    const std::string& path)
{
    const std::filesystem::path result(path);
    return result.is_absolute() ? result : project_root / result;
}

// 同一份输出同时写入终端与日志文件。
class TeeBuffer final : public std::streambuf {
public:
    TeeBuffer(std::streambuf* terminal, std::streambuf* log)
        : terminal_(terminal), log_(log) {}

protected:
    int_type overflow(int_type ch) override {
        if (traits_type::eq_int_type(ch, traits_type::eof())) {
            return traits_type::not_eof(ch);
        }
        const char c = traits_type::to_char_type(ch);
        if (traits_type::eq_int_type(terminal_->sputc(c), traits_type::eof()) ||
            traits_type::eq_int_type(log_->sputc(c), traits_type::eof())) {
            return traits_type::eof();
        }
        return ch;
    }

    int sync() override {
        return terminal_->pubsync() == 0 && log_->pubsync() == 0 ? 0 : -1;
    }

private:
    std::streambuf* terminal_;
    std::streambuf* log_;
};

class ScopedStreamTee final {
public:
    ScopedStreamTee(std::ostream& stream, std::streambuf* log)
        : stream_(stream), original_(stream.rdbuf()), tee_(original_, log) {
        stream_.rdbuf(&tee_);
    }

    ~ScopedStreamTee() { stream_.rdbuf(original_); }

    ScopedStreamTee(const ScopedStreamTee&) = delete;
    ScopedStreamTee& operator=(const ScopedStreamTee&) = delete;

private:
    std::ostream& stream_;
    std::streambuf* original_;
    TeeBuffer tee_;
};

static void show_optimization_comparison(
    const std::filesystem::path& project_root,
    const std::filesystem::path& save_path)
{
    namespace fs = std::filesystem;
    const fs::path lsm_mask = save_path / "lsm_mask.txt";
    const fs::path lsm_wafer = save_path / "lsm_wafer.txt";
    const fs::path optimized_mask = save_path / "iterations/mask.txt";
    const fs::path optimized_wafer = save_path / "iterations/wafer.txt";

    for (const auto& file : {lsm_mask, lsm_wafer, optimized_mask, optimized_wafer}) {
        if (!fs::is_regular_file(file)  ) {
            throw std::runtime_error("缺少可视化结果文件: " + file.string());
        }
    }

    // show_multi.py 会在同一个 matplotlib 窗口中显示四张图；关闭窗口后 demo 继续退出。
    const fs::path script = project_root / "scripts/show_multi.py";
    // 优先使用项目虚拟环境，避免系统 python3 缺少可视化依赖。
    const fs::path venv_python = project_root / ".venv/bin/python";
    const std::string python = fs::is_regular_file(venv_python)
        ? shell_quote(venv_python)
        : "python3";
    const std::string command =
        python + " " + shell_quote(script) + " " +
        shell_quote(lsm_mask) + " gray " +
        shell_quote(lsm_wafer) + " gray " +
        shell_quote(optimized_mask) + " gray " +
        shell_quote(optimized_wafer) + " gray " +
        "--titles 'LSM Baseline Mask|LSM Baseline Wafer|"
        "MEEF Optimized Mask|MEEF Optimized Wafer'";

    std::cout << "\n打开可视化窗口：\n"
              << "  1. LSM Baseline Mask\n"
              << "  2. LSM Baseline Wafer\n"
              << "  3. MEEF Optimized Mask\n"
              << "  4. MEEF Optimized Wafer\n";
    const int status = std::system(command.c_str());
    if (status != 0) {
        std::cerr << "[warning] 可视化脚本退出状态异常: " << status << '\n';
    }
}

static void save_meef_figures(const std::filesystem::path& project_root, const std::filesystem::path& save_path) {
    namespace fs = std::filesystem;
    const fs::path script = project_root / "scripts/plot_meef_parametric_curves.py";
    if (!fs::is_regular_file(script)) {
        std::cerr << "[warning] MEEF 绘图脚本不存在: " << script << '\n';
        return;
    }
    const fs::path venv_python = project_root / ".venv/bin/python";
    const std::string python = fs::is_regular_file(venv_python) ? shell_quote(venv_python) : "python3";
    const fs::path matplotlib_cache = save_path / ".matplotlib";
    fs::create_directories(matplotlib_cache);
    const std::string command = "MPLBACKEND=Agg MPLCONFIGDIR=" + shell_quote(matplotlib_cache) +
        " XDG_CACHE_HOME=" + shell_quote(matplotlib_cache) + " " + python + " " + shell_quote(script) +
        " --result-dir " + shell_quote(save_path) + " --best both";
    std::cout << "\n保存 MEEF 参数化曲线和 EP 选点图...\n" << std::flush;
    if (std::system(command.c_str()) != 0) std::cerr << "[warning] MEEF 绘图失败，请检查上述 Python 输出。\n";
}

int main(int argc, char** argv) {
    namespace fs = std::filesystem;

    // demo 按约定从 build-* 目录运行：config.yaml 和 target_pattern/ 在当前目录，
    // 项目根目录则是其父目录。避免依赖某台开发机的绝对路径。
    const fs::path build_dir = fs::current_path();
    const fs::path project_root = build_dir.parent_path();
    const fs::path config_path = (argc > 1)
        ? fs::path(argv[1])
        : build_dir / "config.yaml";
    fs::path console_log_path;
    try {
        if (argc > 2) {
            throw std::invalid_argument(
                "用法: ./demo_MEEF_Optimizer_init [config.yaml]；所有运行参数请在 YAML 的 meef 节设置");
        }
        if (!fs::is_regular_file(config_path)) {
            throw std::runtime_error("配置文件不存在: " + config_path.string());
        }

        const YAML::Node root = YAML::LoadFile(config_path.string());
        const YAML::Node meef_yaml = root["meef"];
        if (!meef_yaml || !meef_yaml.IsMap()) {
            throw std::runtime_error("config.yaml 缺少 meef 配置节");
        }
        const fs::path lsm_path = config_relative_path(
            project_root, yaml_required<std::string>(meef_yaml, "lsm_mask_path"));
        const fs::path save_path = config_relative_path(
            project_root, yaml_required<std::string>(meef_yaml, "output_dir"));
        const std::string config_snapshot_name = yaml_required<std::string>(
            meef_yaml, "config_snapshot_name");
        const std::string console_log_name = yaml_required<std::string>(
            meef_yaml, "console_log_name");
        const std::string mode = yaml_required<std::string>(meef_yaml, "run_mode");
        const std::string pattern_name = yaml_required<std::string>(
            meef_yaml, "pattern_name");
        const std::string main_cp_mode = yaml_optional<std::string>(
            meef_yaml, "main_cp_mode", "target_interval");
        const std::string main_cps_config_path = yaml_optional<std::string>(
            meef_yaml, "main_cps_npy_path", "");
        const std::string sraf_mask_mode = yaml_optional<std::string>(
            meef_yaml, "sraf_mask_mode", "lsm");
        const std::string fitted_sraf_config_path = yaml_optional<std::string>(
            meef_yaml, "fitted_sraf_txt_path", "");
        const fs::path target_path =
            build_dir / "target_pattern" / (pattern_name + ".bmp");
        if (!fs::is_regular_file(lsm_path)) {
            throw std::runtime_error("LSM mask 不存在: " + lsm_path.string());
        }
        if (!fs::is_regular_file(target_path)) {
            throw std::runtime_error("YAML 指定的 target 不存在: " + target_path.string());
        }
        if (main_cp_mode != "target_interval" && main_cp_mode != "lsm_interval" &&
            main_cp_mode != "LSM_interval" && main_cp_mode != "npy") {
            throw std::invalid_argument("meef.main_cp_mode 仅支持 target_interval / lsm_interval / npy");
        }
        fs::path main_cps_npy_path;
        if (main_cp_mode == "npy") {
            if (main_cps_config_path.empty()) {
                throw std::invalid_argument(
                    "main_cp_mode=npy 时必须设置 meef.main_cps_npy_path");
            }
            main_cps_npy_path = config_relative_path(
                project_root, main_cps_config_path);
            if (!fs::is_regular_file(main_cps_npy_path)) {
                throw std::runtime_error(
                    "主图形控制点 NPY 不存在: " + main_cps_npy_path.string());
            }
        }
        if (sraf_mask_mode != "lsm" && sraf_mask_mode != "fitted_txt") {
            throw std::invalid_argument(
                "meef.sraf_mask_mode 仅支持 lsm / fitted_txt");
        }
        fs::path fitted_sraf_path;
        if (sraf_mask_mode == "fitted_txt") {
            if (fitted_sraf_config_path.empty()) {
                throw std::invalid_argument(
                    "sraf_mask_mode=fitted_txt 时必须设置 meef.fitted_sraf_txt_path");
            }
            fitted_sraf_path = config_relative_path(
                project_root, fitted_sraf_config_path);
            if (!fs::is_regular_file(fitted_sraf_path)) {
                throw std::runtime_error(
                    "拟合 SRAF TXT 不存在: " + fitted_sraf_path.string());
            }
        }
        if (config_snapshot_name.empty() || fs::path(config_snapshot_name).has_parent_path()) {
            throw std::invalid_argument(
                "meef.config_snapshot_name 必须是不含目录的文件名");
        }
        if (console_log_name.empty() || fs::path(console_log_name).has_parent_path()) {
            throw std::invalid_argument(
                "meef.console_log_name 必须是不含目录的文件名");
        }
        fs::create_directories(save_path);
        const fs::path config_snapshot = save_path / config_snapshot_name;
        fs::copy_file(config_path, config_snapshot, fs::copy_options::overwrite_existing);
        console_log_path = save_path / console_log_name;
        std::ofstream console_log(console_log_path, std::ios::out | std::ios::trunc);
        if (!console_log) {
            throw std::runtime_error("无法写入终端日志: " + console_log_path.string());
        }
        ScopedStreamTee stdout_tee(std::cout, console_log.rdbuf());
        ScopedStreamTee stderr_tee(std::cerr, console_log.rdbuf());

        std::cout << "========== MEEF_Optimizer 初始化测试 ==========\n"
                  << "config    : " << config_path << '\n'
                  << "target    : " << target_path << '\n'
                  << "lsm mask  : " << lsm_path << '\n'
                  << "main CP   : " << main_cp_mode;
        if (!main_cps_npy_path.empty()) std::cout << " (" << main_cps_npy_path << ")";
        std::cout << '\n'
                  << "SRAF mask : " << sraf_mask_mode;
        if (!fitted_sraf_path.empty()) std::cout << " (" << fitted_sraf_path << ")";
        std::cout << '\n'
                  << "save path : " << save_path << '\n'
                  << "config copy: " << config_snapshot << '\n'
                  << "console log: " << console_log_path << "\n\n";

        auto t0 = std::chrono::steady_clock::now();
        

        SimulationParameters params =
            SimulationParameters::from_yaml(config_path.string());
        // meef.pattern_name 是本 demo 唯一的 target 名称来源。
        params.mask.image_name = pattern_name;

        LithographySimulator simulator(params);
        auto t1 = std::chrono::steady_clock::now();

        LithoPrepare prepare(
            simulator._grid, simulator._pupil, simulator._source, true);
        ImagingCache cache = prepare.cache();
        auto t2 = std::chrono::steady_clock::now();

        MEEFPipelineConfig meef_config{};
        meef_config.pattern_name = pattern_name;
        meef_config.ls_mask_path = lsm_path.string();
        meef_config.save_file_path = save_path.string();
        meef_config.meef_builder = yaml_optional<std::string>(
            meef_yaml, "meef_builder", "finite_difference");
        meef_config.meef_matrix_update_mode = yaml_optional<std::string>(meef_yaml, "meef_matrix_update_mode", "every_iteration");
        const bool valid_update_mode = meef_config.meef_matrix_update_mode == "every_iteration" ||
                                       meef_config.meef_matrix_update_mode == "periodic" ||
                                       meef_config.meef_matrix_update_mode == "initial_only";
        if (!valid_update_mode) {
            throw std::invalid_argument("meef.meef_matrix_update_mode 仅支持 every_iteration / periodic / initial_only");
        }
        meef_config.meef_rebuild_interval = yaml_optional<int>(meef_yaml, "meef_rebuild_interval", 1);
        if (meef_config.meef_rebuild_interval <= 0) throw std::invalid_argument("meef.meef_rebuild_interval 必须为正整数");
        meef_config.epe_histogram_bin_width_nm = yaml_optional<double>(
            meef_yaml, "epe_histogram_bin_width_nm", 0.25);
        if (!std::isfinite(meef_config.epe_histogram_bin_width_nm) ||
            meef_config.epe_histogram_bin_width_nm <= 0.0) {
            throw std::invalid_argument(
                "meef.epe_histogram_bin_width_nm 必须是正数");
        }
        meef_config.move_strategy = yaml_required<std::string>(meef_yaml, "move_strategy");
        meef_config.iter = yaml_required<int>(meef_yaml, "iter");
        meef_config.stop_mode = yaml_optional<std::string>(meef_yaml, "stop_mode", "fixed_iterations");
        if (meef_config.stop_mode != "fixed_iterations" && meef_config.stop_mode != "small_step") {
            throw std::invalid_argument("meef.stop_mode 仅支持 fixed_iterations / small_step");
        }
        meef_config.step_tol = yaml_optional<double>(meef_yaml, "step_tol", 0.02);
        meef_config.patience = yaml_required<int>(meef_yaml, "patience");
        if (meef_config.iter <= 0) throw std::invalid_argument("meef.iter 必须为正整数");
        if (meef_config.patience <= 0) throw std::invalid_argument("meef.patience 必须为正整数");
        if (meef_config.stop_mode == "small_step" && (!std::isfinite(meef_config.step_tol) || meef_config.step_tol <= 0.0)) {
            throw std::invalid_argument("small_step 模式下 meef.step_tol 必须为有限正数 (pixel)");
        }

        meef_config.main_cp_mode = (main_cp_mode == "npy") ? "file" : main_cp_mode;
        meef_config.main_cp_interval = yaml_required<int>(meef_yaml, "main_cp_interval");
        if (meef_config.main_cp_interval < 0) throw std::invalid_argument("meef.main_cp_interval 不能为负数");
        meef_config.main_symmetry = yaml_required<std::string>(meef_yaml, "main_symmetry");

        if (main_cp_mode == "npy") {
            const fs::path converter =
                project_root / "scripts/convert_npy_control_points.py";
            if (!fs::is_regular_file(converter)) {
                throw std::runtime_error(
                    "NPY 控制点转换脚本不存在: " + converter.string());
            }
            const fs::path venv_python = project_root / ".venv/bin/python";
            const std::string python = fs::is_regular_file(venv_python)
                ? shell_quote(venv_python)
                : "python3";
            const fs::path converted_cps = save_path / "imported_main_cps.txt";
            const std::string convert_command =
                python + " " + shell_quote(converter) +
                " --input " + shell_quote(main_cps_npy_path) +
                " --output " + shell_quote(converted_cps);
            std::cout << "转换 NPY 控制点: " << main_cps_npy_path << '\n';
            const int convert_status = std::system(convert_command.c_str());
            if (convert_status != 0 || !fs::is_regular_file(converted_cps)) {
                throw std::runtime_error(
                    "NPY 控制点转换失败，退出状态: " +
                    std::to_string(convert_status));
            }
            meef_config.main_cps_path = converted_cps.string();
        }

        meef_config.sraf_mask_mode = sraf_mask_mode;
        if (!fitted_sraf_path.empty()) {
            meef_config.fitted_sraf_txt_path = fitted_sraf_path.string();
        }

        meef_config.sraf_cp_interval = yaml_required<int>(meef_yaml, "sraf_cp_interval");
        meef_config.sraf_min_cps = yaml_required<int>(meef_yaml, "sraf_min_cps");
        meef_config.sraf_min_aera = yaml_required<int>(meef_yaml, "sraf_min_aera");

        meef_config.msaa_level = yaml_required<int>(meef_yaml, "msaa_level");
        meef_config.rasterizer = yaml_optional<std::string>(
            meef_yaml, "rasterizer", "msaa");
        meef_config.curve_type = yaml_required<std::string>(meef_yaml, "curve_type");
        meef_config.sraf_curve_type = yaml_optional<std::string>(meef_yaml, "sraf_curve_type", meef_config.curve_type);
        if (meef_config.sraf_curve_type != "BS" && meef_config.sraf_curve_type != "OA" && meef_config.sraf_curve_type != "CR") {
            throw std::invalid_argument("meef.sraf_curve_type 仅支持 BS / OA / CR");
        }
        meef_config.delta = yaml_required<double>(meef_yaml, "delta");
        meef_config.dilate_radius = yaml_required<int>(meef_yaml, "dilate_radius");

        meef_config.interval_line = yaml_required<int>(meef_yaml, "interval_line");
        meef_config.interval_corner = yaml_required<int>(meef_yaml, "interval_corner");
        meef_config.mid_weight = yaml_required<double>(meef_yaml, "mid_weight");
        meef_config.other_weight = yaml_required<double>(meef_yaml, "other_weight");
        meef_config.wepe_all_eps = yaml_optional<bool>(meef_yaml, "wepe_all_eps", false);
        meef_config.optimize_wepe_only = yaml_required<bool>(meef_yaml, "optimize_wepe_only");

        std::cout << "MEEF EP mode: "
                  << (meef_config.optimize_wepe_only
                          ? "WEPE-included EP points only"
                          : "all EP points")
                  << '\n'
                  << "WEPE weights: " << (meef_config.wepe_all_eps ? "all selected EP = 1" : "key EP = 1, other EP = 0") << '\n'
                  << "Curve rasterizer: " << meef_config.rasterizer << '\n'
                  << "SRAF curve type: " << meef_config.sraf_curve_type << '\n'
                  << "MEEF builder: " << meef_config.meef_builder << '\n'
                  << "MEEF matrix update: " << meef_config.meef_matrix_update_mode << '\n'
                  << "MEEF rebuild interval: " << meef_config.meef_rebuild_interval << '\n'
                  << "MEEF stop mode: " << meef_config.stop_mode << '\n'
                  << "MEEF step tolerance: " << meef_config.step_tol << " pixel\n"
                  << "EPE histogram bin width: "
                  << meef_config.epe_histogram_bin_width_nm << " nm\n";

        MEEF_Optimizer optimizer(simulator, cache, meef_config);
        auto t3 = std::chrono::steady_clock::now();

        std::cout << "\n--- MEEF 初始化规模 ---\n"
                  << "主图形控制点个数 : "
                  << optimizer.main_control_point_count() << '\n'
                  << "SRAF 控制点个数   : "
                  << optimizer.sraf_control_point_count() << '\n'
                  << "Mx/My 矩阵维度     : "
                  << optimizer.evaluation_point_count() << " x "
                  << optimizer.main_control_point_count()
                  << "（行=EP 点，列=主图形控制点）\n";

        const auto task_start = t3;
        auto task_end = task_start;
        std::string task_label = "init-only";
        bool open_comparison = false;

        if (mode == "--build-meef") {
            task_label = "build MEEF matrix";
            auto meef_start = std::chrono::steady_clock::now();
            MEEFMatrixXY meef = optimizer.build_meef_matrix_xy_selected();
            auto meef_end = std::chrono::steady_clock::now();
            task_end = meef_end;

            auto save_matrix = [&](const fs::path& path,
                                   const Eigen::MatrixXd& matrix) {
                std::ofstream f(path);
                if (!f) {
                    throw std::runtime_error("无法写入 MEEF 矩阵: " + path.string());
                }
                f << std::fixed << std::setprecision(6);
                for (int r = 0; r < matrix.rows(); ++r) {
                    for (int c = 0; c < matrix.cols(); ++c) {
                        f << matrix(r, c);
                        if (c + 1 < matrix.cols()) f << ' ';
                    }
                    f << '\n';
                }
            };

            save_matrix(save_path / "meef_mx.txt", meef.mx);
            save_matrix(save_path / "meef_my.txt", meef.my);
            std::cout << "\n--- X/Y MEEF 矩阵 ---\n"
                      << "Mx: " << meef.mx.rows() << " x " << meef.mx.cols()
                      << ", norm=" << meef.mx.norm() << '\n'
                      << "My: " << meef.my.rows() << " x " << meef.my.cols()
                      << ", norm=" << meef.my.norm() << '\n'
                      << "time: "
                      << std::chrono::duration<double>(meef_end - meef_start).count()
                      << " s\n";
        } else if (mode == "--optimize") {
            task_label = "optimization";
            std::cout << "\n--- 完整 MEEF 优化测试 ---\n"
                      << "iterations: " << meef_config.iter << '\n';
            optimizer.optimize();
            task_end = std::chrono::steady_clock::now();
            save_meef_figures(project_root, save_path);
            open_comparison = true;
        } else if (mode != "--init-only") {
            throw std::invalid_argument(
                "未知运行模式: " + mode +
                "（支持 --init-only / --build-meef / --optimize）");
        }

        const auto simulator_s = std::chrono::duration<double>(t1 - t0).count();
        const auto prepare_s = std::chrono::duration<double>(t2 - t1).count();
        const auto optimizer_init_s = std::chrono::duration<double>(t3 - t2).count();
        const auto task_s = std::chrono::duration<double>(task_end - task_start).count();
        const auto total_compute_s = std::chrono::duration<double>(task_end - t0).count();

        std::cout << "\n--- 整体计算耗时（不含可视化窗口等待）---\n"
                  << "simulator init : " << simulator_s << " s\n"
                  << "litho prepare  : " << prepare_s << " s\n"
                  << "optimizer init : " << optimizer_init_s << " s\n"
                  << task_label << " : " << task_s << " s\n"
                  << "total compute  : " << total_compute_s << " s\n";

        if (open_comparison) {
            show_optimization_comparison(project_root, save_path);
        }

        const char* expected_files[] = {
            "main_cps.txt",
            "sraf_cps.txt",
            "eps.txt",
            "eps_weights.txt",
            "sraf_mask.txt",
            "main_mask.txt",
        };

        bool output_ok = true;
        std::cout << "\n--- 初始化输出 ---\n";
        for (const char* name : expected_files) {
            fs::path file = save_path / name;
            bool ok = fs::is_regular_file(file) && fs::file_size(file) > 0;
            std::cout << (ok ? "[OK]   " : "[FAIL] ") << file << '\n';
            output_ok = output_ok && ok;
        }

        if (!output_ok) {
            std::cerr << "\nMEEF_Optimizer 已构造，但初始化输出不完整。\n";
            return 2;
        }

        std::cout << "\nMEEF_Optimizer 测试完成。\n";
        return 0;
    } catch (const std::exception& e) {
        const std::string message =
            "\nMEEF_Optimizer 运行失败: " + std::string(e.what()) + "\n";
        std::cerr << message;
        if (!console_log_path.empty()) {
            std::ofstream error_log(console_log_path, std::ios::out | std::ios::app);
            if (error_log) error_log << message;
        }
        return 1;
    }
}
