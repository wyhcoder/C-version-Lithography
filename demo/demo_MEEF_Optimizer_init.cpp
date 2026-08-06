#include "MEEF_Optimizer.h"
#include "litho_prepare.h"
#include "lithography_simulator.h"
#include "simulation_parameters.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

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
        if (!fs::is_regular_file(file)) {
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
        "--title_prefix 'LSM vs Optimized'";

    std::cout << "\n打开可视化窗口：LSM mask / LSM wafer / optimized mask / optimized wafer\n";
    const int status = std::system(command.c_str());
    if (status != 0) {
        std::cerr << "[warning] 可视化脚本退出状态异常: " << status << '\n';
    }
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
    const fs::path lsm_path = (argc > 2)
        ? fs::path(argv[2])
        : project_root / "assets/lsm_mask/ls_image工字型.txt";
    const fs::path save_path = (argc > 3)
        ? fs::path(argv[3])
        : project_root / "result/MEEF_result/test";

    try {
        if (!fs::is_regular_file(config_path)) {
            throw std::runtime_error("配置文件不存在: " + config_path.string());
        }
        if (!fs::is_regular_file(lsm_path)) {
            throw std::runtime_error("LSM mask 不存在: " + lsm_path.string());
        }

        std::cout << "========== MEEF_Optimizer 初始化测试 ==========\n"
                  << "config    : " << config_path << '\n'
                  << "target    : assets/target_pattern/工字型.bmp\n"
                  << "lsm mask  : " << lsm_path << '\n'
                  << "save path : " << save_path << "\n\n";

        auto t0 = std::chrono::steady_clock::now();
        

        SimulationParameters params =
            SimulationParameters::from_yaml(config_path.string());
        params.mask.image_name = "工字型";

        LithographySimulator simulator(params);
        auto t1 = std::chrono::steady_clock::now();

        LithoPrepare prepare(
            simulator._grid, simulator._pupil, simulator._source, true);
        ImagingCache cache = prepare.cache();
        auto t2 = std::chrono::steady_clock::now();

        MEEFPipelineConfig meef_config{};
        meef_config.pattern_name = "工字型";
        meef_config.ls_mask_path = lsm_path.string();
        meef_config.save_file_path = save_path.string();
        meef_config.move_strategy = "xy";
        meef_config.iter = (argc > 5) ? std::stoi(argv[5]) : 100;
        meef_config.step_tol = 0.0;
        meef_config.patience = 3;

        meef_config.main_cp_interval = 7;
        meef_config.main_symmetry = "none";

        // 当前 C++ 配置字段名为 sraf_cp_interval；这里取用户配置的 5 像素间隔。
        meef_config.sraf_cp_interval = 5;
        meef_config.sraf_min_cps = 8;
        meef_config.sraf_min_aera = 50;

        meef_config.msaa_level = 16;
        meef_config.curve_type = "BS";
        meef_config.delta = 0.15;
        meef_config.dilate_radius = 2;

        meef_config.interval_line = 5;
        meef_config.interval_corner = 2;
        meef_config.mid_weight = 4.0;
        meef_config.other_weight = 1.0;
        meef_config.optimize_wepe_only =
            (argc > 6 && std::string(argv[6]) == "--wepe-only");

        std::cout << "MEEF EP mode: "
                  << (meef_config.optimize_wepe_only
                          ? "WEPE key points only"
                          : "all EP points")
                  << '\n';

        MEEF_Optimizer optimizer(simulator, cache, meef_config);
        auto t3 = std::chrono::steady_clock::now();

        // 第 4 个可选参数传 --build-meef 时，实际构建并保存 Mx/My。
        // 默认只测试初始化，避免每次运行都执行 4*num_cps 次光刻仿真。
        const std::string mode = (argc > 4) ? argv[4] : "--init-only";
        if (mode == "--build-meef") {
            auto meef_start = std::chrono::steady_clock::now();
            MEEFMatrixXY meef = optimizer.build_meef_matrix_xy();
            auto meef_end = std::chrono::steady_clock::now();

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
            std::cout << "\n--- 完整 MEEF 优化测试 ---\n"
                      << "iterations: " << meef_config.iter << '\n';
            optimizer.optimize();
            show_optimization_comparison(project_root, save_path);
        } else if (mode != "--init-only") {
            throw std::invalid_argument(
                "未知运行模式: " + mode +
                "（支持 --init-only / --build-meef / --optimize）");
        }

        const auto simulator_ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        const auto prepare_ms =
            std::chrono::duration<double, std::milli>(t2 - t1).count();
        const auto optimizer_ms =
            std::chrono::duration<double, std::milli>(t3 - t2).count();

        std::cout << "\n--- 初始化耗时 ---\n"
                  << "simulator      : " << simulator_ms << " ms\n"
                  << "litho prepare  : " << prepare_ms << " ms\n"
                  << "MEEF optimizer : " << optimizer_ms << " ms\n";

        const char* expected_files[] = {
            "main_cps.txt",
            "sraf_cps.txt",
            "eps.txt",
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
        std::cerr << "\nMEEF_Optimizer 初始化失败: " << e.what() << '\n';
        return 1;
    }
}
