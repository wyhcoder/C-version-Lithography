#include "imbinarize.h"
#include "save_txt.h"
#include "litho_prepare.h"
#include "imaging.h"
#include "CTM_Optimizer.h"
#include "simulation_parameters.h"
#include "lithography_simulator.h"
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>
#include <cmath>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>



using namespace litho;

// 使用单引号保护传给 POSIX shell 的路径，避免空格等字符改变命令结构。
static std::string shell_quote(const std::filesystem::path& path) {
    const std::string text = path.string();
    std::string quoted = "'";
    for (char ch : text) {
        if (ch == '\'') quoted += "'\"'\"'";
        else quoted += ch;
    }
    return quoted + "'";
}

int main(int argc, char** argv){
    // std::cout << "========== CTM优化demo ==========\n" << std::endl;

    // 允许从命令行传入 yaml 路径；默认相对于项目根（build 目录下用 "../config.yaml"）
    std::string cfg_path = (argc > 1) ? argv[1] : "config.yaml";
    std::cout << "Loading config: " << cfg_path << std::endl;

    SimulationParameters params;
    std::string optimizer_method = "gradient_descent";
    int max_iterations = 100;
    double learning_rate = 0.9;
    int lbfgs_history_size = 10;
    double gradient_tolerance = 0.0;
    try {
        params = SimulationParameters::from_yaml(cfg_path);
        const YAML::Node ctm = YAML::LoadFile(cfg_path)["ctm"];
        if (ctm && !ctm.IsMap()) throw std::invalid_argument("ctm must be a YAML mapping");
        if (ctm && ctm["optimizer"]) optimizer_method = ctm["optimizer"].as<std::string>();
        if (ctm && ctm["max_iteration"]) max_iterations = ctm["max_iteration"].as<int>();
        if (ctm && ctm["learning_rate"]) learning_rate = ctm["learning_rate"].as<double>();
        if (ctm && ctm["lbfgs_history_size"]) lbfgs_history_size = ctm["lbfgs_history_size"].as<int>();
        if (ctm && ctm["gradient_tolerance"]) gradient_tolerance = ctm["gradient_tolerance"].as<double>();
        if (optimizer_method != "gradient_descent" && optimizer_method != "lbfgs") {
            throw std::invalid_argument("ctm.optimizer must be gradient_descent or lbfgs");
        }
        if (max_iterations <= 0) throw std::invalid_argument("ctm.max_iteration must be positive");
        if (!std::isfinite(learning_rate) || learning_rate <= 0.0) throw std::invalid_argument("ctm.learning_rate must be positive");
        if (lbfgs_history_size <= 0) throw std::invalid_argument("ctm.lbfgs_history_size must be positive");
        if (!std::isfinite(gradient_tolerance) || gradient_tolerance < 0.0) {
            throw std::invalid_argument("ctm.gradient_tolerance must be non-negative");
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to load config: " << e.what() << "\n"
                  << "Hint: run from project root, or pass path explicitly:\n"
                  << "      ./demo_CTM /absolute/path/to/config.yaml" << std::endl;
        return 1;
    }
    std::cout << "CTM optimizer: " << optimizer_method << ", max iterations: " << max_iterations
              << ", gradient tolerance: " << gradient_tolerance << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    LithographySimulator simulator(params);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Simulator time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() * 0.001 << " s" << std::endl;
    
    LithoPrepare prepare(simulator._grid, simulator._pupil, simulator._source, true);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Prepare time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() * 0.001<< " s" << std::endl;
    
    // 获取提前计算的缓存
    ImagingCache cache = prepare.cache();
    // 保存目标掩膜、 pupil、 source
    SaveTxt::save_mat_int(simulator._mask.data(), "System_result/target_mask.txt");
    SaveTxt::save_mat(simulator._pupil.get_amplitude(), "System_result/pupil.txt");
    SaveTxt::save_mat(simulator._source.get_source_map(), "System_result/source.txt");
    // CTM优化
    auto t_begin = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd result;
    try {
        CTM_Optimizer optimizer(simulator, cache, max_iterations, learning_rate, optimizer_method, lbfgs_history_size, gradient_tolerance);
        result = optimizer.optimize();
    } catch (const std::exception& e) {
        std::cerr << "CTM optimization failed: " << e.what() << std::endl;
        return 1;
    }
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Optimize time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_begin).count() * 0.001 << " s" << std::endl;
    // 成像
    Imaging imaging(cache);
    Imaging_Result imaging_result = imaging.compute(result, simulator._params.resist.threshold, simulator._params.resist.alpha);
    SaveTxt::save_mat(result, "CTM_result/gray_mask.txt");
    SaveTxt::save_mat(imaging_result.aerial_image, "CTM_result/aerial_image_G.txt");
    SaveTxt::save_mat(imaging_result.wafer_image, "CTM_result/wafer_image_G.txt");
   

    // 二值化
    // Eigen::MatrixXd binary_mask = Imbinarize::binarize(result);
    // SaveTxt::save_mat_int(binary_mask, "binary_mask.txt");
    // Eigen::MatrixXd binarry_mask_OTSU = Imbinarize::binarize_OTSU(result);
    // SaveTxt::save_mat_int(binarry_mask_OTSU, "binarry_mask_OTSU.txt");
    BinarizeResult binary_mask = Imbinarize::binarize_advanced(result, simulator._mask.data());
    SaveTxt::save_mat_int(binary_mask.final_mask, "CTM_result/binary_mask.txt");
    // SaveTxt::save_mat(binary_new.prob, "binary_new_probability.txt");
    Imaging_Result CTM_result = imaging.compute(binary_mask.final_mask);
    SaveTxt::save_mat(CTM_result.aerial_image, "CTM_result/aerial_image_B.txt");
    SaveTxt::save_mat(CTM_result.wafer_image, "CTM_result/wafer_image_B.txt");

    // 可视化结果：从 SaveTxt 的真实结果目录反推出项目根，避免写死用户名和路径。
    namespace fs = std::filesystem;
    const fs::path result_root = SaveTxt::_result_path();
    const fs::path project_root = result_root.parent_path();
    const fs::path script = project_root / "scripts/show_multi.py";
    const fs::path venv_python = project_root / ".venv/bin/python";
    const std::string python = fs::is_regular_file(venv_python)
        ? shell_quote(venv_python)
        : "python3";

    if (!fs::is_regular_file(script)) {
        std::cerr << "[warning] 可视化脚本不存在: " << script << '\n';
    } else {
        const fs::path ctm_result = result_root / "CTM_result";
        const std::string cmd =
            python + " " + shell_quote(script) + " " +
            shell_quote(ctm_result / "gray_mask.txt") + " hot " +
            shell_quote(ctm_result / "aerial_image_G.txt") + " gray " +
            shell_quote(ctm_result / "wafer_image_G.txt") + " gray " +
            shell_quote(ctm_result / "binary_mask.txt") + " gray " +
            shell_quote(ctm_result / "aerial_image_B.txt") + " gray " +
            shell_quote(ctm_result / "wafer_image_B.txt") + " gray";
        const int status = std::system(cmd.c_str());
        if (status != 0) {
            std::cerr << "[warning] 可视化脚本退出状态异常: "
                      << status << '\n';
        }
    }



    





    
}
