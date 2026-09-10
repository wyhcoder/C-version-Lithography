#include "save_txt.h"
#include "litho_prepare.h"
#include "imaging.h"
#include "simulation_parameters.h"
#include "lithography_simulator.h"
#include "LSM_Optimizer.h"
#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <filesystem>
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
    namespace fs = std::filesystem;
    // std::cout << "========== LSM优化demo ==========\n" << std::endl;

    // 允许从命令行传入 yaml 路径；默认相对于项目根（build 目录下用 "../config.yaml"）
    

    // LSM优化
    std::string cfg_path = (argc > 1) ? argv[1] : "config.yaml";
    std::cout << "Loading config: " << cfg_path << std::endl;

    SimulationParameters params;
    try {
        params = SimulationParameters::from_yaml(cfg_path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load config: " << e.what() << "\n"
                  << "Hint: run from project root, or pass path explicitly:\n"
                  << "      ./demo_LSM /absolute/path/to/config.yaml" << std::endl;
        return 1;
    }
    std::cout << "Loaded config: " << std::endl;
    auto t0 = std::chrono::high_resolution_clock::now();
    LithographySimulator simulator(params);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Simulator time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() * 0.001 << " s" << std::endl;
    
    LithoPrepare prepare(simulator._grid, simulator._pupil, simulator._source, true);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "Prepare time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() * 0.001<< " s" << std::endl;
    
    // 获取提前计算的缓存
    ImagingCache cache = prepare.cache();
    // LSM 优化
    Imaging imaging(cache);
    Eigen::MatrixXd binary_mask;
    const fs::path result_root = SaveTxt::_result_path();
    const fs::path project_root = result_root.parent_path();
    const fs::path ctm_binary_mask =
        result_root / "CTM_result/binary_mask.txt";
    SaveTxt::load_txt(ctm_binary_mask.string(), binary_mask);
    //(LithographySimulator& simulator, ImagingCache& cache, double dx, double dy, double cfl, double b, int iter);
    LSM_Optimizer lsm_optimizer(simulator, cache, params.system.pixel_size_nm , params.system.pixel_size_nm, 0.9, 0.01, 100);
    // 启用非主图形区域显影惩罚（penalty=10, threshold=0.01）
    lsm_optimizer.set_penalty(0.0, 0.0);
    auto t_begin = std::chrono::high_resolution_clock::now();
    Level_set_result lsm_result = lsm_optimizer.optimize(binary_mask);
    auto t_end = std::chrono::high_resolution_clock::now();
    std::cout << "Optimize time: " << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_begin).count() * 0.001 << " s" << std::endl;
    SaveTxt::save_mat(lsm_result.mask, "LSM_result/LSM_mask.txt");

    Imaging_Result lsm_imaging = imaging.compute(lsm_result.mask);
    SaveTxt::save_mat(lsm_imaging.aerial_image, "LSM_result/LSM_aerial.txt");
    SaveTxt::save_mat(lsm_imaging.wafer_image, "LSM_result/LSM_wafer.txt");
    SaveTxt::save_mat(lsm_result.initial_sdf, "LSM_result/LSM_initial_sdf.txt");
    SaveTxt::save_mat(lsm_result.sdf, "LSM_result/LSM_sdf.txt");
    Eigen::MatrixXd diff = lsm_result.mask - binary_mask;
    SaveTxt::save_mat(diff, "LSM_result/LSM_diff.txt");

    // 可视化结果：基于实际项目根拼接路径，不依赖某台机器的用户名。
    const fs::path script = project_root / "scripts/show_multi.py";
    const fs::path venv_python = project_root / ".venv/bin/python";
    const std::string python = fs::is_regular_file(venv_python)
        ? shell_quote(venv_python)
        : "python3";

    if (!fs::is_regular_file(script)) {
        std::cerr << "[warning] 可视化脚本不存在: " << script << '\n';
    } else {
        const fs::path lsm_result_dir = result_root / "LSM_result";
        const std::string cmd =
            python + " " + shell_quote(script) + " " +
            shell_quote(lsm_result_dir / "LSM_mask.txt") + " gray " +
            shell_quote(lsm_result_dir / "LSM_aerial.txt") + " gray " +
            shell_quote(lsm_result_dir / "LSM_wafer.txt") + " gray " +
            shell_quote(lsm_result_dir / "LSM_initial_sdf.txt") + " viridis " +
            shell_quote(lsm_result_dir / "LSM_sdf.txt") + " viridis " +
            shell_quote(lsm_result_dir / "LSM_diff.txt") + " viridis";
        const int status = std::system(cmd.c_str());
        if (status != 0) {
            std::cerr << "[warning] 可视化脚本退出状态异常: "
                      << status << '\n';
        }
    }
                   





    
}
