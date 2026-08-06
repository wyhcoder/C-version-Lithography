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
#include <string>



using namespace litho;

int main(int argc, char** argv){
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
                  << "      ./demo_CTM /absolute/path/to/config.yaml" << std::endl;
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
    SaveTxt::load_txt("/Users/wyh/Desktop/学校/Litho_cpp/result/CTM_result/binary_mask.txt", binary_mask);
    LSM_Optimizer lsm_optimizer(simulator, cache, 4.0, 4.0, 0.5, 0.01, 100);
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

    std::string base = "/Users/wyh/Desktop/学校/Litho_cpp";
    std::string cmd = "python3 " + base + "/scripts/show_multi.py "
                    + base + "/result/LSM_result/LSM_mask.txt gray "
                    + base + "/result/LSM_result/LSM_aerial.txt gray "
                    + base + "/result/LSM_result/LSM_wafer.txt gray "
                    + base + "/result/LSM_result/LSM_initial_sdf.txt viridis "
                    + base + "/result/LSM_result/LSM_sdf.txt  viridis "
                    + base + "/result/LSM_result/LSM_diff.txt viridis";
    std::system(cmd.c_str());
                   





    
}