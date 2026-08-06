#include "imbinarize.h"
#include "save_txt.h"
#include "litho_prepare.h"
#include "imaging.h"
#include "CTM_Optimizer.h"
#include "simulation_parameters.h"
#include "lithography_simulator.h"
#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <string>



using namespace litho;

int main(int argc, char** argv){
    // std::cout << "========== CTM优化demo ==========\n" << std::endl;

    // 允许从命令行传入 yaml 路径；默认相对于项目根（build 目录下用 "../config.yaml"）
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
    // 保存目标掩膜、 pupil、 source
    SaveTxt::save_mat_int(simulator._mask.data(), "System_result/target_mask.txt");
    SaveTxt::save_mat(simulator._pupil.get_amplitude(), "System_result/pupil.txt");
    SaveTxt::save_mat(simulator._source.get_source_map(), "System_result/source.txt");
    // CTM优化
    auto t_begin = std::chrono::high_resolution_clock::now();
    CTM_Optimizer optimizer(simulator, cache, 50, 0.9);
    Eigen::MatrixXd result= optimizer.optimize();
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

    // 可视化结果
    std::string base = "/Users/wyh/Desktop/学校/Litho_cpp";
    std::string cmd = "python3 " + base + "/scripts/show_multi.py "
                    + base + "/result/CTM_result/gray_mask.txt hot "
                    + base + "/result/CTM_result/aerial_image_G.txt gray "
                    + base + "/result/CTM_result/wafer_image_G.txt gray "
                    + base + "/result/CTM_result/binary_mask.txt gray "
                    + base + "/result/CTM_result/aerial_image_B.txt gray "
                    + base + "/result/CTM_result/wafer_image_B.txt gray";
    std::system(cmd.c_str());



    





    
}