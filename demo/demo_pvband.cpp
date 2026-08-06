#include "save_txt.h"
#include "simulation_parameters.h"
#include "lithography_simulator.h"
#include "pv_band_computer.h"
#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <string>

using namespace litho;

int main(int argc, char** argv) {
    // ── 1. 加载 config ────────────────────────────────────────────
    std::string cfg_path = (argc > 1) ? argv[1] : "config.yaml";
    std::cout << "Loading config: " << cfg_path << std::endl;

    SimulationParameters params;
    try {
        params = SimulationParameters::from_yaml(cfg_path);
    } catch (const std::exception& e) {
        std::cerr << "Failed to load config: " << e.what() << std::endl;
        return 1;
    }

    // ── 2. 构建 simulator ─────────────────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();
    LithographySimulator simulator(params);
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "Simulator time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() * 0.001
              << " s" << std::endl;

    // ── 3. 加载 mask（用 CTM 优化后的 binary_mask 做测试）─────────
    Eigen::MatrixXd mask;
    SaveTxt::load_txt("/Users/wyh/Desktop/学校/Litho_cpp/result/CTM_result/binary_mask.txt", mask);
    std::cout << "Loaded mask: " << mask.rows() << "x" << mask.cols() << std::endl;

    // ── 4. 构造 PvbandComputer ────────────────────────────────────
    // 离焦: -100 ~ +100 nm, 3 步 → {-100, 0, +100}
    // 剂量容差: ±5% → dose_margin = 0.05
    // full = 3 threshold × 3 defocus；defocus_only = nominal threshold；dose_only = z=0 dose corners
    double dose_margin = 0.05;
    double defocus_range = 100.0;   // total range，内部转换为 [-100, +100] nm
    int    defocus_steps = 3;
    std::string mode = "full";

    std::cout << "\n=== PV Band Computation ===" << std::endl;
    std::cout << "  dose_margin   : ±" << (dose_margin * 100) << "%" << std::endl;
    std::cout << "  defocus_range : [" << (-defocus_range/2) << ", +"
              << (defocus_range/2) << "] nm" << std::endl;
    std::cout << "  defocus_steps : " << defocus_steps << std::endl;
    std::cout << "  mode          : " << mode << std::endl;

    PvbandComputer pvb(simulator, dose_margin, defocus_range, defocus_steps, mode);

    // ── 5. 计算 PV band 与其对 mask 的梯度 ─────────────────────────
    auto t_begin = std::chrono::high_resolution_clock::now();
    const Pvband_result pv_result = pvb.compute_pvband(mask);
    auto t_end = std::chrono::high_resolution_clock::now();
    const Eigen::MatrixXd& pv_map = pv_result.pv_map;
    std::cout << "PV band compute time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_begin).count() * 0.001
              << " s" << std::endl;

    std::cout << "PV map: min=" << pv_map.minCoeff()
              << " max=" << pv_map.maxCoeff()
              << " mean=" << pv_map.mean()
              << " sum=" << pv_result.pv_loss << std::endl;

    auto g_begin = std::chrono::high_resolution_clock::now();
    const Eigen::MatrixXd pv_gradient = pvb.compute_pvloss_gradient(mask);
    auto g_end = std::chrono::high_resolution_clock::now();
    std::cout << "PV gradient compute time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(g_end - g_begin).count() * 0.001
              << " s" << std::endl;
    std::cout << "PV gradient: min=" << pv_gradient.minCoeff()
              << " max=" << pv_gradient.maxCoeff()
              << " mean=" << pv_gradient.mean()
              << " norm=" << pv_gradient.norm() << std::endl;

    // ── 6. 保存结果 ──────────────────────────────────────────────
    SaveTxt::save_mat(pv_map, "pv_band_map.txt");
    SaveTxt::save_mat(pv_gradient, "pv_band_gradient.txt");
    SaveTxt::save_mat(mask,   "pv_band_input_mask.txt");

    // ── 7. 可视化 ────────────────────────────────────────────────
    std::string base = "/Users/wyh/Desktop/学校/Litho_cpp";
    std::string cmd = "python3 " + base + "/scripts/show_multi.py "
                    + base + "/result/pv_band_input_mask.txt gray "
                    + base + "/result/pv_band_map.txt hot "
                    + base + "/result/pv_band_gradient.txt seismic";
    std::system(cmd.c_str());

    return 0;
}
