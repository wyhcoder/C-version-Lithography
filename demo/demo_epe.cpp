#include "save_txt.h"
#include "litho_prepare.h"
#include "imaging.h"
#include "gradient.h"
#include "loss.h"
#include "ep_select.h"
#include "level_set_utils.h"
#include "simulation_parameters.h"
#include "lithography_simulator.h"
#include <Eigen/Dense>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <string>

using namespace litho;

int main(int argc, char** argv) {
    std::string cfg_path = (argc > 1) ? argv[1] : "config.yaml";
    std::cout << "Loading config: " << cfg_path << std::endl;

    SimulationParameters params = SimulationParameters::from_yaml(cfg_path);
    LithographySimulator simulator(params);
    LithoPrepare prepare(simulator._grid, simulator._pupil, simulator._source, true);
    ImagingCache cache = prepare.cache();

    Imaging imaging(cache);
    Gradient gradient(cache, params.resist.threshold, params.resist.alpha);

    // 加载 mask
    Eigen::MatrixXd mask;
    SaveTxt::load_txt("/Users/wyh/Desktop/学校/Litho_cpp/result/CTM_result/binary_mask.txt", mask);

    // 计算 aerial image
    Imaging_Result result = imaging.compute(mask, params.resist.threshold, params.resist.alpha);
    Eigen::MatrixXd aerial = result.aerial_image;
    SaveTxt::save_mat(aerial, "epe_aerial.txt");

    // target mask
    Eigen::MatrixXd target = simulator._mask.data();
    SaveTxt::save_mat(target, "epe_target.txt");

    // ── 1. EPE loss ──────────────────────────────────────────────────
    EpSelect ep_select(target, 1.0, 1.0);
    EpsResult eps_result = ep_select.select_eps_others(6, 2);
    std::cout << "EPE points: " << eps_result.eps.rows() << std::endl;

    double threshold = params.resist.threshold;
    // double dx = params.system.pixel_size_nm;
    double dx = 0.5;
    EpeEvaluation epe_result = Loss::evaluate_epe(
        aerial, eps_result.eps, threshold, dx);
    double epe = epe_result.total_epe;
    double wepe = Loss::weighted_epe(
        eps_result.weight_epe, epe_result.epe_vector);
    std::cout << "EPE loss: " << epe
              << "  WEPE: " << wepe
              << "  vector size: " << epe_result.epe_vector.size()
              << std::endl;

    // ── 2. EPE gradient ──────────────────────────────────────────────
    Eigen::MatrixXd Gm = gradient.epe_gradient(
        aerial, imaging.get_electric_field(), eps_result, threshold, dx);
    SaveTxt::save_mat(Gm, "epe_gradient.txt");
    std::cout << "EPE gradient: min=" << Gm.minCoeff()
              << " max=" << Gm.maxCoeff()
              << " mean=" << Gm.mean() << std::endl;

    // ── 3. 生成 EPE 点叠加图（target + 红点）────────────────────────
    // 保存为 3 通道：target 灰度，EPE 点标记为亮值
    int N = target.rows();
    Eigen::MatrixXd ep_overlay = target;
    for (int i = 0; i < eps_result.eps.rows(); ++i) {
        int y = (int)eps_result.eps(i, 0);
        int x = (int)eps_result.eps(i, 1);
        ep_overlay(y, x) = 0.5;  // 半灰标记 EPE 点
    }
    SaveTxt::save_mat(ep_overlay, "epe_points_overlay.txt");

    // ── 4. 可视化 ───────────────────────────────────────────────────
    std::string base = "/Users/wyh/Desktop/学校/Litho_cpp";
    std::string cmd = "python3 " + base + "/scripts/show_multi.py "
                    + base + "/result/epe_target.txt gray "
                    + base + "/result/epe_points_overlay.txt gray "
                    + base + "/result/epe_aerial.txt hot "
                    + base + "/result/epe_gradient.txt seismic";
    std::system(cmd.c_str());

    return 0;
}
