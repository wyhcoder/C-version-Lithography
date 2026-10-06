#include "full_chip_lsm.h"
#include "full_chip_imaging_compare.h"
#include "lithography_simulator.h"
#include "save_txt.h"

#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

namespace fs = std::filesystem;
using namespace litho;

// 单引号保护绘图命令中的路径，路径中的单引号单独转义。
std::string shell_quote(const fs::path& path) {
    std::string quoted = "'";
    for (char ch : path.string()) {
        if (ch == '\'') quoted += "'\"'\"'";
        else quoted += ch;
    }
    return quoted + "'";
}

// 全局矩阵按 (y,x) 保存 double 原值；phi 的坐标/长度单位与 options.dx/dy 相同。
void save_matrix(const Eigen::MatrixXd& matrix, const fs::path& path) {
    std::ofstream file(path);
    file << "# rows=" << matrix.rows() << " cols=" << matrix.cols() << '\n' << std::setprecision(17) << matrix << '\n';
}

void save_binary_preview(const Eigen::MatrixXd& mask, const fs::path& path) {
    cv::Mat image(mask.rows(), mask.cols(), CV_8UC1);
    for (int y = 0; y < mask.rows(); ++y) {
        for (int x = 0; x < mask.cols(); ++x) image.at<unsigned char>(y, x) = static_cast<unsigned char>(255.0 * mask(y, x));
    }
    cv::imwrite(path.string(), image);
}

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: demo_full_chip_lsm [config.yaml] [iterations] [binary_mask.txt] [output_dir]\n";
        return 0;
    }
    const fs::path project = fs::path(__FILE__).parent_path().parent_path();
    const fs::path config = fs::absolute(argc > 1 ? fs::path(argv[1]) : project / "config_full_chip_lsm.yaml");
    YAML::Node root = YAML::LoadFile(config.string());
    YAML::Node full = root["full_chip_lsm"];
    const auto params = SimulationParameters::from_yaml(config.string());
    const fs::path target_path = config.parent_path() / full["target_path"].as<std::string>();
    const fs::path binary_path = config.parent_path() / (argc > 3 ? argv[3] : full["initial_mask_path"].as<std::string>());
    const fs::path output = config.parent_path() / (argc > 4 ? argv[4] : full["output_dir"].as<std::string>());
    const auto input = cv::imread(target_path.string(), cv::IMREAD_GRAYSCALE);
    Eigen::MatrixXd target(input.rows, input.cols), binary;
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) target(y, x) = input.at<unsigned char>(y, x) / 255.0;
    }
    // CTM 已经通过原 C++ Imbinarize 输出二值结果；LSM 直接读取该 TXT，不重新做一遍阈值筛选。
    SaveTxt::load_txt(binary_path.string(), binary);
    const auto meta = Partition::build(target.rows(), target.cols(), full["grid_rows"].as<int>(), full["grid_cols"].as<int>(),
                                       full["halo"].as<int>(), full["guard"].as<int>());
    // 成像网格按完整 Core + Halo 尺寸准备一次；每个线程独占可变 FFT 缓冲区，光学 cache 只读共享。
    const Grid grid(meta.tile_size, params.system.pixel_size_nm, params.optics.na, params.system.wavelength_nm);
    const auto& coords = grid.grid_coords();
    const Pupil pupil(LithographySimulator::_make_pupil_params(params), coords.Fx_2d, coords.Fy_2d);
    const Source source(LithographySimulator::_make_source_params(params, coords.Fx_1d));
    const LithoPrepare prepare(grid, pupil, source, true);
    FullChipLSMOptimizer optimizer(meta, prepare.cache(), params.resist.threshold, params.resist.alpha);
    FullChipLSMOptions options;
    options.iterations = argc > 2 ? std::stoi(argv[2]) : full["iterations"].as<int>();
    options.dx = options.dy = params.system.pixel_size_nm;
    options.cfl = full["cfl"].as<double>();
    options.curvature_weight = full["curvature_weight"].as<double>();
    options.reinit_interval = full["reinit_interval"].as<int>();
    options.parallel_micro = full["parallel_micro"].as<bool>();
    auto& gradient = options.gradient_options;
    const std::map<std::string, MicroLossType> losses = {{"PE", MicroLossType::PE}, {"PEPenalty", MicroLossType::PEPenalty},
                                                       {"EPE", MicroLossType::EPE}, {"PE_EPE", MicroLossType::PE_EPE}};
    gradient.loss_type = losses.at(full["loss_type"].as<std::string>());
    gradient.pe_weight = full["pe_weight"].as<double>();
    gradient.epe_weight = full["epe_weight"].as<double>();
    gradient.penalty_weight = full["penalty_weight"].as<double>();
    gradient.penalty_threshold = full["penalty_threshold"].as<double>();
    gradient.epe_interval_line = full["epe_interval_line"].as<int>();
    gradient.epe_interval_corner = full["epe_interval_corner"].as<int>();
    gradient.epe_pixel_size = params.system.pixel_size_nm;
    gradient.enable_cept = full["enable_cept"].as<bool>();
    gradient.cept_weight = full["cept_weight"].as<double>();
    gradient.enable_grad_fusion = full["enable_grad_fusion"].as<bool>();
    const std::map<std::string, GradientFusionMode> modes = {{"Average", GradientFusionMode::Average},
                                                           {"Linear", GradientFusionMode::Linear}};
    gradient.fusion_mode = modes.at(full["fusion_mode"].as<std::string>());
    gradient.fusion_weight_owner = full["fusion_weight_owner"].as<double>();
    gradient.fusion_weight_neighbor = full["fusion_weight_neighbor"].as<double>();
    std::cout << "Full-chip LSM: " << target.rows() << 'x' << target.cols() << ", " << meta.tiles.size()
              << " Micro, tile=" << meta.tile_size << ", iterations=" << options.iterations << std::endl;
    const auto start = std::chrono::steady_clock::now();
    const auto result = optimizer.optimize(binary, target, options);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    fs::create_directories(output);
    save_matrix(result.global_mask, output / "lsm_mask.txt");
    save_matrix(result.global_phi, output / "lsm_phi.txt");
    save_matrix(result.initial_global_phi, output / "initial_phi.txt");
    save_matrix(binary, output / "initial_mask.txt");
    save_binary_preview(result.global_mask, output / "lsm_mask.bmp");
    save_binary_preview(binary, output / "initial_mask.bmp");
    cv::imwrite((output / "target.bmp").string(), input);
    std::ofstream history(output / "history.csv");
    // 原六列保留；追加已加权分项，便于直接核对 data_loss 和总 loss 的组成。
    history << "iteration,loss,data_loss,cept_loss,min_dt,max_dt,pe_contribution,penalty_contribution,epe_contribution,cept_contribution\n"
            << std::setprecision(17);
    for (const auto& row : result.history) {
        history << row.iteration << ',' << row.loss << ',' << row.data_loss << ',' << row.cept_loss
                << ',' << row.min_dt << ',' << row.max_dt << ',' << row.pe_contribution << ',' << row.penalty_contribution
                << ',' << row.epe_contribution << ',' << row.cept_contribution << '\n';
    }
    std::ofstream summary(output / "summary.csv");
    summary << "best_iteration,best_loss,final_iteration,final_loss,seconds\n" << std::setprecision(17);
    summary << result.best_iteration << ',' << result.best_loss << ',' << result.history.back().iteration
            << ',' << result.history.back().loss << ',' << seconds << '\n';
    full["iterations"] = options.iterations;
    full["target_path"] = target_path.string();
    full["initial_mask_path"] = binary_path.string();
    full["output_dir"] = output.string();
    std::ofstream(output / "config_used.yaml") << root;
    std::cout << "LSM completed in " << seconds << " s, initial loss=" << result.history.front().loss
              << ", best loss=" << result.best_loss << " at iter " << result.best_iteration
              << ", final loss=" << result.history.back().loss << "\nOutput: " << output << std::endl;

    // 用全图尺寸另建光学系统：原始 target 与最佳轮 Core 回拼的 LSM mask 共用同一模型，WI 保留连续值。
    const int source_size = full["imaging_source_grid_size"].as<int>(meta.tile_size);
    const auto comparison = compare_full_chip_imaging(params, target, result.global_mask, source_size);
    const fs::path comparison_dir = output / "imaging_comparison";
    save_full_chip_imaging_comparison(comparison, comparison_dir, FullChipImagingStage::LSM);
    // 成像和 PE 均由上面的 C++ 计算；Python 只读取 double TXT，自动保存全图和中心放大对比 PNG。
    const fs::path plot_cache = output / ".matplotlib";
    fs::create_directories(plot_cache);
    const std::string command = "MPLCONFIGDIR=" + shell_quote(plot_cache) + " XDG_CACHE_HOME=" + shell_quote(plot_cache) + " " +
        shell_quote(project / ".venv/bin/python") + " " + shell_quote(project / "scripts/plot_full_chip_imaging_compare.py") +
        " " + shell_quote(comparison_dir);
    if (std::system(command.c_str()) != 0) return 1;
}
