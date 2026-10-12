#include "full_chip_ctm.h"
#include "full_chip_imaging_compare.h"
#include "imbinarize.h"
#include "lithography_simulator.h"

#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

namespace fs = std::filesystem;
using namespace litho;

// 导出灰度 mask 与原 C++ 二值化接口的 final_mask；这里仅转换存储格式，不重新进行阈值处理。
void save_mask(const Eigen::MatrixXd& mask, const Eigen::MatrixXd& binary_mask, const fs::path& directory) {
    cv::Mat gray(mask.rows(), mask.cols(), CV_8UC1), binary(mask.rows(), mask.cols(), CV_8UC1);
    std::ofstream text(directory / "gray_mask.txt");
    std::ofstream binary_text(directory / "binary_mask.txt");
    text << std::setprecision(10) << "# rows=" << mask.rows() << " cols=" << mask.cols() << '\n';
    binary_text << "# rows=" << mask.rows() << " cols=" << mask.cols() << '\n';
    for (int y = 0; y < mask.rows(); ++y) {
        for (int x = 0; x < mask.cols(); ++x) {
            gray.at<unsigned char>(y, x) = static_cast<unsigned char>(std::lround(255.0 * mask(y, x)));
            binary.at<unsigned char>(y, x) = static_cast<unsigned char>(255.0 * binary_mask(y, x));
            text << mask(y, x) << (x + 1 == mask.cols() ? '\n' : ' ');
            binary_text << static_cast<int>(binary_mask(y, x)) << (x + 1 == mask.cols() ? '\n' : ' ');
        }
    }
    cv::imwrite((directory / "gray_mask.bmp").string(), gray);
    cv::imwrite((directory / "binary_mask.bmp").string(), binary);
}

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: demo_full_chip_ctm [config.yaml] [iterations] [learning_rate] [target.bmp]\n";
            return 0;
        }
        const fs::path project = fs::path(__FILE__).parent_path().parent_path();
        const fs::path config = fs::absolute(argc > 1 ? fs::path(argv[1]) : project / "config_full_chip_ctm.yaml");
        YAML::Node root = YAML::LoadFile(config.string());
        YAML::Node full = root["full_chip_ctm"];
        const auto params = SimulationParameters::from_yaml(config.string());
        const fs::path target_path = config.parent_path() / (argc > 4 ? argv[4] : full["target_path"].as<std::string>());
        fs::path output = config.parent_path() / full["output_dir"].as<std::string>();
        if (argc > 4) output = output.parent_path() / target_path.stem();
        const cv::Mat input = cv::imread(target_path.string(), cv::IMREAD_GRAYSCALE);
        Eigen::MatrixXd target(input.rows, input.cols);
        for (int y = 0; y < input.rows; ++y) {
            for (int x = 0; x < input.cols; ++x) target(y, x) = input.at<unsigned char>(y, x) / 255.0;
        }
        // 生成全图分块信息，包含每个 Micro 的位置、尺寸、Halo 与 Guard。以及对应的邻里关系等等
        const auto meta = Partition::build(input.rows, input.cols, full["grid_rows"].as<int>(), full["grid_cols"].as<int>(),
                                           full["halo"].as<int>(), full["guard"].as<int>());
        // 光学系统按含 Halo 的局部矩阵尺寸在外部初始化一次，后续所有 Micro 和所有迭代共享只读 cache。
        const Grid grid(meta.tile_size, params.system.pixel_size_nm, params.optics.na, params.system.wavelength_nm);
        const auto& coords = grid.grid_coords();
        const Pupil pupil(LithographySimulator::_make_pupil_params(params), coords.Fx_2d, coords.Fy_2d);
        const Source source(LithographySimulator::_make_source_params(params, coords.Fx_1d));
        const LithoPrepare prepare(grid, pupil, source, true);
        FullChipCTMOptimizer optimizer(meta, prepare.cache(), params.resist.threshold, params.resist.alpha);
        FullChipCTMOptions options;
        options.iterations = argc > 2 ? std::stoi(argv[2]) : full["iterations"].as<int>();
        options.learning_rate = argc > 3 ? std::stod(argv[3]) : full["learning_rate"].as<double>();
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
        std::cout << "Full-chip CTM: " << input.rows << 'x' << input.cols << ", " << meta.tiles.size()
                  << " Micro, tile=" << meta.tile_size << ", iterations=" << options.iterations
                  << ", theta step=" << options.learning_rate << std::endl;
        const auto start = std::chrono::steady_clock::now();
        const auto result = optimizer.optimize(target, options);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        fs::create_directories(output);
        // 沿用当前入口设置：峰值最低强度/分位数 0.05，面积阈值/最小间距均为 5 像素。
        const auto binary_result = Imbinarize::binarize_advanced(result.global_mask, target, 0.125, 0.1, 5, 5);
        save_mask(result.global_mask, binary_result.final_mask, output);
        cv::imwrite((output / "target.bmp").string(), input);
        std::ofstream history(output / "history.csv");
        // 原始值与加权贡献分开，CSV 与终端日志共用同一轮、同一套损失窗口。
        history << "iteration,loss,data_loss,cept_loss,pe_contribution,penalty_contribution,epe_contribution,cept_contribution,"
                << "pe_loss_raw,bg_penalty_raw,epe_loss_raw\n"
                << std::setprecision(17);
        for (const auto& row : result.history) {
            history << row.iteration << ',' << row.loss << ',' << row.data_loss << ',' << row.cept_loss
                    << ',' << row.pe_contribution << ',' << row.penalty_contribution
                    << ',' << row.epe_contribution << ',' << row.cept_contribution
                    << ',' << row.pe_loss_raw << ',' << row.bg_penalty_raw << ',' << row.epe_loss_raw << '\n';
        }
        // 输出图像对应最佳轮；独立记录最佳轮与末轮，history 仍保留整个优化轨迹。
        std::ofstream summary(output / "summary.csv");
        summary << "best_iteration,best_loss,final_iteration,final_loss\n" << std::setprecision(17);
        summary << result.best_iteration << ',' << result.best_loss << ',' << result.history.back().iteration
                << ',' << result.history.back().loss << '\n';
        // 保存实际使用的参数，包含命令行覆盖后的迭代数、步长和输入路径。
        full["iterations"] = options.iterations;
        full["learning_rate"] = options.learning_rate;
        full["target_path"] = target_path.string();
        full["output_dir"] = output.string();
        std::ofstream(output / "config_used.yaml") << root;
        std::cout << "Completed " << options.iterations << " updates in " << seconds << " s, initial loss="
                  << result.history.front().loss << ", best loss=" << result.best_loss << " at iter " << result.best_iteration
                  << ", final loss=" << result.history.back().loss << "\nOutput: " << output << std::endl;
        // 最佳轮完整局部灰度 mask 分别成像后只拼 Core；复用优化的局部光学 cache，保留各块独立 Guard。
        const auto comparison = compare_micro_imaging(params, meta, prepare.cache(), target, result.local_masks, options.parallel_micro);
        save_full_chip_imaging_comparison(comparison, output / "imaging_comparison");
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
