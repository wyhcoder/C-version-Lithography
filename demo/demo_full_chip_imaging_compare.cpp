#include "full_chip_imaging_compare.h"

#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace fs = std::filesystem;
using namespace litho;

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: demo_full_chip_imaging_compare [config.yaml] [mask.txt] [output_dir]\n";
        return 0;
    }
    const fs::path project = fs::path(__FILE__).parent_path().parent_path();
    const fs::path config = fs::absolute(argc > 1 ? fs::path(argv[1]) : project / "config_full_chip_ctm.yaml");
    const auto root = YAML::LoadFile(config.string());
    // 根据配置段选择已有 CTM 灰度结果或 LSM 最佳渲染 mask；评价始终与原始 target 对照。
    const bool is_lsm = root["full_chip_lsm"].IsDefined();
    const auto full = root[is_lsm ? "full_chip_lsm" : "full_chip_ctm"];
    const auto params = SimulationParameters::from_yaml(config.string());
    const fs::path result_dir = config.parent_path() / full["output_dir"].as<std::string>();
    const fs::path mask_path = argc > 2 ? fs::path(argv[2]) : result_dir / (is_lsm ? "lsm_mask.txt" : "gray_mask.txt");
    const fs::path output = argc > 3 ? fs::path(argv[3]) : result_dir / "imaging_comparison";
    const fs::path target_path = config.parent_path() / full["target_path"].as<std::string>();
    const auto input = cv::imread(target_path.string(), cv::IMREAD_GRAYSCALE);
    Eigen::MatrixXd target(input.rows, input.cols), mask(input.rows, input.cols);
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) target(y, x) = input.at<unsigned char>(y, x) / 255.0;
    }
    // 读取保存的 double 原值：CTM 和 LSM Gray 都保留灰度，忽略 TXT 尺寸注释，不再次二值化。
    std::ifstream file(mask_path);
    std::string line;
    int y = 0;
    while (std::getline(file, line)) {
        if (line.empty() || line.front() == '#') continue;
        std::istringstream values(line);
        for (int x = 0; x < mask.cols(); ++x) values >> mask(y, x);
        ++y;
    }
    const int source_size = full["imaging_source_grid_size"].as<int>(256);
    const auto comparison = compare_full_chip_imaging(params, target, mask, source_size);
    const auto stage = is_lsm ? FullChipImagingStage::LSM : FullChipImagingStage::CTM;
    save_full_chip_imaging_comparison(comparison, output, stage);
    // 保存实际评价配置和 mask 来源，便于复现同一张最佳 mask 的成像结果。
    std::ofstream(output / "evaluation_config.yaml") << root;
    std::ofstream(output / (is_lsm ? "input_lsm_path.txt" : "input_gray_path.txt")) << fs::absolute(mask_path).string() << '\n';
    std::cout << "Full-frame imaging comparison saved to " << output << std::endl;
}
