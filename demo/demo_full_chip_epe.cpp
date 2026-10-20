#include "full_chip_epe.h"
#include "save_txt.h"

#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace litho;

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) == "--help") {
        std::cout << "Usage: demo_full_chip_epe <lsm_result_dir>\n";
        return 0;
    }
    const auto directory = fs::absolute(argv[1]);
    const auto output = directory / "ep_epe.csv";
    std::ifstream sampling(directory / "imaging_comparison/sampling.csv");
    std::string header, metadata;
    std::getline(sampling, header);
    std::getline(sampling, metadata);
    if (metadata.find("micro_core") != std::string::npos) {
        // Core 回拼 AI 不保留独立 Guard 的斜率；新结果直接读取 demo 在完整局部 AI 上导出的逐点值。
        std::ifstream input(output);
        std::string line;
        std::getline(input, line);
        int count = 0;
        double total = 0.0, maximum = 0.0;
        while (std::getline(input, line)) {
            const double epe = std::stod(line.substr(line.rfind(',') + 1));
            if (!std::isfinite(epe)) continue;
            ++count;
            total += epe;
            maximum = std::max(maximum, epe);
        }
        std::cout << std::fixed << std::setprecision(6) << "Micro Core EPE: " << count << " EP, mean="
                  << total / count << " nm, max=" << maximum << " nm\nRead from " << output << std::endl;
        return 0;
    }
    // 使用该次实验保存的参数和全图 AI；不读取当前根配置，不重跑优化或成像。
    const auto config = YAML::LoadFile((directory / "config_used.yaml").string());
    Eigen::MatrixXd aerial_image;
    SaveTxt::load_txt((directory / "imaging_comparison/optimized_ai.txt").string(), aerial_image);
    const auto evaluation = save_full_chip_ep_epe(aerial_image, directory / "ep_points.csv", config["resist"]["threshold"].as<double>(),
                                                config["system"]["pixel_size_nm"].as<double>(), output);
    const int count = evaluation.active_count;
    double maximum = 0.0;
    for (int i = 0; i < evaluation.absolute_epe_nm.size(); ++i) {
        if (std::isfinite(evaluation.absolute_epe_nm(i))) maximum = std::max(maximum, evaluation.absolute_epe_nm(i));
    }
    std::cout << std::fixed << std::setprecision(6) << "Full-frame EPE: " << count << " active EP, mean="
              << evaluation.absolute_sum_nm / count << " nm, max=" << maximum << " nm\n"
              << "Saved to " << output << std::endl;
}
