#include "partition_visualization.h"

#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
using namespace litho;

namespace {

// 为绘图命令中的文件路径保留空格和单引号。
std::string shell_quote(const fs::path& path) {
    std::string quoted = "'";
    for (char ch : path.string()) {
        if (ch == '\'') quoted += "'\"'\"'";
        else quoted += ch;
    }
    return quoted + "'";
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: demo_full_chip_partition [config.yaml] [output_dir] [micro_id]\n";
        return 0;
    }
    const fs::path project = fs::path(__FILE__).parent_path().parent_path();
    const fs::path config = fs::absolute(argc > 1 ? fs::path(argv[1]) : project / "config_full_chip_lsm.yaml");
    const auto root = YAML::LoadFile(config.string());
    const auto full = root[root["full_chip_lsm"].IsDefined() ? "full_chip_lsm" : "full_chip_ctm"];
    const fs::path target_path = config.parent_path() / full["target_path"].as<std::string>();
    const auto target = cv::imread(target_path.string(), cv::IMREAD_GRAYSCALE);
    // 与优化 demo 使用相同的分区接口；只读取目标版图并生成几何，不初始化光学模型或运行优化。
    const auto meta = Partition::build(target.rows, target.cols, full["grid_rows"].as<int>(), full["grid_cols"].as<int>(),
                                       full["halo"].as<int>(), full["guard"].as<int>());
    const std::string case_name = target_path.stem().string() + "_" + std::to_string(meta.grid_rows) + "x" + std::to_string(meta.grid_cols);
    const fs::path output = argc > 2 ? fs::absolute(argv[2]) : project / "result/partition" / case_name;
    save_partition_geometry(meta, root["system"]["pixel_size_nm"].as<double>(), output);
    cv::imwrite((output / "target.bmp").string(), target);
    // 默认高亮网格中间的一块；这里是 Micro 编号，不是像素坐标。
    const int micro_id = argc > 3 ? std::stoi(argv[3]) : (meta.grid_rows / 2) * meta.grid_cols + meta.grid_cols / 2;
    const fs::path cache = output / ".matplotlib";
    fs::create_directories(cache);
    const std::string command = "MPLCONFIGDIR=" + shell_quote(cache) + " XDG_CACHE_HOME=" + shell_quote(cache) + " " +
        shell_quote(project / ".venv/bin/python") + " " + shell_quote(project / "scripts/plot_full_chip_partition.py") +
        " " + shell_quote(output) + " --micro-id " + std::to_string(micro_id);
    if (std::system(command.c_str()) != 0) return 1;
    std::cout << "Partition: " << meta.grid_rows << 'x' << meta.grid_cols << " Micro, tile=" << meta.tile_size
              << ", halo=" << meta.halo << ", guard=" << meta.guard << " pixels\nOutput: " << output << std::endl;
}
