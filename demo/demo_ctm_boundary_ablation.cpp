#include "CTM_Optimizer.h"
#include "full_chip_ctm.h"
#include "full_chip_imaging_compare.h"
#include "lithography_simulator.h"
#include "mask_continuity.h"
#include "partition_visualization.h"

#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>

namespace fs = std::filesystem;
using namespace litho;

namespace {

struct ExperimentCase {
    const char* name;
    bool fusion, cept; // 只改变这两个开关；各组从同一 target 初始化 theta。
};

void save_matrix(const Eigen::MatrixXd& matrix, const fs::path& path) {
    std::ofstream output(path);
    output << "# rows=" << matrix.rows() << " cols=" << matrix.cols() << '\n' << std::setprecision(17) << matrix << '\n';
}

void save_mask_preview(const Eigen::MatrixXd& mask, const fs::path& path) {
    cv::Mat image(mask.rows(), mask.cols(), CV_8UC1);
    for (int y = 0; y < mask.rows(); ++y) {
        for (int x = 0; x < mask.cols(); ++x) image.at<unsigned char>(y, x) = std::lround(255.0 * mask(y, x));
    }
    cv::imwrite(path.string(), image);
}

std::string shell_quote(const fs::path& path) {
    std::string quoted = "'";
    for (char ch : path.string()) {
        if (ch == '\'') quoted += "'\"'\"'";
        else quoted += ch;
    }
    return quoted + "'";
}

// 记录实际 Core 回拼图，同时独立统计完整局部 mask 的 Guard-Core 差异；不能从回拼 mask 恢复独立 Guard。
void save_snapshot(const ExperimentCase& item, const char* snapshot, int iteration, const FullChipCTMResult& result,
                   const std::vector<Eigen::MatrixXd>& masks, const FullChipCTMIteration& row, double seconds,
                   const SimulationParameters& params, const PartitionMeta& meta, const ImagingCache& cache,
                   const Eigen::MatrixXd& target, const PartitionRect& roi, bool parallel, const fs::path& output,
                   std::ofstream& metrics) {
    const auto directory = output / item.name / snapshot;
    fs::create_directories(directory);
    const auto comparison = compare_micro_imaging(params, meta, cache, target, masks, parallel);
    const auto continuity = evaluate_mask_continuity(meta, masks, target);
    save_matrix(comparison.optimized_mask, directory / "mask.txt");
    save_mask_preview(comparison.optimized_mask, directory / "mask.bmp");
    // 放大矩阵仍来自全局 Core 回拼；ROI 为全局半开区间，TXT 本身从局部 (0,0) 开始排列。
    save_matrix(comparison.optimized_mask.block(roi.y0, roi.x0, roi.rows(), roi.cols()), directory / "roi_mask.txt");
    save_matrix(comparison.optimized_image.aerial_image.block(roi.y0, roi.x0, roi.rows(), roi.cols()), directory / "roi_ai.txt");
    save_matrix(comparison.optimized_image.wafer_image.block(roi.y0, roi.x0, roi.rows(), roi.cols()), directory / "roi_wi.txt");
    metrics << item.name << ',' << item.fusion << ',' << item.cept << ',' << snapshot << ',' << iteration << ','
            << result.best_iteration << ',' << result.history.back().iteration << ',' << row.loss << ',' << row.data_loss << ','
            << row.cept_loss << ',' << seconds << ',' << comparison.optimized_pe << ',' << continuity.overlap_samples << ','
            << continuity.overlap_sse << ',' << continuity.overlap_rmse << ',' << continuity.overlap_mae << ','
            << continuity.overlap_max_abs << ',' << continuity.seam_samples << ',' << continuity.seam_jump_mae << ','
            << continuity.seam_reference_error_mae << ',' << continuity.flat_seam_samples << ','
            << continuity.flat_seam_jump_mae << ',' << continuity.foreground_seam_samples << ','
            << continuity.foreground_seam_jump_mae << '\n';
    metrics.flush();
    std::cout << item.name << '/' << snapshot << ": overlap RMSE=" << continuity.overlap_rmse
              << ", foreground seam jump=" << continuity.foreground_seam_jump_mae << std::endl;
}

} // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: demo_ctm_boundary_ablation [config.yaml] [iterations] [output_dir]\n";
        return 0;
    }
    const auto project = fs::path(__FILE__).parent_path().parent_path();
    const auto config = fs::absolute(argc > 1 ? fs::path(argv[1]) : project / "config_ctm_boundary_ablation.yaml");
    auto root = YAML::LoadFile(config.string());
    auto full = root["full_chip_ctm"];
    const auto params = SimulationParameters::from_yaml(config.string());
    const auto target_path = config.parent_path() / full["target_path"].as<std::string>();
    const auto output = argc > 3 ? fs::absolute(argv[3]) : config.parent_path() / full["output_dir"].as<std::string>();
    const auto input = cv::imread(target_path.string(), cv::IMREAD_GRAYSCALE);
    Eigen::MatrixXd target(input.rows, input.cols);
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) target(y, x) = input.at<unsigned char>(y, x) / 255.0;
    }
    const auto meta = Partition::build(target.rows(), target.cols(), full["grid_rows"].as<int>(), full["grid_cols"].as<int>(),
                                       full["halo"].as<int>(), full["guard"].as<int>());
    FullChipCTMOptions options;
    options.iterations = argc > 2 ? std::stoi(argv[2]) : full["iterations"].as<int>();
    options.learning_rate = full["learning_rate"].as<double>();
    options.parallel_micro = full["parallel_micro"].as<bool>();
    options.retain_final_masks = true;
    auto& gradient = options.gradient_options;
    const std::map<std::string, MicroLossType> losses = {{"PE", MicroLossType::PE}, {"PEPenalty", MicroLossType::PEPenalty},
        {"EPE", MicroLossType::EPE}, {"PE_EPE", MicroLossType::PE_EPE}};
    gradient.loss_type = losses.at(full["loss_type"].as<std::string>());
    gradient.pe_weight = full["pe_weight"].as<double>();
    gradient.epe_weight = full["epe_weight"].as<double>();
    gradient.penalty_weight = full["penalty_weight"].as<double>();
    gradient.penalty_threshold = full["penalty_threshold"].as<double>();
    gradient.epe_pixel_size = params.system.pixel_size_nm;
    gradient.epe_interval_line = full["epe_interval_line"].as<int>();
    gradient.epe_interval_corner = full["epe_interval_corner"].as<int>();
    gradient.cept_weight = full["cept_weight"].as<double>();
    gradient.fusion_mode = full["fusion_mode"].as<std::string>() == "Linear" ? GradientFusionMode::Linear : GradientFusionMode::Average;
    gradient.fusion_weight_owner = full["fusion_weight_owner"].as<double>();
    gradient.fusion_weight_neighbor = full["fusion_weight_neighbor"].as<double>();
    const auto region = root["ablation"]["roi"];
    const PartitionRect roi{region["y0"].as<int>(), region["y1"].as<int>(), region["x0"].as<int>(), region["x1"].as<int>()};
    fs::create_directories(output);
    save_partition_geometry(meta, params.system.pixel_size_nm, output);
    cv::imwrite((output / "target.bmp").string(), input);
    save_matrix(target.block(roi.y0, roi.x0, roi.rows(), roi.cols()), output / "roi_target.txt");
    const auto theta = CTM_Optimizer::initialize_theta(target);
    save_matrix((0.5 * (1.0 + theta.array().cos())).matrix(), output / "initial_mask.txt");
    full["target_path"] = target_path.string();
    full["output_dir"] = output.string();
    full["iterations"] = options.iterations;
    std::ofstream(output / "config_used.yaml") << root;
    std::ofstream roi_csv(output / "roi.csv");
    roi_csv << "y0,y1,x0,x1,seam_x,ai_threshold,pixel_size_nm\n";
    roi_csv << roi.y0 << ',' << roi.y1 << ',' << roi.x0 << ',' << roi.x1 << ',' << root["ablation"]["seam_x"].as<int>()
            << ',' << params.resist.threshold << ',' << params.system.pixel_size_nm << '\n';
    roi_csv.close();
    // 四组共享完全相同的光学准备；compact_socs 只降低 SVD 中间矩阵内存，不改变采样或局部 FFT 尺寸。
    const Grid grid(meta.tile_size, params.system.pixel_size_nm, params.optics.na, params.system.wavelength_nm);
    const auto& coords = grid.grid_coords();
    const Pupil pupil(LithographySimulator::_make_pupil_params(params), coords.Fx_2d, coords.Fy_2d);
    const Source source(LithographySimulator::_make_source_params(params, coords.Fx_1d));
    const LithoPrepare prepare(grid, pupil, source, true, root["ablation"]["compact_socs"].as<bool>());
    FullChipCTMOptimizer optimizer(meta, prepare.cache(), params.resist.threshold, params.resist.alpha);
    const std::array<ExperimentCase, 4> cases = {{{"off", false, false}, {"fusion_only", true, false},
                                               {"cept_only", false, true}, {"on", true, true}}};
    std::ofstream metrics(output / "metrics.csv");
    metrics << std::setprecision(17)
            << "case,fusion,cept,snapshot,iteration,best_iteration,iterations,optimizer_loss,data_loss,cept_loss_logged,"
            << "optimization_seconds,core_pe,overlap_samples,overlap_sse,overlap_rmse,overlap_mae,overlap_max_abs,"
            << "seam_samples,seam_jump_mae,seam_reference_error_mae,flat_seam_samples,flat_seam_jump_mae,"
            << "foreground_seam_samples,foreground_seam_jump_mae\n";
    for (const auto& item : cases) {
        gradient.enable_grad_fusion = item.fusion;
        gradient.enable_cept = item.cept;
        std::cout << "\n=== CTM ablation " << item.name << ": fusion=" << item.fusion << ", CEPT=" << item.cept
                  << ", iterations=" << options.iterations << " ===" << std::endl;
        const auto start = std::chrono::steady_clock::now();
        const auto result = optimizer.optimize(target, options);
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const auto case_dir = output / item.name;
        fs::create_directories(case_dir);
        auto case_config = YAML::Clone(root);
        case_config["full_chip_ctm"]["enable_grad_fusion"] = item.fusion;
        case_config["full_chip_ctm"]["enable_cept"] = item.cept;
        case_config["full_chip_ctm"]["output_dir"] = case_dir.string();
        std::ofstream(case_dir / "config_used.yaml") << case_config;
        std::ofstream history(case_dir / "history.csv");
        history << "iteration,loss,data_loss,cept_loss\n" << std::setprecision(17);
        for (const auto& row : result.history) {
            history << row.iteration << ',' << row.loss << ',' << row.data_loss << ',' << row.cept_loss << '\n';
        }
        // 主对比使用完全相同的更新次数；最佳轮另存，保持生产 demo 的最佳结果语义不变。
        save_snapshot(item, "final", options.iterations, result, result.final_local_masks, result.history.back(), seconds,
                      params, meta, prepare.cache(), target, roi, options.parallel_micro, output, metrics);
        save_snapshot(item, "best", result.best_iteration, result, result.local_masks, result.history[result.best_iteration], seconds,
                      params, meta, prepare.cache(), target, roi, options.parallel_micro, output, metrics);
    }
    metrics.close();
    const auto cache = output / ".matplotlib";
    fs::create_directories(cache);
    const std::string command = "MPLCONFIGDIR=" + shell_quote(cache) + " XDG_CACHE_HOME=" + shell_quote(cache) + " " +
        shell_quote(project / ".venv/bin/python") + " " + shell_quote(project / "scripts/plot_ctm_boundary_ablation.py") + " " +
        shell_quote(output);
    return std::system(command.c_str()) == 0 ? 0 : 1;
}
