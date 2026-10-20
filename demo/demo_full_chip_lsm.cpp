#include "full_chip_lsm.h"
#include "full_chip_imaging_compare.h"
#include "full_chip_epe.h"
#include "lithography_simulator.h"
#include "save_txt.h"
#include "console_log.h"

#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
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

// 0/1 或覆盖率 mask 映射到 8 位预览；TXT 保留原值，BMP 的边界灰度按最近整数保存。
void save_mask_preview(const Eigen::MatrixXd& mask, const fs::path& path) {
    cv::Mat image(mask.rows(), mask.cols(), CV_8UC1);
    for (int y = 0; y < mask.rows(); ++y) {
        for (int x = 0; x < mask.cols(); ++x) {
            image.at<unsigned char>(y, x) = static_cast<unsigned char>(std::lround(255.0 * mask(y, x)));
        }
    }
    cv::imwrite(path.string(), image);
}

// 导出实际使用的全局 EP 和 Core 几何；坐标为像素索引 (y,x)，Core 区间不含 y1/x1。
void save_ep_points(const Eigen::MatrixXd& target, const PartitionMeta& meta, const MicroGradientOptions& options,
                    const fs::path& output) {
    const auto eps = select_micro_epe_points(target, options);
    std::ofstream points(output / "ep_points.csv");
    points << "ep_id,y,x,micro_id,weight_epe,weight_meef\n" << std::setprecision(17);
    for (const auto& tile : meta.tiles) {
        const auto& core = tile.core_global;
        for (int i = 0; i < eps.eps.rows(); ++i) {
            const double y = eps.eps(i, 0), x = eps.eps(i, 1);
            if (y >= core.y0 && y < core.y1 && x >= core.x0 && x < core.x1) {
                points << i << ',' << y << ',' << x << ',' << tile.id << ',' << eps.weight_epe(i) << ',' << eps.weight_meef(i) << '\n';
            }
        }
    }
    std::ofstream regions(output / "core_regions.csv");
    regions << "micro_id,y0,y1,x0,x1\n";
    for (const auto& tile : meta.tiles) {
        const auto& c = tile.core_global;
        regions << tile.id << ',' << c.y0 << ',' << c.y1 << ',' << c.x0 << ',' << c.x1 << '\n';
    }
    std::cout << "EP points: " << eps.eps.rows() << ", saved to " << output / "ep_points.csv" << std::endl;
}

// 执行 LSM demo；日志由 main 持有，使本函数抛出异常后 catch 的输出仍能写入结果目录。
int run_full_chip_lsm(int argc, char** argv, std::unique_ptr<ConsoleLog>& console_log) {
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
    // 在读取初始 mask、准备 SOCS 和绘图前启动日志，路径始终跟随实际 output_dir。
    fs::create_directories(output);
    const fs::path log_path = output / "console_output.log";
    console_log = std::make_unique<ConsoleLog>(log_path);
    std::cout << "Console log: " << log_path << std::endl;
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
    // 初始化 LSM 优化器
    FullChipLSMOptimizer optimizer(meta, prepare.cache(), params.resist.threshold, params.resist.alpha);
    FullChipLSMOptions options;
    // 命令行参数覆盖配置文件中的迭代次数；其他参数仍沿用配置文件。
    options.iterations = argc > 2 ? std::stoi(argv[2]) : full["iterations"].as<int>();
    options.dx = options.dy = params.system.pixel_size_nm;
    options.cfl = full["cfl"].as<double>();
    options.curvature_weight = full["curvature_weight"].as<double>();
    options.reinit_interval = full["reinit_interval"].as<int>();
    const std::map<std::string, LevelSetMaskType> mask_types = {{"binary", LevelSetMaskType::Binary}, {"gray", LevelSetMaskType::Gray}};
    options.mask_type = mask_types.at(full["levelset_update_mask_type"].as<std::string>());
    options.render_scale = full["levelset_phi_renderer_scale"].as<int>();
    options.parallel_micro = full["parallel_micro"].as<bool>();
    auto& gradient = options.gradient_options;
    const std::map<std::string, MicroLossType> losses = {{"PE", MicroLossType::PE}, {"PEPenalty", MicroLossType::PEPenalty},
                                                       {"EPE", MicroLossType::EPE}, {"PE_EPE", MicroLossType::PE_EPE}};
    gradient.loss_type = losses.at(full["loss_type"].as<std::string>());
    gradient.pe_weight = full["pe_weight"].as<double>();
    gradient.epe_weight = full["epe_weight"].as<double>();
    gradient.penalty_weight = full["penalty_weight"].as<double>();
    gradient.penalty_threshold = full["penalty_threshold"].as<double>();
    gradient.main_region_dilation_pixels = full["main_region_dilation_pixels"].as<int>();
    const std::map<std::string, MicroEpeSampling> sampling = {{"contact", MicroEpeSampling::Contact}, {"metal", MicroEpeSampling::Metal}};
    gradient.epe_sampling = sampling.at(full["epe_sampling"].as<std::string>());
    gradient.epe_all_eps = full["epe_all_eps"].as<bool>();
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
    // 优化前保存选点图，绘图仅消费 C++ 导出的坐标，不在 Python 中重新选点。
    cv::imwrite((output / "target.bmp").string(), input);
    save_ep_points(target, meta, gradient, output);
    const fs::path plot_cache = output / ".matplotlib";
    fs::create_directories(plot_cache);
    const std::string plot_prefix = "MPLCONFIGDIR=" + shell_quote(plot_cache) + " XDG_CACHE_HOME=" + shell_quote(plot_cache) + " " +
                                   shell_quote(project / ".venv/bin/python") + " ";
    const std::string ep_command = plot_prefix + shell_quote(project / "scripts/plot_full_chip_eps.py") + " " + shell_quote(output);
    if (run_logged_command(ep_command) != 0) return 1;
    std::cout << "Full-chip LSM: " << target.rows() << 'x' << target.cols() << ", " << meta.tiles.size()
              << " Micro, tile=" << meta.tile_size << ", iterations=" << options.iterations << std::endl;
    const auto start = std::chrono::steady_clock::now();
    const auto result = optimizer.optimize(binary, target, options);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    save_matrix(result.global_mask, output / "lsm_mask.txt");
    save_matrix(result.global_phi, output / "lsm_phi.txt");
    save_matrix(result.initial_global_phi, output / "initial_phi.txt");
    save_matrix(binary, output / "initial_mask.txt");
    save_matrix(result.initial_global_mask, output / "initial_rendered_mask.txt");
    // 额外保存最佳 phi 的硬二值图；最佳轮选择和最终 Micro 成像仍使用实际渲染 mask。
    const Eigen::MatrixXd best_binary = (result.global_phi.array() >= 0.0).cast<double>().matrix();
    save_matrix(best_binary, output / "lsm_binary_mask.txt");
    save_mask_preview(result.global_mask, output / "lsm_mask.bmp");
    save_mask_preview(best_binary, output / "lsm_binary_mask.bmp");
    save_mask_preview(binary, output / "initial_mask.bmp");
    save_mask_preview(result.initial_global_mask, output / "initial_rendered_mask.bmp");
    std::ofstream history(output / "history.csv");
    // EPE-sq 是 Python 对齐的像素平方目标；EPE 绝对和/均值为 nm，Core 不重叠，total 汇总 Core + Guard。
    history << "iteration,loss,data_loss,cept_loss,min_dt,max_dt,pe_contribution,penalty_contribution,epe_contribution,cept_contribution,"
            << "core_pe,core_bg_penalty_raw,core_pe_penalty,core_epe_squared_pixel,core_epe_abs_sum_nm,core_ep_count,"
            << "core_epe_mean_nm,core_data_loss,total_pe,total_bg_penalty_raw,total_epe_squared_pixel,total_epe_abs_sum_nm,"
            << "total_ep_count,epe_mean_nm\n"
            << std::setprecision(17);
    for (const auto& row : result.history) {
        history << row.iteration << ',' << row.loss << ',' << row.data_loss << ',' << row.cept_loss
                << ',' << row.min_dt << ',' << row.max_dt << ',' << row.pe_contribution << ',' << row.penalty_contribution
                << ',' << row.epe_contribution << ',' << row.cept_contribution << ',' << row.core.pe_loss
                << ',' << row.core.bg_penalty_raw << ',' << row.core.pe_penalty_loss << ',' << row.core.epe_loss
                << ',' << row.core.epe_abs_sum_nm << ',' << row.core.ep_count << ',' << row.core.epe_mean << ',' << row.core.data_loss
                << ',' << row.total_pe_loss << ',' << row.total_bg_penalty_raw << ',' << row.total_epe_loss
                << ',' << row.total_epe_abs_sum_nm
                << ',' << row.total_ep_count << ',' << row.epe_mean << '\n';
    }
    // 最佳轮各块的 Core 统计与 lsm_mask/local_masks 同轮；便于定位损失大的 Micro，而非只看全图汇总。
    std::ofstream cores(output / "core_metrics_best.csv");
    cores << "best_iteration,micro_id,y0,y1,x0,x1,core_pe,core_bg_penalty_raw,core_pe_penalty,"
          << "core_epe_squared_pixel,core_epe_abs_sum_nm,core_ep_count,core_epe_mean_nm,core_data_loss\n" << std::setprecision(17);
    for (const auto& tile : meta.tiles) {
        const auto& c = tile.core_global;
        const auto& m = result.best_core_metrics[tile.id];
        cores << result.best_iteration << ',' << tile.id << ',' << c.y0 << ',' << c.y1 << ',' << c.x0 << ',' << c.x1
              << ',' << m.pe_loss << ',' << m.bg_penalty_raw << ',' << m.pe_penalty_loss << ',' << m.epe_loss
              << ',' << m.epe_abs_sum_nm << ',' << m.ep_count << ',' << m.epe_mean << ',' << m.data_loss << '\n';
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

    // 最佳轮实际渲染的完整局部 mask 分别成像后只拼 Core；复用已有 cache，WI 保留 sigmoid 连续值。
    const auto comparison = compare_micro_imaging(params, meta, prepare.cache(), target, result.local_masks, options.parallel_micro);
    const fs::path comparison_dir = output / "imaging_comparison";
    save_full_chip_imaging_comparison(comparison, comparison_dir, FullChipImagingStage::LSM);
    // EP 在所属 Micro 的完整 AI 上求空间差分，避免跨回拼边界取斜率；每个全局 EP 只输出一次。
    save_micro_ep_epe(comparison.optimized_local_aerial_images, meta, output / "ep_points.csv", params.resist.threshold,
                      params.system.pixel_size_nm, output / "ep_epe.csv");
    std::cout << "Per-point Micro EPE saved to " << output / "ep_epe.csv" << std::endl;
    // 成像和 PE 均由上面的 C++ 计算；Python 只读取 double TXT，自动保存全图和中心放大对比 PNG。
    const std::string command = plot_prefix + shell_quote(project / "scripts/plot_full_chip_imaging_compare.py") +
        " " + shell_quote(comparison_dir);
    if (run_logged_command(command) != 0) return 1;
    return 0;
}

int main(int argc, char** argv) {
    std::unique_ptr<ConsoleLog> console_log;
    try {
        return run_full_chip_lsm(argc, argv, console_log);
    } catch (const std::exception& error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
