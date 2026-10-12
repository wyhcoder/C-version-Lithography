#include "level_set_utils.h"
#include "levelset_mask_renderer.h"
#include "lithography_simulator.h"
#include "micro_gradient_exchange.h"
#include "save_txt.h"

#include <opencv2/imgcodecs.hpp>
#include <yaml-cpp/yaml.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace litho;

namespace {

// 保存同一组局部 mask 像素上两项梯度的统计；包含零值，不按各项非零区域分别计算 RMS。
struct GradientSamples {
    std::vector<double> pe_abs, epe_abs; // mask 梯度的绝对值；每个元素对应一个 Micro 的局部像素。
    double pe_square = 0.0, epe_square = 0.0, dot = 0.0; // 平方和及同位置点积，用于 RMS 和方向余弦。

    void add(double pe, double epe) {
        pe_abs.push_back(std::abs(pe));
        epe_abs.push_back(std::abs(epe));
        pe_square += pe * pe;
        epe_square += epe * epe;
        dot += pe * epe;
    }
};

struct Magnitude {
    double rms, mean_abs, p99_abs, max_abs;
};

// 输入同一统计区域的绝对梯度和平方和；P99 使用最近秩定义 ceil(0.99*N)-1，计入零值。
Magnitude magnitude(std::vector<double>& values, double square) {
    if (values.empty()) return {0.0, 0.0, 0.0, 0.0}; // 空边界带没有可统计的像素。
    double sum = 0.0;
    for (double value : values) sum += value;
    std::sort(values.begin(), values.end());
    const auto index = static_cast<std::size_t>(std::ceil(0.99 * values.size())) - 1;
    return {std::sqrt(square / values.size()), sum / values.size(), values[index], values.back()};
}

// 输出同一支持区域的原始尺度，以及假设启用混合损失时 YAML 配置权重产生的 RMS 比例。
// 分母为零时该比例无定义，CSV 写空字段，不把它伪装成 0；有效优化系数单独写入说明文件。
void write_stats(std::ofstream& csv, const std::string& stage, const std::string& region, int id,
                 GradientSamples& samples, double pe_weight, double epe_weight) {
    const auto count = samples.pe_abs.size();
    const auto pe = magnitude(samples.pe_abs, samples.pe_square);
    const auto epe = magnitude(samples.epe_abs, samples.epe_square);
    csv << stage << ',' << region << ',' << id << ',' << count << ','
        << pe.rms << ',' << epe.rms << ',' << pe.mean_abs << ',' << epe.mean_abs << ','
        << pe.p99_abs << ',' << epe.p99_abs << ',' << pe.max_abs << ',' << epe.max_abs << ',';
    if (pe.rms != 0.0) csv << epe.rms / pe.rms;
    csv << ',' << pe_weight * pe.rms << ',' << epe_weight * epe.rms << ',';
    if (pe_weight * pe.rms != 0.0) csv << epe_weight * epe.rms / (pe_weight * pe.rms);
    csv << ',';
    if (samples.pe_square * samples.epe_square != 0.0) {
        csv << samples.dot / std::sqrt(samples.pe_square * samples.epe_square);
    }
    csv << '\n';
    if (id == -1) {
        std::cout << std::left << std::setw(10) << stage << std::setw(15) << region << std::right << std::setw(11) << count
                  << std::scientific << std::setprecision(4) << std::setw(15) << pe.rms << std::setw(15) << epe.rms;
        if (pe.rms != 0.0) std::cout << std::setw(15) << epe.rms / pe.rms;
        else std::cout << std::setw(15) << "undefined";
        std::cout << std::setw(15) << pe.max_abs << std::setw(15) << epe.max_abs << '\n';
    }
}

// 汇总完整 Core+Halo、更新窗口和边界带；所有区间均为局部像素 [y0,y1)×[x0,x1)。
// Boundary 位于 Core+Guard 内且 |phi|<=一个物理像素宽度；重叠区域按各块独立驱动计数。
void collect_stats(const PartitionMeta& meta, const std::vector<Eigen::MatrixXd>& phis,
                   const MicroGradientExchangeResult& pe, const MicroGradientExchangeResult& epe,
                   double band_nm, double pe_weight, double epe_weight, std::ofstream& csv) {
    const std::vector<std::string> names = {"CoreHalo", "CoreGuard", "Boundary"};
    for (bool fused : {false, true}) {
        const std::string stage = fused ? "fused" : "raw";
        std::vector<GradientSamples> totals(3);
        for (const auto& tile : meta.tiles) {
            const auto& p = fused ? pe.locals[tile.id].mask_gradient : pe.locals[tile.id].raw_mask_gradient;
            const auto& e = fused ? epe.locals[tile.id].mask_gradient : epe.locals[tile.id].raw_mask_gradient;
            const auto& update = tile.update_local;
            std::vector<GradientSamples> local(3);
            for (int y = 0; y < p.rows(); ++y) {
                for (int x = 0; x < p.cols(); ++x) {
                    const bool in_update = y >= update.y0 && y < update.y1 && x >= update.x0 && x < update.x1;
                    const bool boundary = in_update && std::abs(phis[tile.id](y, x)) <= band_nm;
                    for (int region = 0; region < 3; ++region) {
                        if ((region == 1 && !in_update) || (region == 2 && !boundary)) continue;
                        local[region].add(p(y, x), e(y, x));
                        totals[region].add(p(y, x), e(y, x));
                    }
                }
            }
            for (int region = 0; region < 3; ++region) {
                write_stats(csv, stage, names[region], tile.id, local[region], pe_weight, epe_weight);
            }
        }
        for (int region = 0; region < 3; ++region) write_stats(csv, stage, names[region], -1, totals[region], pe_weight, epe_weight);
    }
}

} // namespace

// 只评价 LSM 的第 0 轮，不更新 phi；复用实际分区、SDF、灰度渲染和原 PE/EPE 光学反传。
int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--help") {
        std::cout << "Usage: demo_full_chip_gradient_stats [config.yaml] [output_dir]\n";
        return 0;
    }
    const auto project = fs::path(__FILE__).parent_path().parent_path();
    const auto config_path = fs::absolute(argc > 1 ? fs::path(argv[1]) : project / "config_full_chip_lsm.yaml");
    const auto root = YAML::LoadFile(config_path.string());
    const auto full = root["full_chip_lsm"];
    const auto params = SimulationParameters::from_yaml(config_path.string());
    const auto target_path = config_path.parent_path() / full["target_path"].as<std::string>();
    const auto mask_path = config_path.parent_path() / full["initial_mask_path"].as<std::string>();
    const auto output = fs::absolute(argc > 2 ? fs::path(argv[2]) : config_path.parent_path() /
                                    full["output_dir"].as<std::string>() / "gradient_stats_initial");
    const auto input = cv::imread(target_path.string(), cv::IMREAD_GRAYSCALE);
    Eigen::MatrixXd target(input.rows, input.cols), binary;
    for (int y = 0; y < input.rows; ++y) {
        for (int x = 0; x < input.cols; ++x) target(y, x) = input.at<unsigned char>(y, x) / 255.0;
    }
    SaveTxt::load_txt(mask_path.string(), binary);
    const auto meta = Partition::build(target.rows(), target.cols(), full["grid_rows"].as<int>(), full["grid_cols"].as<int>(),
                                       full["halo"].as<int>(), full["guard"].as<int>());
    const std::map<std::string, LevelSetMaskType> mask_types = {{"binary", LevelSetMaskType::Binary}, {"gray", LevelSetMaskType::Gray}};
    const auto mask_type = mask_types.at(full["levelset_update_mask_type"].as<std::string>());
    const int scale = full["levelset_phi_renderer_scale"].as<int>();
    const double pixel_nm = params.system.pixel_size_nm;
    std::vector<Eigen::MatrixXd> phis(meta.tiles.size()), masks(meta.tiles.size());
    for (const auto& tile : meta.tiles) {
        const auto local = Partition::extract_local_tile(binary, tile);
        phis[tile.id] = LevelSetUtils::reinit_sdf((local.array() - 0.5).matrix(), pixel_nm, pixel_nm);
        masks[tile.id] = render_levelset_mask(phis[tile.id], mask_type, scale);
    }
    const Grid grid(meta.tile_size, pixel_nm, params.optics.na, params.system.wavelength_nm);
    const auto& coords = grid.grid_coords();
    const Pupil pupil(LithographySimulator::_make_pupil_params(params), coords.Fx_2d, coords.Fy_2d);
    const Source source(LithographySimulator::_make_source_params(params, coords.Fx_1d));
    const LithoPrepare prepare(grid, pupil, source, true);
    MicroGradientExchange exchange(meta, prepare.cache(), params.resist.threshold, params.resist.alpha);
    MicroGradientOptions options;
    options.loss_region = MicroLossRegion::CoreGuard;
    options.main_region_dilation_pixels = full["main_region_dilation_pixels"].as<int>();
    options.epe_sampling = full["epe_sampling"].as<std::string>() == "contact" ? MicroEpeSampling::Contact : MicroEpeSampling::Metal;
    options.epe_all_eps = full["epe_all_eps"].as<bool>();
    options.epe_interval_line = full["epe_interval_line"].as<int>();
    options.epe_interval_corner = full["epe_interval_corner"].as<int>();
    options.epe_pixel_size = pixel_nm;
    options.enable_grad_fusion = full["enable_grad_fusion"].as<bool>();
    options.fusion_mode = full["fusion_mode"].as<std::string>() == "Linear" ? GradientFusionMode::Linear : GradientFusionMode::Average;
    options.fusion_weight_owner = full["fusion_weight_owner"].as<double>();
    options.fusion_weight_neighbor = full["fusion_weight_neighbor"].as<double>();
    const bool parallel = full["parallel_micro"].as<bool>();
    // 分别提取混合模式的背景 PE 和主图 EPE，不能用全区域纯 PE 代替混合模式的背景分项。
    options.loss_type = MicroLossType::PE_EPE;
    options.pe_weight = 1.0;
    options.epe_weight = 0.0;
    const auto pe = exchange.compute(masks, target, options, parallel);
    options.pe_weight = 0.0;
    options.epe_weight = 1.0;
    const auto epe = exchange.compute(masks, target, options, parallel);
    const double pe_weight = full["pe_weight"].as<double>(), epe_weight = full["epe_weight"].as<double>();
    fs::create_directories(output);
    std::ofstream csv(output / "gradient_stats.csv");
    csv << std::setprecision(17) << "stage,region,micro_id,count,pe_rms,epe_rms,pe_mean_abs,epe_mean_abs,pe_p99_abs,epe_p99_abs,"
        << "pe_max_abs,epe_max_abs,epe_over_pe_rms,mixed_pe_rms,mixed_epe_rms,mixed_epe_over_pe_rms,cosine\n";
    std::cout << "\nLSM initial state: " << meta.tiles.size() << " Micro, local grid=" << meta.tile_size << 'x' << meta.tile_size
              << ", boundary |phi| <= " << pixel_nm << " nm\n"
              << "PE_EPE components: background PE and main-pattern EPE gradients; no penalty/CEPT or normalization.\n"
              << "Configured mode=" << full["loss_type"].as<std::string>() << "; candidate mixed weights="
              << pe_weight << '/' << epe_weight << " (candidate values do not change optimization).\n\n"
              << std::left << std::setw(10) << "Stage" << std::setw(15) << "Region" << std::right << std::setw(11) << "Count"
              << std::setw(15) << "PE-RMS" << std::setw(15) << "EPE-RMS" << std::setw(15) << "EPE/PE"
              << std::setw(15) << "PE-MaxAbs" << std::setw(15) << "EPE-MaxAbs" << '\n';
    collect_stats(meta, phis, pe, epe, pixel_nm, pe_weight, epe_weight, csv);
    // 核验拆项后与原混合评价一致：固定 Guard 融合是线性操作，可分别融合后再加权。
    options.loss_type = MicroLossType::PE_EPE;
    options.pe_weight = pe_weight;
    options.epe_weight = epe_weight;
    const auto mixed = exchange.compute(masks, target, options, parallel);
    double raw_error = 0.0, fused_error = 0.0, reference_max = 0.0;
    for (const auto& tile : meta.tiles) {
        const int id = tile.id;
        const Eigen::MatrixXd raw = pe_weight * pe.locals[id].raw_mask_gradient + epe_weight * epe.locals[id].raw_mask_gradient;
        const Eigen::MatrixXd fused = pe_weight * pe.locals[id].mask_gradient + epe_weight * epe.locals[id].mask_gradient;
        raw_error = std::max(raw_error, (raw - mixed.locals[id].raw_mask_gradient).cwiseAbs().maxCoeff());
        fused_error = std::max(fused_error, (fused - mixed.locals[id].mask_gradient).cwiseAbs().maxCoeff());
        reference_max = std::max(reference_max, fused.cwiseAbs().maxCoeff());
    }
    std::ofstream check(output / "linearity_check.csv");
    check << std::setprecision(17) << "raw_max_abs_error,fused_max_abs_error,reference_max_abs\n"
          << raw_error << ',' << fused_error << ',' << reference_max << '\n';
    // 快照中的输入/结果路径转为绝对路径，便于从诊断目录再次运行；不修改项目原 YAML。
    auto saved_config = YAML::Clone(root);
    saved_config["full_chip_lsm"]["target_path"] = fs::absolute(target_path).string();
    saved_config["full_chip_lsm"]["initial_mask_path"] = fs::absolute(mask_path).string();
    saved_config["full_chip_lsm"]["output_dir"] = fs::absolute(config_path.parent_path() / full["output_dir"].as<std::string>()).string();
    std::ofstream(output / "config_used.yaml") << saved_config;
    std::cout << "\nMixed-gradient check: raw max error=" << raw_error << ", fused max error=" << fused_error
              << "\nSaved to " << output << std::endl;
}
