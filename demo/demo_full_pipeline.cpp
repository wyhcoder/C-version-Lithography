#include "CTM_Optimizer.h"
#include "LSM_Optimizer.h"
#include "MEEF_Optimizer.h"
#include "SRAF_Optimizer.h"
#include "imbinarize.h"
#include "imaging.h"
#include "litho_prepare.h"
#include "lithography_simulator.h"
#include "save_txt.h"
#include "simulation_parameters.h"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <yaml-cpp/yaml.h>

namespace {

std::filesystem::path project_path(const std::filesystem::path& project_root, const std::string& configured_path) {
    const std::filesystem::path path(configured_path);
    return (path.is_absolute() ? path : project_root / path).lexically_normal();
}

litho::MEEFPipelineConfig load_meef_config(const YAML::Node& node, const std::filesystem::path& project_root) {
    litho::MEEFPipelineConfig config;
    config.pattern_name = node["pattern_name"].as<std::string>();
    config.ls_mask_path = project_path(project_root, node["lsm_mask_path"].as<std::string>()).string();
    config.save_file_path = project_path(project_root, node["output_dir"].as<std::string>()).string();
    config.epe_mode = node["epe_mode"] ? node["epe_mode"].as<std::string>() : "legacy";
    config.meef_builder = node["meef_builder"].as<std::string>();
    config.meef_matrix_update_mode = node["meef_matrix_update_mode"].as<std::string>();
    config.meef_rebuild_interval = node["meef_rebuild_interval"].as<int>();
    config.epe_histogram_bin_width_nm = node["epe_histogram_bin_width_nm"].as<double>();
    config.move_strategy = node["move_strategy"].as<std::string>();
    config.iter = node["iter"].as<int>();
    config.stop_mode = node["stop_mode"].as<std::string>();
    config.step_tol = node["step_tol"].as<double>();
    config.patience = node["patience"].as<int>();
    config.main_cp_mode = node["main_cp_mode"].as<std::string>();
    config.main_cp_interval = node["main_cp_interval"].as<int>();
    config.main_symmetry = node["main_symmetry"].as<std::string>();
    config.sraf_mask_mode = node["sraf_mask_mode"].as<std::string>();
    config.fitted_sraf_txt_path = project_path(project_root, node["fitted_sraf_txt_path"].as<std::string>()).string();
    config.sraf_cp_interval = node["sraf_cp_interval"].as<int>();
    config.sraf_min_cps = node["sraf_min_cps"].as<int>();
    config.sraf_min_aera = node["sraf_min_aera"].as<int>();
    config.msaa_level = node["msaa_level"].as<int>();
    config.rasterizer = node["rasterizer"].as<std::string>();
    config.curve_type = node["curve_type"].as<std::string>();
    config.sraf_curve_type = node["sraf_curve_type"].as<std::string>();
    config.delta = node["delta"].as<double>();
    config.dilate_radius = node["dilate_radius"].as<int>();
    config.interval_line = node["interval_line"].as<double>();
    config.interval_corner = node["interval_corner"].as<double>();
    config.mid_weight = node["mid_weight"].as<double>();
    config.other_weight = node["other_weight"].as<double>();
    config.wepe_all_eps = node["wepe_all_eps"].as<bool>();
    config.optimize_wepe_only = node["optimize_wepe_only"].as<bool>();
    return config;
}

litho::SRAFConfig load_sraf_config(const YAML::Node& node, const std::filesystem::path& project_root, const std::string& pattern_name) {
    litho::SRAFConfig config;
    config.pattern_name = pattern_name;

    const YAML::Node paths = node["paths"];
    config.ls_mask_path = project_path(project_root, paths["lsm_mask_path"].as<std::string>()).string();
    config.main_cps_path = project_path(project_root, paths["main_control_points_path"].as<std::string>()).string();
    config.save_file_path = project_path(project_root, paths["output_dir"].as<std::string>()).string();

    const YAML::Node geometry = node["geometry"];
    config.foreground_threshold = geometry["foreground_threshold"].as<double>();
    config.target_threshold = geometry["target_threshold"].as<double>();
    config.overlap_ratio = geometry["overlap_ratio"].as<double>();
    config.fallback_dilate_radius = geometry["fallback_dilate_radius"].as<int>();
    config.opening_radius = geometry["opening_radius"].as<int>();
    config.minimum_component_area = geometry["minimum_component_area"].as<int>();
    config.control_point_interval = geometry["control_point_interval"].as<int>();

    const YAML::Node rendering = node["rendering"];
    config.curve_type = rendering["curve_type"].as<std::string>();
    config.initial_half_width = rendering["initial_half_width"].as<double>();
    config.maximum_half_width = rendering["maximum_half_width"].as<double>();
    config.samples_per_axis = rendering["samples_per_axis"].as<int>();
    config.msaa_level = rendering["main_mask_msaa_level"].as<int>();

    const YAML::Node epe = node["evaluation"]["epe"];
    config.interval_line = epe["interval_line"].as<int>();
    config.interval_corner = epe["interval_corner"].as<int>();
    config.mid_weight = epe["mid_weight"].as<double>();
    config.other_weight = epe["other_weight"].as<double>();

    const YAML::Node pv_band = node["evaluation"]["pv_band"];
    config.dose_margin = pv_band["dose_margin"].as<double>();
    config.defocus_range = pv_band["defocus_range"].as<double>();
    config.defocus_step = pv_band["defocus_step"].as<int>();
    config.mode = pv_band["mode"].as<std::string>();

    const YAML::Node width = node["width_optimization"];
    config.independent_sraf_widths = width["mode"].as<std::string>() == "independent";
    config.shared_width_min = width["shared"]["min_half_width"].as<double>();
    config.shared_width_max = width["shared"]["max_half_width"].as<double>();
    config.shared_width_coarse_step = width["shared"]["coarse_step"].as<double>();
    config.shared_width_fine_step = width["shared"]["fine_step"].as<double>();
    config.independent_width_min = width["independent"]["min_half_width"].as<double>();
    config.independent_width_max = width["independent"]["max_half_width"].as<double>();
    config.cma_initial_sigma = width["independent"]["cma_es"]["initial_sigma"].as<double>();
    config.cma_population_size = width["independent"]["cma_es"]["population_size"].as<int>();
    config.cma_max_evaluations = width["independent"]["cma_es"]["max_evaluations"].as<int>();
    config.cma_tolerance_x = width["independent"]["cma_es"]["tolerance_x"].as<double>();
    config.cma_tolerance_fun = width["independent"]["cma_es"]["tolerance_fun"].as<double>();
    config.cma_seed = width["independent"]["cma_es"]["seed"].as<unsigned int>();

    const YAML::Node objective = node["objective"];
    config.weight_pvband = objective["pv_band_weight"].as<double>();
    config.weight_wepe = objective["weighted_epe_weight"].as<double>();
    config.weight_pe = objective["pixel_error_weight"].as<double>();
    return config;
}

double seconds_since(const std::chrono::steady_clock::time_point& start) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

}  // namespace

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    using namespace litho;

    const fs::path build_dir = fs::current_path();
    const fs::path project_root = build_dir.parent_path();
    const fs::path config_path = argc > 1 ? fs::absolute(argv[1]).lexically_normal() : build_dir / "config.yaml";
    const YAML::Node root = YAML::LoadFile(config_path.string());
    const SimulationParameters params = SimulationParameters::from_yaml(config_path.string());

    std::cout << "========== CTM + LSM + MEEF + SRAF full pipeline ==========" << '\n'
              << "config  : " << config_path << '\n'
              << "pattern : " << params.mask.image_name << "\n\n";

    const auto total_start = std::chrono::steady_clock::now();
    LithographySimulator simulator(params);
    LithoPrepare prepare(simulator._grid, simulator._pupil, simulator._source, true);
    ImagingCache cache = prepare.cache();
    Imaging imaging(cache);

    const fs::path result_root = SaveTxt::_result_path();
    fs::create_directories(result_root / "System_result");
    fs::create_directories(result_root / "CTM_result");
    fs::create_directories(result_root / "LSM_result");
    SaveTxt::save_mat_int(simulator._mask.data(), "System_result/target_mask.txt");
    SaveTxt::save_mat(simulator._pupil.get_amplitude(), "System_result/pupil.txt");
    SaveTxt::save_mat(simulator._source.get_source_map(), "System_result/source.txt");

    std::cout << "[1/4] CTM optimization\n";
    const auto ctm_start = std::chrono::steady_clock::now();
    const YAML::Node ctm_yaml = root["ctm"];
    CTM_Optimizer ctm_optimizer(simulator, cache, ctm_yaml["max_iteration"].as<int>(), ctm_yaml["learning_rate"].as<double>(),
                                ctm_yaml["optimizer"].as<std::string>(), ctm_yaml["lbfgs_history_size"].as<int>(),
                                ctm_yaml["gradient_tolerance"].as<double>());
    Eigen::MatrixXd ctm_gray_mask = ctm_optimizer.optimize();
    Imaging_Result ctm_gray_imaging = imaging.compute(ctm_gray_mask, params.resist.threshold, params.resist.alpha);
    BinarizeResult ctm_binary = Imbinarize::binarize_advanced(ctm_gray_mask, simulator._mask.data(), 0.1, 0.15);
    Imaging_Result ctm_binary_imaging = imaging.compute(ctm_binary.final_mask, params.resist.threshold, params.resist.alpha);
    SaveTxt::save_mat(ctm_gray_mask, "CTM_result/gray_mask.txt");
    SaveTxt::save_mat(ctm_gray_imaging.aerial_image, "CTM_result/aerial_image_G.txt");
    SaveTxt::save_mat(ctm_gray_imaging.wafer_image, "CTM_result/wafer_image_G.txt");
    SaveTxt::save_mat_int(ctm_binary.final_mask, "CTM_result/binary_mask.txt");
    SaveTxt::save_mat(ctm_binary_imaging.aerial_image, "CTM_result/aerial_image_B.txt");
    SaveTxt::save_mat(ctm_binary_imaging.wafer_image, "CTM_result/wafer_image_B.txt");
    std::cout << "CTM completed in " << seconds_since(ctm_start) << " s\n\n";

    std::cout << "[2/4] Level-set optimization\n";
    const auto lsm_start = std::chrono::steady_clock::now();
    LSM_Optimizer lsm_optimizer(simulator, cache, params.system.pixel_size_nm, params.system.pixel_size_nm, 0.5, 0.01, 200);
    lsm_optimizer.set_penalty(0.0, 0.0);
    Level_set_result lsm_result = lsm_optimizer.optimize(ctm_binary.final_mask);
    Imaging_Result lsm_imaging = imaging.compute(lsm_result.mask, params.resist.threshold, params.resist.alpha);
    SaveTxt::save_mat(lsm_result.mask, "LSM_result/LSM_mask.txt");
    SaveTxt::save_mat(lsm_imaging.aerial_image, "LSM_result/LSM_aerial.txt");
    SaveTxt::save_mat(lsm_imaging.wafer_image, "LSM_result/LSM_wafer.txt");
    SaveTxt::save_mat(lsm_result.initial_sdf, "LSM_result/LSM_initial_sdf.txt");
    SaveTxt::save_mat(lsm_result.sdf, "LSM_result/LSM_sdf.txt");
    SaveTxt::save_mat(lsm_result.mask - ctm_binary.final_mask, "LSM_result/LSM_diff.txt");
    std::cout << "LSM completed in " << seconds_since(lsm_start) << " s\n\n";

    std::cout << "[3/4] MEEF control-point optimization\n";
    const auto meef_start = std::chrono::steady_clock::now();
    const YAML::Node meef_yaml = root["meef"];
    const MEEFPipelineConfig meef_config = load_meef_config(meef_yaml, project_root);
    fs::create_directories(meef_config.save_file_path);
    fs::copy_file(config_path, fs::path(meef_config.save_file_path) / meef_yaml["config_snapshot_name"].as<std::string>(),
                  fs::copy_options::overwrite_existing);
    MEEF_Optimizer meef_optimizer(simulator, cache, meef_config);
    meef_optimizer.optimize();
    std::cout << "MEEF completed in " << seconds_since(meef_start) << " s\n\n";

    std::cout << "[4/4] SRAF width optimization\n";
    const auto sraf_start = std::chrono::steady_clock::now();
    const SRAFConfig sraf_config = load_sraf_config(root["sraf_optimizer"], project_root, params.mask.image_name);
    SRAF_Optimizer sraf_optimizer(simulator, cache, sraf_config);
    sraf_optimizer.optimize();
    std::cout << "SRAF width optimization completed in " << seconds_since(sraf_start) << " s\n\n"
              << "Pipeline completed in " << seconds_since(total_start) << " s\n"
              << "  CTM  : " << result_root / "CTM_result" << '\n'
              << "  LSM  : " << result_root / "LSM_result" << '\n'
              << "  MEEF : " << meef_config.save_file_path << '\n'
              << "  SRAF : " << sraf_config.save_file_path << '\n';
    return 0;
}
