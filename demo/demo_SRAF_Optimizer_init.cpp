#include "SRAF_Optimizer.h"
#include "litho_prepare.h"
#include "lithography_simulator.h"
#include "simulation_parameters.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <yaml-cpp/yaml.h>

namespace {

std::filesystem::path config_relative_path(
    const std::filesystem::path& project_root,
    const std::string& configured_path) {
    namespace fs = std::filesystem;
    const fs::path path(configured_path);
    return (path.is_absolute() ? path : project_root / path)
        .lexically_normal();
}

// 将 config.yaml 的 sraf_optimizer 节完整转换为优化器配置。
litho::SRAFConfig load_sraf_config(
    const YAML::Node& sraf_yaml,
    const std::filesystem::path& project_root,
    const std::string& pattern_name) {
    litho::SRAFConfig sraf_config;
    sraf_config.pattern_name = pattern_name;

    const YAML::Node& paths = sraf_yaml["paths"];
    sraf_config.ls_mask_path = config_relative_path(
        project_root,
        paths["lsm_mask_path"].as<std::string>()).string();
    sraf_config.main_cps_path = config_relative_path(
        project_root,
        paths["main_control_points_path"].as<std::string>()).string();
    sraf_config.save_file_path = config_relative_path(
        project_root,
        paths["output_dir"].as<std::string>()).string();

    const YAML::Node& geometry = sraf_yaml["geometry"];
    sraf_config.foreground_threshold =
        geometry["foreground_threshold"].as<double>();
    sraf_config.target_threshold =
        geometry["target_threshold"].as<double>();
    sraf_config.overlap_ratio = geometry["overlap_ratio"].as<double>();
    sraf_config.fallback_dilate_radius =
        geometry["fallback_dilate_radius"].as<int>();
    sraf_config.opening_radius = geometry["opening_radius"].as<int>();
    sraf_config.minimum_component_area =
        geometry["minimum_component_area"].as<int>();
    sraf_config.control_point_interval =
        geometry["control_point_interval"].as<int>();

    const YAML::Node& rendering = sraf_yaml["rendering"];
    sraf_config.curve_type = rendering["curve_type"].as<std::string>();
    sraf_config.initial_half_width =
        rendering["initial_half_width"].as<double>();
    sraf_config.maximum_half_width =
        rendering["maximum_half_width"].as<double>();
    sraf_config.samples_per_axis =
        rendering["samples_per_axis"].as<int>();
    sraf_config.msaa_level =
        rendering["main_mask_msaa_level"].as<int>();

    const YAML::Node& evaluation = sraf_yaml["evaluation"];
    const YAML::Node& epe = evaluation["epe"];
    sraf_config.interval_line = epe["interval_line"].as<int>();
    sraf_config.interval_corner = epe["interval_corner"].as<int>();
    sraf_config.mid_weight = epe["mid_weight"].as<double>();
    sraf_config.other_weight = epe["other_weight"].as<double>();

    const YAML::Node& pv_band = evaluation["pv_band"];
    sraf_config.dose_margin = pv_band["dose_margin"].as<double>();
    sraf_config.defocus_range = pv_band["defocus_range"].as<double>();
    sraf_config.defocus_step = pv_band["defocus_step"].as<int>();
    sraf_config.mode = pv_band["mode"].as<std::string>();

    const YAML::Node& width = sraf_yaml["width_optimization"];
    const std::string width_mode = width["mode"].as<std::string>();
    if (width_mode != "shared" && width_mode != "independent") {
        throw std::invalid_argument(
            "sraf_optimizer.width_optimization.mode must be shared or "
            "independent");
    }
    sraf_config.independent_sraf_widths = (width_mode == "independent");

    const YAML::Node& shared = width["shared"];
    sraf_config.shared_width_min =
        shared["min_half_width"].as<double>();
    sraf_config.shared_width_max =
        shared["max_half_width"].as<double>();
    sraf_config.shared_width_coarse_step =
        shared["coarse_step"].as<double>();
    sraf_config.shared_width_fine_step =
        shared["fine_step"].as<double>();

    const YAML::Node& independent = width["independent"];
    sraf_config.independent_width_min =
        independent["min_half_width"].as<double>();
    sraf_config.independent_width_max =
        independent["max_half_width"].as<double>();

    const YAML::Node& cma_es = independent["cma_es"];
    sraf_config.cma_initial_sigma =
        cma_es["initial_sigma"].as<double>();
    sraf_config.cma_population_size =
        cma_es["population_size"].as<int>();
    sraf_config.cma_max_evaluations =
        cma_es["max_evaluations"].as<int>();
    sraf_config.cma_tolerance_x =
        cma_es["tolerance_x"].as<double>();
    sraf_config.cma_tolerance_fun =
        cma_es["tolerance_fun"].as<double>();
    sraf_config.cma_seed = cma_es["seed"].as<unsigned int>();

    const YAML::Node& objective = sraf_yaml["objective"];
    sraf_config.weight_pvband =
        objective["pv_band_weight"].as<double>();
    sraf_config.weight_wepe =
        objective["weighted_epe_weight"].as<double>();
    sraf_config.weight_pe =
        objective["pixel_error_weight"].as<double>();

    return sraf_config;
}

// 使用单引号保护传给 shell 的路径，兼容路径中包含空格或单引号的情况。
std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for (const char character : value) {
        if (character == '\'') {
            quoted += "'\\''";
        } else {
            quoted += character;
        }
    }
    quoted += "'";
    return quoted;
}

// 独立宽度 CMA-ES 完成后，将历史 CSV 绘制为候选损失和最优损失曲线。
void generate_cma_es_loss_curve(
    const std::filesystem::path& project_root,
    const std::filesystem::path& output_dir) {
    namespace fs = std::filesystem;
    const fs::path absolute_root = fs::absolute(project_root).lexically_normal();
    const fs::path script_path =
        absolute_root / "scripts/plot_cma_es_loss.py";
    const fs::path history_path =
        fs::absolute(output_dir / "independent_width_history.csv")
            .lexically_normal();
    const fs::path figure_path =
        fs::absolute(output_dir / "cma_es_loss_curve.png").lexically_normal();
    const fs::path venv_python = absolute_root / ".venv/bin/python";
    const std::string python =
        fs::is_regular_file(venv_python) ? venv_python.string() : "python3";

    if (!fs::is_regular_file(history_path)) {
        std::cerr << "Warning: CMA-ES history does not exist; loss curve was "
                  << "not generated: " << history_path << '\n';
        return;
    }
    if (!fs::is_regular_file(script_path)) {
        std::cerr << "Warning: loss-curve script does not exist: "
                  << script_path << '\n';
        return;
    }

    const std::string command =
        shell_quote(python) + " " + shell_quote(script_path.string()) +
        " --input " + shell_quote(history_path.string()) +
        " --output " + shell_quote(figure_path.string());
    const int status = std::system(command.c_str());
    if (status != 0) {
        std::cerr << "Warning: failed to generate CMA-ES loss curve; command "
                  << "exit status=" << status << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    using namespace litho;

    try {
        if (argc > 2) {
            throw std::invalid_argument(
                "usage: ./demo_SRAF_Optimizer_init [config.yaml]");
        }

        // 按项目约定从 build-* 目录运行；相对输入/输出路径以项目根目录为准。
        const fs::path build_dir = fs::current_path();
        const fs::path project_root =
            fs::absolute(build_dir.parent_path()).lexically_normal();
        const fs::path config_path = (argc == 2)
            ? fs::absolute(fs::path(argv[1])).lexically_normal()
            : project_root / "config.yaml";

        if (!fs::is_regular_file(config_path)) {
            throw std::runtime_error(
                "config file does not exist: " + config_path.string());
        }

        const YAML::Node root = YAML::LoadFile(config_path.string());
        const SimulationParameters params =
            SimulationParameters::from_yaml(config_path.string());
        const YAML::Node sraf_yaml = root["sraf_optimizer"];
        if (!sraf_yaml || !sraf_yaml.IsMap()) {
            throw std::runtime_error(
                "config.yaml missing sraf_optimizer section");
        }
        const SRAFConfig sraf_config = load_sraf_config(
            sraf_yaml, project_root, params.mask.image_name);
        const fs::path lsm_mask_path(sraf_config.ls_mask_path);
        const fs::path main_control_points_path(sraf_config.main_cps_path);
        const fs::path output_dir(sraf_config.save_file_path);

        if (!fs::is_regular_file(lsm_mask_path)) {
            throw std::runtime_error(
                "LSM mask does not exist: " + lsm_mask_path.string());
        }
        if (!fs::is_regular_file(main_control_points_path)) {
            throw std::runtime_error(
                "main control-points file does not exist: " +
                main_control_points_path.string());
        }

        LithographySimulator simulator(params);
        LithoPrepare prepare(
            simulator._grid,
            simulator._pupil,
            simulator._source,
            true);

        SRAF_Optimizer optimizer(simulator, prepare.cache(), sraf_config);
        std::cout << "\nTesting initial-mask PV Band...\n";
        optimizer.optimize();
        if (sraf_config.independent_sraf_widths) {
            generate_cma_es_loss_curve(project_root, output_dir);
        }
        std::cout << "\nSRAF optimizer initialization completed\n"
                  << "  components       : "
                  << optimizer.geometry_result().centerlines.size() << '\n'
                  << "  main contours    : "
                  << optimizer.main_control_points().size() << '\n'
                  << "  main control pts : "
                  << optimizer.main_control_point_count() << '\n'
                  << "  SRAF control pts : "
                  << optimizer.control_point_count() << '\n'
                  << "  width variables  : "
                  << optimizer.initial_half_widths().size() << '\n'
                  << "  optimized mask   : "
                  << (sraf_config.independent_sraf_widths
                          ? output_dir / "optimized_independent_mask.png"
                          : output_dir / "optimized_shared_mask.png")
                  << '\n'
                  << "  loss curve       : "
                  << (sraf_config.independent_sraf_widths
                          ? output_dir / "cma_es_loss_curve.png"
                          : fs::path("not generated in shared-width mode"))
                  << '\n'
                  << "  output directory : " << output_dir << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "demo_SRAF_Optimizer_init failed: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
