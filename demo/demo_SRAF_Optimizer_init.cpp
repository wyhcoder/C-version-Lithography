#include "SRAF_Optimizer.h"
#include "litho_prepare.h"
#include "lithography_simulator.h"
#include "simulation_parameters.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    namespace fs = std::filesystem;
    using namespace litho;

    try {
        // 当前 demo 从 build-release 目录直接运行，不读取命令行参数。
        const fs::path project_root = "..";
        const fs::path config_path = project_root / "config.yaml";
        const fs::path lsm_mask_path =
            project_root / "result/LSM_result/LSM_mask.txt";
        const fs::path main_control_points_path =
            project_root /
            "result/MEEF_result/test/best_wepe/control_points.txt";
        const fs::path output_dir =
            project_root / "outputs/sraf_optimizer_init";

        if (!fs::is_regular_file(config_path)) {
            throw std::runtime_error(
                "config file does not exist: " + config_path.string());
        }
        if (!fs::is_regular_file(lsm_mask_path)) {
            throw std::runtime_error(
                "LSM mask does not exist: " + lsm_mask_path.string());
        }
        if (!fs::is_regular_file(main_control_points_path)) {
            throw std::runtime_error(
                "main control-points file does not exist: " +
                main_control_points_path.string());
        }

        SimulationParameters params =
            SimulationParameters::from_yaml(config_path.string());
        LithographySimulator simulator(params);
        LithoPrepare prepare(
            simulator._grid,
            simulator._pupil,
            simulator._source,
            true);

        SRAFConfig sraf_config;
        sraf_config.pattern_name = params.mask.image_name;
        sraf_config.ls_mask_path = lsm_mask_path.string();
        sraf_config.main_cps_path = main_control_points_path.string();
        sraf_config.save_file_path = output_dir.string();
        sraf_config.curve_type = "BS";
        sraf_config.control_point_interval = 10;
        sraf_config.minimum_component_area = 3;
        sraf_config.opening_radius = 0;
        sraf_config.initial_half_width = 4.0;
        sraf_config.maximum_half_width = 8.0;
        sraf_config.samples_per_axis = 4;
        // false：全部 SRAF 共用一个宽度变量；true：每根独立优化。
        sraf_config.independent_sraf_widths = false;

        SRAF_Optimizer optimizer(simulator, prepare.cache(), sraf_config);
        std::cout << "\nTesting initial-mask PV Band...\n";
        optimizer.optimize();
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
                  << "  combined mask    : "
                  << (output_dir / "initial_combined_mask.png") << '\n'
                  << "  output directory : " << output_dir << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "demo_SRAF_Optimizer_init failed: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
