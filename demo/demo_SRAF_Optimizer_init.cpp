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
        // 当前版图名称，只用于输出信息和结果标识。
        sraf_config.pattern_name = params.mask.image_name;
        // LSM 生成的 mask 路径，用来提取 SRAF 区域及中心线。
        sraf_config.ls_mask_path = lsm_mask_path.string();
        // 已优化的主图形控制点文件，控制点坐标顺序为 y x。
        sraf_config.main_cps_path = main_control_points_path.string();
        // mask、PV Band 图和优化历史等结果的保存目录。
        sraf_config.save_file_path = output_dir.string();
        // 主图形曲线类型；BS 表示 B 样条曲线。
        sraf_config.curve_type = "BS";
        // 沿 SRAF 骨架每隔多少个像素选取一个控制点。
        sraf_config.control_point_interval = 10;
        // 删除像素面积小于该值的孤立 SRAF 连通区域。
        sraf_config.minimum_component_area = 3;
        // 形态学开运算半径；0 表示不进行开运算。
        sraf_config.opening_radius = 0;
        // 优化前每根 SRAF 的初始半宽，完整线宽为它的两倍，单位为像素。
        sraf_config.initial_half_width = 2.0;
        // 距离缓存支持的最大半宽，任何优化宽度都不能超过它。
        sraf_config.maximum_half_width = 5.0;
        // 每个像素轴向的子像素采样数；4 表示每个像素使用 4x4 个采样点。
        sraf_config.samples_per_axis = 4;

        // 每根 SRAF 使用一个独立半宽变量，并通过 CMA-ES 联合优化。
        // false 时所有 SRAF 共用一个半宽，并使用下面的粗细两级搜索参数。
        sraf_config.independent_sraf_widths = true;
        // 统一宽度模式的半宽搜索下界和上界，单位为像素。
        sraf_config.shared_width_min = 1.0;
        sraf_config.shared_width_max = 4.0;
        // 统一宽度模式先按 0.5 像素粗搜，再按 0.1 像素细搜。
        sraf_config.shared_width_coarse_step = 0.5;
        sraf_config.shared_width_fine_step = 0.1;

        // 独立宽度模式中，每根 SRAF 半宽的允许范围，单位为像素。
        sraf_config.independent_width_min = 1.0;
        sraf_config.independent_width_max = 4.0;
        // CMA-ES 初始搜索步长，越大表示初始探索的宽度变化越大。
        sraf_config.cma_initial_sigma = 0.375;
        // 每一代生成并评价的候选宽度向量数量。
        sraf_config.cma_population_size = 12;
        // 最大目标函数评价次数；程序按完整种群向下取整。
        sraf_config.cma_max_evaluations = 1200;
        // 参数变化小于该值时可以停止，单位与半宽相同（像素）。
        sraf_config.cma_tolerance_x = 1e-2;
        // 相邻搜索结果的 cost 改善小于该值时可以停止。
        sraf_config.cma_tolerance_fun = 1e-3;
        // CMA-ES 随机种子；固定为正数可以复现实验结果。
        sraf_config.cma_seed = 1;

        // 总 cost 中归一化 PV Band、加权 EPE 和 PE 的权重。
        // 三项通常设置为和等于 1；当前配置更关注降低 PV Band。
        sraf_config.weight_pvband = 0.9;
        sraf_config.weight_wepe = 0.05;
        sraf_config.weight_pe = 0.05;

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
                  << "  optimized mask   : "
                  << (sraf_config.independent_sraf_widths
                          ? output_dir / "optimized_independent_mask.png"
                          : output_dir / "optimized_shared_mask.png")
                  << '\n'
                  << "  output directory : " << output_dir << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "demo_SRAF_Optimizer_init failed: "
                  << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
