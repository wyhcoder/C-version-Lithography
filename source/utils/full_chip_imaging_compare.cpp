#include "full_chip_imaging_compare.h"
#include "lithography_simulator.h"
#include "loss.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace litho {

FullChipImagingComparison compare_full_chip_imaging(const SimulationParameters& params, const Eigen::MatrixXd& target,
                                                    const Eigen::MatrixXd& optimized_mask, int source_grid_size) {
    FullChipImagingComparison result{};
    result.pixel_size_nm = params.system.pixel_size_nm;
    // 照明积分采样与全图 FFT 尺寸独立，沿用配置中的光源采样密度。
    const Grid source_grid(source_grid_size, params.system.pixel_size_nm, params.optics.na, params.system.wavelength_nm);
    result.source_grid_size = source_grid.size();
    const auto source_params = LithographySimulator::_make_source_params(params, source_grid.grid_coords().Fx_1d);

    // 按全图尺寸初始化一次光学模型；当前 1024 版图对应 1024×1024 的光瞳、频域核和 FFT 计划。
    const Grid grid(target.cols(), params.system.pixel_size_nm, params.optics.na, params.system.wavelength_nm);
    const auto& coords = grid.grid_coords();
    const Pupil pupil(LithographySimulator::_make_pupil_params(params), coords.Fx_2d, coords.Fy_2d);
    const Source source(source_params);
    std::cout << "Preparing full-frame optics: " << grid.size() << 'x' << grid.size() << std::endl;
    const LithoPrepare prepare(grid, pupil, source, true, true);
    Imaging imaging(prepare.cache());
    result.grid_size = grid.size();
    result.source_points = prepare.cache().source_ws.size();
    result.kernels = prepare.cache().socs_eigenvalues.size();

    // 两次成像仅更换输入 mask；CTM 灰度或 LSM 二值结果直接传入，光学和显影参数完全相同。
    result.target_mask = target;
    result.optimized_mask = optimized_mask;
    result.target_image = imaging.compute(target, params.resist.threshold, params.resist.alpha);
    result.optimized_image = imaging.compute(optimized_mask, params.resist.threshold, params.resist.alpha);
    result.target_pe = Loss::pe_loss(result.target_image.wafer_image, target);
    result.optimized_pe = Loss::pe_loss(result.optimized_image.wafer_image, target);
    std::cout << "Target PE=" << result.target_pe << ", optimized mask PE=" << result.optimized_pe << std::endl;
    return result;
}

namespace {

// TXT 保留 double 精度，供离线画图/误差计算；索引为全局 (y,x)，行列范围 [0,height) × [0,width)。
void save_matrix(const Eigen::MatrixXd& matrix, const std::filesystem::path& path) {
    std::ofstream file(path);
    file << "# rows=" << matrix.rows() << " cols=" << matrix.cols() << '\n' << std::setprecision(17);
    file << matrix << '\n';
}

} // namespace

void save_full_chip_imaging_comparison(const FullChipImagingComparison& result, const std::filesystem::path& directory,
                                     FullChipImagingStage stage) {
    std::filesystem::create_directories(directory);
    const std::string optimized_case = stage == FullChipImagingStage::LSM ? "optimized_lsm" : "optimized_gray";
    save_matrix(result.target_mask, directory / "target_mask.txt");
    save_matrix(result.optimized_mask, directory / (optimized_case + "_mask.txt"));
    save_matrix(result.target_image.aerial_image, directory / "target_ai.txt");
    save_matrix(result.target_image.wafer_image, directory / "target_wi.txt");
    save_matrix(result.optimized_image.aerial_image, directory / "optimized_ai.txt");
    save_matrix(result.optimized_image.wafer_image, directory / "optimized_wi.txt");
    std::ofstream metadata(directory / "sampling.csv");
    metadata << "grid_size,source_grid_size,source_points,kernels,pixel_size_nm\n";
    metadata << result.grid_size << ',' << result.source_grid_size << ',' << result.source_points << ','
             << result.kernels << ',' << result.pixel_size_nm << '\n';
    std::ofstream metrics(directory / "metrics.csv");
    metrics << "case,pe\n" << std::setprecision(17);
    metrics << "target," << result.target_pe << '\n' << optimized_case << ',' << result.optimized_pe << '\n';
}

} // namespace litho
