#include "full_chip_imaging_compare.h"
#include "lithography_simulator.h"
#include "loss.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace litho {

FullChipImagingComparison compare_micro_imaging(const SimulationParameters& params, const PartitionMeta& meta,
                                              const ImagingCache& cache, const Eigen::MatrixXd& target,
                                              const std::vector<Eigen::MatrixXd>& optimized_local_masks, bool parallel_micro) {
    FullChipImagingComparison result{};
    result.mode = FullChipImagingMode::MicroCore;
    result.grid_size = result.source_grid_size = cache.N;
    result.image_rows = meta.height;
    result.image_cols = meta.width;
    result.micro_count = static_cast<int>(meta.tiles.size());
    result.source_points = cache.source_ws.size();
    result.kernels = cache.optical_kernels_frequency.size();
    result.pixel_size_nm = params.system.pixel_size_nm;
    result.target_mask = target;
    result.optimized_mask = Partition::assemble_global(optimized_local_masks, meta);

    int worker_count = 1;
#ifdef _OPENMP
    if (parallel_micro) worker_count = std::min(result.micro_count, omp_get_max_threads());
#endif
    // FFTW 计划串行创建；每个线程独占 Imaging 的可变缓冲区，各线程共享只读光学 cache。
    std::vector<std::unique_ptr<Imaging>> workers;
    for (int i = 0; i < worker_count; ++i) workers.push_back(std::make_unique<Imaging>(cache));
    std::vector<Eigen::MatrixXd> target_ai(result.micro_count), target_wi(result.micro_count), optimized_wi(result.micro_count);
    result.optimized_local_aerial_images.resize(result.micro_count);
    std::cout << "Micro imaging: " << result.micro_count << " tiles, optics=" << cache.N << 'x' << cache.N
              << ", Core assembly=" << meta.height << 'x' << meta.width << std::endl;
#ifdef _OPENMP
#pragma omp parallel for if(parallel_micro) num_threads(worker_count) schedule(static)
#endif
    for (int id = 0; id < result.micro_count; ++id) {
        int worker_id = 0;
#ifdef _OPENMP
        worker_id = omp_get_thread_num();
#endif
        auto& imaging = *workers[worker_id];
        // target 的外围 Halo 按原分区规则周期提取；优化输入直接使用最佳轮局部状态，保留独立 Guard。
        const auto local_target = Partition::extract_local_tile(target, meta.tiles[id]);
        auto baseline = imaging.compute(local_target, params.resist.threshold, params.resist.alpha);
        auto optimized = imaging.compute(optimized_local_masks[id], params.resist.threshold, params.resist.alpha);
        target_ai[id] = std::move(baseline.aerial_image);
        target_wi[id] = std::move(baseline.wafer_image);
        result.optimized_local_aerial_images[id] = std::move(optimized.aerial_image);
        optimized_wi[id] = std::move(optimized.wafer_image);
    }
    // 成像结束后串行只拼 Core；Guard/Halo 提供局部成像上下文，不在全局结果中重复计入。
    result.target_image = {Partition::assemble_global(target_ai, meta), Partition::assemble_global(target_wi, meta)};
    result.optimized_image = {Partition::assemble_global(result.optimized_local_aerial_images, meta),
                              Partition::assemble_global(optimized_wi, meta)};
    result.target_pe = Loss::pe_loss(result.target_image.wafer_image, target);
    result.optimized_pe = Loss::pe_loss(result.optimized_image.wafer_image, target);
    std::cout << "Core-assembled PE: target=" << result.target_pe << ", optimized=" << result.optimized_pe << std::endl;
    return result;
}

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
    result.image_rows = target.rows();
    result.image_cols = target.cols();
    result.source_points = prepare.cache().source_ws.size();
    result.kernels = prepare.cache().socs_eigenvalues.size();

    // 两次成像仅更换输入 mask；CTM 灰度或 LSM Binary/Gray 结果直接传入，光学和显影参数完全相同。
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
    metadata << "grid_size,source_grid_size,source_points,kernels,pixel_size_nm,imaging_mode,image_rows,image_cols,micro_count\n";
    metadata << result.grid_size << ',' << result.source_grid_size << ',' << result.source_points << ','
             << result.kernels << ',' << result.pixel_size_nm << ','
             << (result.mode == FullChipImagingMode::MicroCore ? "micro_core" : "full_frame") << ','
             << result.image_rows << ',' << result.image_cols << ',' << result.micro_count << '\n';
    std::ofstream metrics(directory / "metrics.csv");
    metrics << "case,pe\n" << std::setprecision(17);
    metrics << "target," << result.target_pe << '\n' << optimized_case << ',' << result.optimized_pe << '\n';
}

} // namespace litho
