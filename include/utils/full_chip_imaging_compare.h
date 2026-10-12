#pragma once

#include "imaging.h"
#include "partition.h"
#include "simulation_parameters.h"

#include <filesystem>

namespace litho {

// 仅决定导出文件名和图注。
enum class FullChipImagingStage { CTM, LSM };
enum class FullChipImagingMode { FullFrame, MicroCore }; // 整图直接成像，或局部成像后只拼 Core。

struct FullChipImagingComparison {
    FullChipImagingMode mode = FullChipImagingMode::FullFrame;
    int grid_size; // 实际光学/FFT 网格边长，单位：像素；Micro 模式为含 Halo 的局部尺寸。
    int image_rows, image_cols, micro_count; // 导出矩阵的全局像素行/列数；整图直接成像时 micro_count=0。
    int source_grid_size, source_points; // 照明积分采样网格边长及非零光源点数。
    int kernels; // 此套光学缓存保留的相干核数。
    double pixel_size_nm;
    Eigen::MatrixXd target_mask; // 原始 target 版图，行列为全局像素索引。
    Eigen::MatrixXd optimized_mask; // 最佳轮回拼版图：CTM 为灰度，LSM 按 Binary/Gray 配置渲染；保留输入原值。
    Imaging_Result target_image; // 全局 AI/连续显影 WI；Micro 模式由各块 Core 回拼。
    Imaging_Result optimized_image; // 优化版图的全局 AI/WI，采用与 target 相同的成像方式。
    std::vector<Eigen::MatrixXd> optimized_local_aerial_images; // Micro 模式保存完整局部 AI，供 EP 空间差分使用。
    double target_pe, optimized_pe; // 各自的 Σ(WI-target)²，仅计算成像误差，不包含 CEPT。
};

// 复用外部已准备的 Micro 光学 cache（cache.N == meta.tile_size），不重新初始化全图光学系统。
// target 为全局矩阵；optimized_local_masks[id] 为最佳轮完整局部 mask，保留该块自己的 Guard/Halo。
// 各块在完整局部范围成像，再将 core_local 的 [y0,y1)×[x0,x1) 写入对应 core_global；无重叠平均。
// parallel_micro 控制块间并行；返回两组 Core 回拼的 AI/WI，以及未裁切的优化局部 AI。
FullChipImagingComparison compare_micro_imaging(const SimulationParameters& params, const PartitionMeta& meta,
                                              const ImagingCache& cache, const Eigen::MatrixXd& target,
                                              const std::vector<Eigen::MatrixXd>& optimized_local_masks,
                                              bool parallel_micro = true);

// target 与 optimized_mask 是相同尺寸的全局正方形矩阵，索引为全局像素 (y,x)。
// 按 target 的边长重建一套全图光学模型，分别对两个输入成像；mask 和显影 WI 均保留输入/计算原值。
// source_grid_size 仅控制照明积分采样；两次成像共享同一套光源、光瞳、SOCS 核和显影参数。
FullChipImagingComparison compare_full_chip_imaging(const SimulationParameters& params, const Eigen::MatrixXd& target,
                                                    const Eigen::MatrixXd& optimized_mask, int source_grid_size = 256);

// 保存 mask、AI/WI 的 double 矩阵、评价指标和采样参数；stage 区分 CTM 与 LSM 结果的文件名。
void save_full_chip_imaging_comparison(const FullChipImagingComparison& result, const std::filesystem::path& directory,
                                     FullChipImagingStage stage = FullChipImagingStage::CTM);

} // namespace litho
