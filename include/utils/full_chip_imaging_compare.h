#pragma once

#include "imaging.h"
#include "simulation_parameters.h"

#include <filesystem>

namespace litho {

// 仅决定导出文件名和图注；两种阶段都使用同一个全图成像接口。
enum class FullChipImagingStage { CTM, LSM };

struct FullChipImagingComparison {
    int grid_size; // 全图成像/FFT 网格边长，单位：像素；当前版图为 1024。
    int source_grid_size, source_points; // 照明积分采样网格边长及非零光源点数，与全图 FFT 尺寸独立。
    int kernels; // 此套全图光学模型保留的 SOCS 核数。
    double pixel_size_nm;
    Eigen::MatrixXd target_mask; // 原始 target 版图，行列为全局像素索引。
    Eigen::MatrixXd optimized_mask; // 最佳轮回拼版图：CTM 为灰度，LSM 为 0/1；保留输入原值。
    Imaging_Result target_image; // target 通过全图模型得到的 AI 和连续显影 WI。
    Imaging_Result optimized_image; // 优化版图通过同一模型得到的 AI/WI。
    double target_pe, optimized_pe; // 各自的 Σ(WI-target)²，仅计算成像误差，不包含 CEPT。
};

// target 与 optimized_mask 是相同尺寸的全局矩阵；当前全图成像使用正方形、2 的幂尺寸。
// 按 target 的边长重建一套全图光学模型，分别对两个输入成像；mask 和显影 WI 均保留输入/计算原值。
// source_grid_size 仅控制照明积分采样；两次成像共享同一套光源、光瞳、SOCS 核和显影参数。
FullChipImagingComparison compare_full_chip_imaging(const SimulationParameters& params, const Eigen::MatrixXd& target,
                                                    const Eigen::MatrixXd& optimized_mask, int source_grid_size = 256);

// 保存 mask、AI/WI 的 double 矩阵、评价指标和采样参数；stage 区分 CTM 灰度与 LSM 二值结果的文件名。
void save_full_chip_imaging_comparison(const FullChipImagingComparison& result, const std::filesystem::path& directory,
                                     FullChipImagingStage stage = FullChipImagingStage::CTM);

} // namespace litho
