#pragma once

#include "loss.h"
#include "partition.h"

#include <filesystem>

namespace litho {

// 输入全图 AI、原 ep_points.csv、显影阈值和像素尺寸（nm/像素），按现有 Loss::evaluate_epe() 导出逐点 EPE。
// EP 坐标是全局像素索引 (y,x)，范围 [0,height) × [0,width)；保留原 EP 编号、Core 归属和权重，不重新选点。
// 输出 CSV 的 epe_nm 是未加权绝对 EPE（nm），aerial_intensity 是该 EP 的 AI；返回同一行序的 EPE 向量及总和。
EpeEvaluation save_full_chip_ep_epe(const Eigen::MatrixXd& aerial_image, const std::filesystem::path& ep_points_csv,
                                  double threshold, double pixel_size_nm, const std::filesystem::path& output_csv);

// 输入各 Micro 未裁切的局部 AI，按 ep_points.csv 中唯一的 Core 归属取值并求斜率，不跨 Core 回拼边界差分。
// 全局 EP 像素坐标减去所属 tile_global 的起点得到局部索引；返回/导出行序和未加权 nm 单位与上面接口一致。
EpeEvaluation save_micro_ep_epe(const std::vector<Eigen::MatrixXd>& local_aerial_images, const PartitionMeta& meta,
                               const std::filesystem::path& ep_points_csv, double threshold, double pixel_size_nm,
                               const std::filesystem::path& output_csv);

} // namespace litho
