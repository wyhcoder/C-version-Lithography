#pragma once

#include "partition.h"

#include <filesystem>

namespace litho {

// 导出真实 PartitionMeta 的全图分块坐标，供绘图脚本在同一张完整 target 上叠加各区域。
// 所有矩形为全局像素索引 [y0,y1)×[x0,x1)；update 是 Core+Guard，tile 是完整 Core+Halo 成像范围。
// pixel_size_nm 为 nm/像素，仅用于图注；directory 内写 partition_meta.csv 和 micro_regions.csv。
void save_partition_geometry(const PartitionMeta& meta, double pixel_size_nm, const std::filesystem::path& directory);

} // namespace litho
