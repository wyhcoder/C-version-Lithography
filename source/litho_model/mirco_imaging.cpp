#include "mirco_imaging.h"

#include <utility>

namespace litho {

MicroImaging::MicroImaging(Imaging& imaging, const PartitionMeta& meta) : _imaging(imaging), _meta(meta) {}

MicroImagingResult MicroImaging::compute(const Eigen::MatrixXd& global_mask, double threshold, double alpha) {
    MicroImagingResult result;
    result.local_aerial_images.reserve(_meta.tiles.size());
    result.local_wafer_images.reserve(_meta.tiles.size());

    // tiles 按 Micro id 排列；每块读取同一张 global_mask，边界 Halo 由 Partition 进行周期取值。
    for (const MicroTile& tile : _meta.tiles) {
        const Eigen::MatrixXd local_mask = Partition::extract_local_tile(global_mask, tile);
        Imaging_Result local_result = _imaging.compute(local_mask, threshold, alpha);
        // 移交矩阵的存储空间，避免再复制一份完整的局部 AI/WI。
        result.local_aerial_images.push_back(std::move(local_result.aerial_image));
        result.local_wafer_images.push_back(std::move(local_result.wafer_image));
    }

    // Halo 只参与局部成像；回拼时每个 Core 写入自己拥有的全局区域。
    result.global_aerial_image = Partition::assemble_global(result.local_aerial_images, _meta);
    result.global_wafer_image = Partition::assemble_global(result.local_wafer_images, _meta);
    return result;
}

} // namespace litho
