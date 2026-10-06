#pragma once

#include <Eigen/Dense>
#include <vector>

#include "imaging.h"
#include "partition.h"

namespace litho {

struct MicroImagingResult {
    Eigen::MatrixXd global_aerial_image; // 各块 AI 的 Core 回拼结果，尺寸为 meta.height × meta.width。
    Eigen::MatrixXd global_wafer_image;  // 各块 WI 的 Core 回拼结果，尺寸与全局 mask 相同。
    std::vector<Eigen::MatrixXd> local_aerial_images; // 完整局部 AI，含 Halo；下标为 Micro id，尺寸为 tile_size × tile_size。
    std::vector<Eigen::MatrixXd> local_wafer_images;  // 完整局部 WI（显影后的连续值），含 Halo；顺序与 local_aerial_images 相同。
};

class MicroImaging {
public:
    // imaging 的外部光学缓存须满足 cache.N == meta.tile_size；复用缓存和 FFT 计划，不在此处初始化光学系统。
    // 本对象只保存引用，imaging 和 meta 须比本对象存活更久。
    MicroImaging(Imaging& imaging, const PartitionMeta& meta);
    ~MicroImaging() = default;

    // global_mask 为完整大版图，其行列数对应 meta.height/width；threshold 和 alpha 为显影阈值和 sigmoid 陡峭程度。
    // 串行提取 Core + Halo 并成像，返回完整局部结果和只回拼 Core 的全局结果；多次调用可传入更新后的 mask。
    MicroImagingResult compute(const Eigen::MatrixXd& global_mask, double threshold = 0.25, double alpha = 50.0);

private:
    Imaging& _imaging; // 引用外部成像对象，不在每个 Micro 中重复初始化。
    const PartitionMeta& _meta; // 引用已有的分区几何信息，不持有全局 mask。
};

} // namespace litho
