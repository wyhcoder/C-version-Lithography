#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "gradient.h"
#include "imaging.h"
#include "partition.h"

namespace litho {

enum class MicroLossType { PE, PEPenalty, EPE, PE_EPE };
enum class MicroLossRegion { Core, CoreGuard }; // 数据损失的输出像素/EP 区域；CoreGuard 包含四个角区。
enum class GradientFusionMode { Average, Linear };

struct MicroGradientOptions {
    MicroLossType loss_type = MicroLossType::PE; // 数据项：PE、带背景惩罚的 PE、现有 C++ EPE、PE + EPE。
    MicroLossRegion loss_region = MicroLossRegion::Core; // CTM 默认只评价 Core；LSM 固定评价 Core + Guard。
    double pe_weight = 0.5; // 仅 PE_EPE 使用；权重直接乘入 loss 和梯度，不自动归一化。
    double epe_weight = 0.5; // 仅 PE_EPE 使用。
    double penalty_weight = 2.0; // PEPenalty 的背景惩罚系数，背景定义为 target < 0.5。
    double penalty_threshold = 0.01; // 背景 WI 超过该值时，加入平方惩罚。
    int epe_interval_line = 6; // 全局目标轮廓直线段的 EP 采样间隔，默认与 C++ EpSelect 一致，单位：像素，须为正数。
    int epe_interval_corner = 2; // EP 选点使用的角部间隔，单位：像素。
    double epe_pixel_size = 1.0; // 直接传给现有 C++ EPE 接口的 dx；1 表示像素，设为 pixel_size_nm 时 loss 的单位为 nm。
    bool enable_cept = false; // 是否加入 Guard 与邻居 Core 的 mask 一致性平方损失。
    bool cept_guard_only = false; // true 时邻居 Core 作为冻结参考，仅驱动本块 Guard；LSM 固定使用此规则。
    double cept_weight = 1.1; // CEPT loss 及其梯度使用相同系数。
    bool enable_grad_fusion = false; // 是否融合本块 Guard 梯度与邻居 Core 梯度。
    GradientFusionMode fusion_mode = GradientFusionMode::Average; // 固定权重或沿 Guard 向外增强邻居贡献。
    double fusion_weight_owner = 0.5; // 本块 Guard 原始梯度的权重；融合权重须非负且总和为正。
    double fusion_weight_neighbor = 0.5; // 邻居 Core 原始梯度的权重；实际融合时归一化。
};

// 一个邻居提供的两条边界梯度；矩阵只存切片数据，放置位置由对应的 MicroNeighbor 决定。
struct NeighborGradientData {
    int neighbor_id; // 数据来源的 Micro id。
    NeighborDirection direction; // 邻居位于本块的左、右、上或下。
    Eigen::MatrixXd guard_gradient; // 邻居 Core 的梯度，对应本块 my_guard_local；尺寸为该 Guard 切片的行列数。
    Eigen::MatrixXd overlap_gradient; // 邻居 Guard 的梯度，对应本块 my_overlap_local；尺寸为该 Core 边界切片的行列数。
    Eigen::MatrixXd guard_mask; // 启用 CEPT 时保存邻居 Core 的 mask，对应本块 my_guard_local。
    Eigen::MatrixXd overlap_mask; // CEPT 双端模式保存邻居 Guard 的 mask；cept_guard_only 时不使用。
};

struct MicroGradientData {
    double loss; // 本块 data_loss + cept_weight * cept_loss；关闭 CEPT 时等于 data_loss。
    double data_loss; // 所选数据项，在 loss_region 指定的 Core 或 Core + Guard 内评价。
    double pe_loss; // 未乘混合权重的区域 PE；PEPenalty 模式时包含该区域内的背景惩罚。
    double penalty_loss; // pe_loss 中已经乘 penalty_weight 的背景惩罚，仅 PEPenalty 非零，用于日志分解。
    double epe_loss; // 所选区域内 EP 的原 C++ 绝对 EPE 之和；CoreGuard 的重叠 EP 可被多个 Micro 评价。
    double cept_loss; // 本块 Guard 与邻居 Core 的平方差之和，未乘 cept_weight。
    Eigen::MatrixXd raw_mask_gradient; // 按现有 PE/EPE 接口计算的数据项梯度，尺寸为 tile_size × tile_size，包含 Halo。
    Eigen::MatrixXd cept_mask_gradient; // 未加权 CEPT 驱动；cept_guard_only 时仅 Guard 非零，否则包含 Core 端贡献。
    Eigen::MatrixXd mask_gradient; // 最终驱动：可选 Guard 融合后的数据梯度 + 加权 CEPT 梯度。
    std::vector<NeighborGradientData> received_neighbors; // 与 meta.tiles[id].neighbors 顺序一致，保存四邻域交换结果。
};

struct MicroGradientExchangeResult {
    double loss; // 全局 data_loss + 加权 CEPT loss；梯度融合不改变该标量。
    double data_loss; // 各块所选数据 loss 之和。
    double cept_loss; // 各块有向 Guard→邻居 Core 约束的平方差之和，未乘权重。
    std::vector<MicroGradientData> locals; // 下标为 Micro id；原始梯度与交换结果分开保存。
};

class MicroGradientExchange {
public:
    // meta 只读引用；cache 在外部按完整局部尺寸准备，须满足 cache.N == meta.tile_size。
    // meta 和 cache 须比本对象存活更久；按构造时 OpenMP 最大线程数创建工作对象，串行准备并复用 FFT 计划。
    MicroGradientExchange(const PartitionMeta& meta, const ImagingCache& cache, double threshold = 0.25, double alpha = 50.0);

    // 两张输入矩阵均为全局尺寸 meta.height × meta.width；mask 为当前透射率，target 为目标图。
    // 先并行完成各块成像、loss 和梯度，再串行交换、计算 CEPT 和融合；parallel_micro=false 改为逐块串行计算。
    // 默认使用 PE，关闭 CEPT 和融合；不修改输入 mask。
    MicroGradientExchangeResult compute(const Eigen::MatrixXd& global_mask, const Eigen::MatrixXd& global_target,
                                       bool parallel_micro = true);

    // 配置损失、CEPT 和融合；全图输入会提取一致的局部 mask，因此其 CEPT loss/梯度为 0。
    MicroGradientExchangeResult compute(const Eigen::MatrixXd& global_mask, const Eigen::MatrixXd& global_target,
                                       const MicroGradientOptions& options, bool parallel_micro = true);

    // local_masks[id] 为该块独立保存的完整局部 mask，可包含与邻居 Core 不同的 Guard 值，让 CEPT 发挥作用。
    // PE/PEPenalty 关闭融合且使用双端 CEPT 时，mask_gradient 对应总 loss 导数。
    // cept_guard_only 使用冻结邻居 Core 的局部驱动；EPE 分支沿用原 C++ 梯度接口的计算规则。
    MicroGradientExchangeResult compute(const std::vector<Eigen::MatrixXd>& local_masks, const Eigen::MatrixXd& global_target,
                                       const MicroGradientOptions& options = {}, bool parallel_micro = true);

private:
    // 每个工作线程独占可变电场、FFT 缓冲区和计划；光学 cache 在各工作对象之间只读共享。
    struct Worker {
        Worker(const ImagingCache& cache, double threshold, double alpha);
        Imaging imaging;
        Gradient gradient;
    };

    const PartitionMeta& _meta; // 分区坐标与四邻域关系。
    double _threshold; // 显影阈值，与内部 Gradient 使用的阈值一致。
    double _alpha; // 显影 sigmoid 陡峭程度，与梯度链式法则使用的 alpha 一致。
    std::vector<std::unique_ptr<Worker>> _workers; // 数量不超过 Micro 数；按线程编号索引，每轮计算复用。
};

} // namespace litho
