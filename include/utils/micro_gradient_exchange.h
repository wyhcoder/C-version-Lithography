#pragma once

#include <Eigen/Dense>
#include <memory>
#include <vector>

#include "gradient.h"
#include "imaging.h"
#include "partition.h"

namespace litho {

enum class MicroLossType { PE, PEPenalty, EPE, PE_EPE }; // 四种数据损失；CEPT 由独立开关控制。
enum class MicroLossRegion { Core, CoreGuard }; // 数据损失的输出像素/EP 区域；CoreGuard 包含四个角区。
enum class GradientFusionMode { Average, Linear };
// 保留配置兼容；全芯片三种版图统一使用 Python 的亚像素轮廓选点。
enum class MicroEpeSampling { Via, Contact, Metal };

struct MicroGradientOptions {
    MicroLossType loss_type = MicroLossType::PE; // PE 全区域；PEPenalty 主图 PE+背景惩罚；EPE 主图 EP；PE_EPE 膨胀分区。
    MicroLossRegion loss_region = MicroLossRegion::Core; // CTM 默认评价 Core；LSM 固定 Core + Guard，反传仍覆盖完整 Halo。
    int main_region_dilation_pixels = 2; // 仅 PE_EPE 使用；target > 0.5 后 3×3 膨胀次数，每次外扩 1 像素，0 不膨胀。
    double pe_weight = 1.0; // PE、PEPenalty、PE_EPE 的 PE 系数；同一系数用于 loss 和 mask 梯度。
    double epe_weight = 1.0; // EPE、PE_EPE 的 EPE 系数；不自动归一化，保留原平方 EPE 速度项。
    double penalty_weight = 2.0; // 仅 PEPenalty 使用的背景惩罚系数，背景为 target < 0.5。
    double penalty_threshold = 0.01; // 背景 WI 超过该值时，加入平方惩罚。
    MicroEpeSampling epe_sampling = MicroEpeSampling::Via; // 仅保留版图类型标签，不再切换不同的 EP 选点算法。
    bool epe_all_eps = false; // 选点后统一 weight_epe=1，让全部已选 EP 参与梯度；与 MEEF 的 wepe_all_eps 同义。
    int epe_interval_line = 6; // 全局目标轮廓直线段的 EP 采样间隔，单位：像素，须为正数。
    int epe_interval_corner = 2; // EP 选点使用的角部间隔，单位：像素。
    double epe_pixel_size = 1.0; // 单位 nm/pixel；只换算绝对 EPE 统计，平方优化目标保持 pixel²。
    bool collect_core_metrics = false; // 独立统计 Core 指标及损失区域全部分项；LSM 固定开启。
    bool enable_cept = false; // 是否加入 Guard 与邻居 Core 的 mask 一致性平方损失。
    bool cept_guard_only = false; // true 时邻居 Core 作为冻结参考，仅驱动本块 Guard。
    double cept_weight = 1.1; // CEPT loss 及其梯度使用相同系数。
    bool enable_grad_fusion = false; // 是否融合本块 Guard 梯度与邻居 Core 梯度。
    GradientFusionMode fusion_mode = GradientFusionMode::Average; // 固定权重或沿 Guard 向外增强邻居贡献。
    double fusion_weight_owner = 0.5; // 本块 Guard 原始梯度的权重；融合权重须非负且总和为正。
    double fusion_weight_neighbor = 0.5; // 邻居 Core 原始梯度的权重；实际融合时归一化。
};

struct MicroLossWeights {
    double pe, epe, penalty; // 所选模式的有效系数；penalty 只乘 BG_raw，不再包含一份 PE。
};

struct MicroLossWindows {
    Eigen::MatrixXd pe; // 完整局部尺寸的 0/1 PE 窗口；PE 为全区域，PEPenalty 为原主图，PE_EPE 为膨胀主图外。
    Eigen::MatrixXd epe; // 同尺寸 0/1 窗口；EP 的局部像素索引落在 1 内才参与 EPE。
    Eigen::MatrixXd penalty; // 同尺寸的原始背景窗口 target < 0.5，与 PEPenalty 的主图 PE 窗口分开。
};

// 输入完整局部 target 和局部半开像素区间 region=[y0,y1)×[x0,x1)，返回 PE/EPE 的评价窗口。
// 仅 PE_EPE 先膨胀完整 target 再截 region，PE/EPE 窗口互斥；纯 EPE 评价 target 轮廓 EP，不膨胀选点。
MicroLossWindows build_micro_loss_windows(const Eigen::MatrixXd& local_target, const PartitionRect& region,
                                         const MicroGradientOptions& options);

// 返回所选类型的有效系数；启用项均读取配置权重，未启用项为 0，单项模式也不忽略 YAML 权重。
MicroLossWeights micro_loss_weights(const MicroGradientOptions& options);

// 从全局 target 的 0.5 等值线选 EP，坐标为全局浮点像素坐标 (y,x)，沿轮廓弧长线性插值采样。
// Contact/Metal/Via 都复用 Python 全芯片方法；epe_all_eps=true 时再把全部候选点的 EPE 权重设为 1。
EpsResult select_micro_epe_points(const Eigen::MatrixXd& global_target, const MicroGradientOptions& options);

struct CoreLossMetrics {
    double pe_loss = 0.0; // PE 评价窗口与 Core 交集内的 Σ(WI-target)^2，不乘混合权重。
    double bg_penalty_raw = 0.0; // 同一窗口内 target < 0.5 的 Σ max(WI-penalty_threshold,0)^2，不乘 penalty_weight。
    double pe_penalty_loss = 0.0; // Core 的 pe_weight*原主图 PE + penalty_weight*BG_raw，独立诊断 PEPenalty。
    double epe_loss = 0.0; // 激活 EP 的 Σ weight_epe*epe_pixel²，是参与总目标的 Python 稀疏 EPE 原始项。
    double epe_abs_sum_nm = 0.0; // 激活 EP 的绝对 EPE 总和，单位 nm，仅用于统计。
    int ep_count = 0; // Core 所属且 weight_epe>0 的 EP 数；全图汇总时各点只统计一次。
    double epe_mean = 0.0; // epe_abs_sum_nm/ep_count，单位 nm；Core 没有激活 EP 时为 0。
    double data_loss = 0.0; // 所选模式只在 Core 上的加权数据损失，不含 CEPT；用于评价，不作为更新目标。
};

// 一个邻居提供的两条边界梯度；矩阵只存切片数据，放置位置由对应的 MicroNeighbor 决定。
struct NeighborGradientData {
    int neighbor_id; // 数据来源的 Micro id。
    NeighborDirection direction; // 邻居位于本块的左、右、上或下。
    Eigen::MatrixXd guard_gradient; // 邻居 Core 的梯度，对应本块 my_guard_local；尺寸为该 Guard 切片的行列数。
    Eigen::MatrixXd overlap_gradient; // 邻居 Guard 梯度，对应本块 my_overlap_local；尺寸等于 Core 边界切片。
    Eigen::MatrixXd guard_mask; // 启用 CEPT 时保存邻居 Core 的 mask，对应本块 my_guard_local。
    Eigen::MatrixXd overlap_mask; // CEPT 双端模式保存邻居 Guard 的 mask；cept_guard_only 时不使用。
};

struct MicroGradientData {
    double loss; // 本块 data_loss + cept_weight * cept_loss；关闭 CEPT 时等于 data_loss。
    double data_loss; // 所选数据项，在 loss_region 内按 PE/EPE 窗口分别评价。
    double pe_loss; // 未乘混合权重的普通区域 PE，不含背景惩罚。
    double penalty_loss; // 已乘 penalty_weight 的背景惩罚贡献；仅 PEPenalty 启用。
    double bg_penalty_raw; // 损失区域的原始背景惩罚；不乘系数，collect_core_metrics 开启时也统计。
    double epe_loss; // 所选区域内激活 EP 的加权平方 EPE 原始项，单位 pixel²；重叠 EP 可被多个 Micro 评价。
    double epe_abs_sum_nm; // 同一区域激活 EP 的绝对 EPE 总和，单位 nm，仅用于 EPE-mean 统计。
    int epe_count; // 同一区域 weight_epe>0 的 EP 数；CoreGuard 包括重叠点和周期副本。
    double cept_loss; // 本块 Guard 与邻居 Core 的平方差之和，未乘 cept_weight。
    CoreLossMetrics core; // collect_core_metrics 开启时，用同一次局部成像独立统计 Core；不改变 loss/梯度。
    Eigen::MatrixXd raw_mask_gradient; // PE/EPE 数据项梯度，尺寸为 tile_size × tile_size，包含 Halo。
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
    // meta 和 cache 须比本对象存活更久；按 OpenMP 最大线程数创建工作对象，串行准备并复用 FFT 计划。
    MicroGradientExchange(const PartitionMeta& meta, const ImagingCache& cache, double threshold = 0.25, double alpha = 50.0);

    // 两张输入矩阵均为全局尺寸 meta.height × meta.width；mask 为当前透射率，target 为目标图。
    // 先并行完成各块成像、loss 和梯度，再串行交换、计算 CEPT 和融合。
    // parallel_micro=false 时逐块串行计算。
    // 默认使用 PE，关闭 CEPT 和融合；不修改输入 mask。
    MicroGradientExchangeResult compute(const Eigen::MatrixXd& global_mask, const Eigen::MatrixXd& global_target,
                                       bool parallel_micro = true);

    // 配置损失、CEPT 和融合；全图输入会提取一致的局部 mask，因此其 CEPT loss/梯度为 0。
    MicroGradientExchangeResult compute(const Eigen::MatrixXd& global_mask, const Eigen::MatrixXd& global_target,
                                       const MicroGradientOptions& options, bool parallel_micro = true);

    // local_masks[id] 为该块独立保存的完整局部 mask，可包含与邻居 Core 不同的 Guard 值，让 CEPT 发挥作用。
    // PE/PEPenalty 关闭融合且使用双端 CEPT 时，mask_gradient 对应总 loss 导数。
    // cept_guard_only 使用冻结邻居 Core 的局部驱动；EPE 使用亚像素双线性取样和项目已有的光学反传。
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
