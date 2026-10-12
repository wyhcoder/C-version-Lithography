# 全芯片局部梯度、CEPT 与边界融合

接口位于 `include/utils/micro_gradient_exchange.h`，实现位于 `source/utils/micro_gradient_exchange.cpp`。

## 调用方式

光学缓存由外部准备，满足 `cache.N == meta.tile_size`。构造对象时按 OpenMP 最大线程数创建工作对象，数量不超过 Micro 数。每个工作对象独占成像、梯度和 FFT 计划，计划在构造阶段串行创建，后续每轮计算复用。

```cpp
#include "micro_gradient_exchange.h"

litho::MicroGradientExchange evaluator(meta, cache, params.resist.threshold, params.resist.alpha);
litho::MicroGradientOptions options;
options.loss_type = litho::MicroLossType::PE_EPE;
options.pe_weight = 0.8;
options.epe_weight = 0.2;
options.enable_cept = true;
options.cept_weight = 1.1;
options.enable_grad_fusion = true;
options.fusion_mode = litho::GradientFusionMode::Average;

// local_masks[id] 保存该块独立的、含 Halo 的 mask。
auto result = evaluator.compute(local_masks, global_target, options);
```

也可直接传全局 mask：

```cpp
auto result = evaluator.compute(global_mask, global_target, options);
```

全图接口会先提取所有局部 mask。同一张全局 mask 的 Guard 和邻居 Core 在同一全局位置取值相同，所以这次调用的 CEPT 为 0。
独立局部 mask 接口可保留各块优化产生的边界差异，CEPT 随这些差异产生损失及梯度。

## 损失选择

`loss_region=Core` 仅在 Core 评价，是全芯片 CTM 的默认规则；
`CoreGuard` 在 Core + Guard 的完整矩形内评价，LSM 固定使用此范围，损失窗口由 `loss_type` 决定。
两种区域的反传结果都覆盖完整局部 mask，包括 Halo。
PE 不划分主图/背景；PEPenalty 使用未膨胀主图 PE 和未膨胀背景惩罚两个窗口。
仅 PE_EPE 先对完整局部 `target>0.5` 做 3×3 膨胀，再截所选区域得到 EPE 窗口，外部为 PE 窗口。
`main_region_dilation_pixels=2` 表示向外扩 2 像素，不改变原始 EP 选点坐标。
窗口限制输出误差和 EP 的评价位置，光学反传后保留完整梯度，更新窗口由优化器单独控制。

| `loss_type` | 数据损失 |
| --- | --- |
| `PE` | `pe_weight * Σ[全部区域] (W - T)²` |
| `PEPenalty` | `pe_weight * Σ[target>0.5] (W - T)² + penalty_weight * Σ[target<0.5] max(W - penalty_threshold, 0)²` |
| `EPE` | `epe_weight * 主图轮廓 EP 的原 C++ 绝对 EPE 之和` |
| `PE_EPE` | `pe_weight * 膨胀主图外背景 PE + epe_weight * 膨胀主图内 EP 的 EPE` |

仅支持这四种类型；所有启用项的配置权重同时乘入 loss 和梯度，单项模式也读取权重，不自动归一化。
EPE 保留之前指定的平方 EPE 近似驱动，报告损失仍为绝对 EPE，不是该统计量的严格导数。

EP 从全局 target 选取：Via 调用 `select_eps_via(2)`，Contact 仅保留 `select_eps_via(0)` 的四边中点，Metal 调用原 `select_eps_others()`。
`epe_all_eps` 可以在选点后统一权重为 1；全芯片适配筛选 EPE 窗口内的 EP，减去局部矩阵原点转为局部坐标。
Core 模式下每个 EP 归属于一个 Core；CoreGuard 模式允许重叠评价，并为芯片边界外的周期 Guard 提供对应 EP 副本。
随后直接调用已有 C++ 接口：

```cpp
Loss::epe_loss(aerial_image, local_eps, threshold, options.epe_pixel_size);
gradient.epe_gradient(aerial_image, electric_field, local_eps, threshold, options.epe_pixel_size);
```

`Loss::epe_loss()` 的公式为 `Σ |I - threshold| / (|gradient(I)| + 1e-12) * dx`，不对 EP 加权。
`epe_pixel_size` 直接作为 `dx` 传入；1 表示像素，设为像素的 nm 尺度时 loss 单位为 nm。
梯度使用 `Gradient::epe_gradient(aerial, electric_field, eps, threshold, dx)`：
平方 EPE 驱动为 `Σ weight_epe*(I-threshold)^2*dx^2/G^2`，其中 `G` 由步长 1 像素的空间差分获得，本次反传冻结 `G`。
2026-10-10 补入了 `dx^2`，并在光学反传末端使用系数 2，计入强度 `I=|E|^2` 对实值 mask 的导数。
报告的 loss 仍为绝对 EPE；平方 EPE 的冻结分母驱动有意与报告指标不同，不作为绝对 EPE 的严格梯度。

## CEPT

对每条本块 Guard→邻居 Core 约束，比较两个独立局部 mask 在同一全局位置的值：

```text
diff = 本块 Guard mask - 邻居 Core mask
CEPT = Σ diff²
```

总 loss 为 `data_loss + cept_weight * cept_loss`。本块 Guard 的导数为 `2 * diff`。
默认 `cept_guard_only=false`，同时收集邻居 Guard 约束对本块 Core 的另一端贡献；此模式的 CEPT 梯度对应全部独立局部 mask 的总损失。
`cept_guard_only=true` 时将邻居 Core 作为冻结参考，仅写本块 Guard，不给 Core 添加 CEPT 梯度；这与 Python 的局部协调流程一致。
全芯片 CTM 和 LSM 均在 optimizer 内固定使用 Guard-only 模式；共享模块独立调用时仍可选择双端模式。

## 梯度融合与并行

`enable_grad_fusion` 独立控制融合，默认关闭。融合只修改 Guard 的数据梯度，Core 数据梯度保持不变。

```text
fused = (owner_weight * 本块原始梯度 + active_neighbor_weight * 邻居 Core 原始梯度)
        / (owner_weight + active_neighbor_weight)
```

`Average` 使用固定邻居权重；`Linear` 将邻居强度从靠近 Core 的 0 增加到 Guard 外侧的 1。
单像素 Guard 的强度为 1。最后再加上加权 CEPT 梯度，得到 `mask_gradient`。

执行顺序为：

```text
串行：准备全局 EP（选用 EPE 时）
并行：Micro 0 / Micro 1 / … → 各自成像、所选区域的 loss、完整局部梯度
同步：等待全部 Micro 计算完成
串行：四邻域信息交换 → CEPT → 可选 Guard 梯度融合 → 总 loss 汇总
```

各线程只读输入与光学 cache，只写分配给自己的 `result.locals[id]`，使用线程独占的 Imaging/Gradient 对象。
`Imaging` 和 `Gradient` 在外层并行时，通过 `omp_in_parallel()` 自动关闭内部光学核并行。
每块成像后立即用同一工作对象的电场反传；下一块覆盖电场之前，当前梯度已保存。
交换始终读取完整的原始梯度，融合后的梯度不会作为后续块的交换源。

最后一个参数 `parallel_micro=false` 改为逐块串行计算，内部光学核并行照常可用，交换阶段仍串行：

```cpp
auto serial = evaluator.compute(local_masks, global_target, options, false);
auto parallel = evaluator.compute(local_masks, global_target, options, true);
```

通过 `OMP_NUM_THREADS=4` 设置线程数，或在构造 evaluator 之前调用 `omp_set_num_threads(4)`。
实际 Micro 线程数取工作对象数量与当前 OpenMP 最大线程数的较小值；构造之后增加线程数不会扩充工作对象。
一个 evaluator 保存可变的工作缓存，应由一个调用者逐轮调用 `compute()`。

## 返回结果与后续更新

| 成员 | 含义 |
| --- | --- |
| `result.loss` | 总数据损失加加权 CEPT |
| `result.data_loss` | 所选数据项之和 |
| `result.cept_loss` | 未乘权重的 CEPT |
| `result.locals[id].raw_mask_gradient` | 本块原始数据梯度 |
| `result.locals[id].cept_mask_gradient` | 所选 CEPT 模式的本块驱动，未乘权重 |
| `result.locals[id].mask_gradient` | 可选融合后数据梯度加加权 CEPT |
| `result.locals[id].received_neighbors` | 按四邻域几何顺序保存的交换数据 |

融合改变的是更新驱动，不改变标量 loss；开启融合后，这个驱动一般不是该标量 loss 的严格导数。
PE/PEPenalty 接标准 L-BFGS 线搜索时使用关闭融合、双端 CEPT 的严格梯度。Guard-only CEPT 是冻结邻居参考的局部驱动。
EPE 沿用原 C++ 近似梯度，即使关闭融合，也不能据此声称它是所报告绝对 EPE loss 的严格导数。

以上梯度相对于独立的局部 mask。若优化变量是唯一的全局 mask，须将各块完整局部梯度按周期坐标累加回全图，包含 Halo 的贡献；只回拼 Core 不能得到该全局 loss 的完整导数。
CTM 的参数梯度进一步由 `G_theta = -0.5 * G_mask * sin(theta)` 计算。此模块不更新 theta 或 phi。

完整的固定步长 CTM 迭代已接入 `FullChipCTMOptimizer`，调用、轮次顺序和运行命令见 [FULL_CHIP_CTM.md](FULL_CHIP_CTM.md)。

## 验证命令

```bash
cmake --build build-release --target test_micro_gradient_exchange --parallel 4
ctest --test-dir build-release -R '^micro_gradient_exchange$' --output-on-failure
cd build-release
OMP_NUM_THREADS=4 ./test_micro_gradient_exchange --real
```

自动测试覆盖 Abbe/SOCS、膨胀主图形/背景窗口、PE/PEPenalty 的完整局部梯度有限差分、双端/Guard-only CEPT、原 C++ EPE 接口、Core/周期 Guard EP、两种融合模式及串行/并行一致性。
真实输入检查比较逐块串行与 Micro 并行的结果，报告各一次预热后的计算耗时；使用两张 1024×1024 版图，启用 PE+EPE、CEPT 和融合，划分为 8×8 Micro、Halo 64、Guard 30。
