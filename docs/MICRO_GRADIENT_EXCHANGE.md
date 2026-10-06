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

`loss_region=Core` 默认仅在 Core 评价，供 CTM 使用；`CoreGuard` 在 Core + Guard 的完整矩形内评价，供 LSM 使用。
两种区域的反传结果都覆盖完整局部 mask，包括 Halo。下表的“区域”指当前所选损失区域。

| `loss_type` | 数据损失 |
| --- | --- |
| `PE` | 区域内 `Σ(W - T)²` |
| `PEPenalty` | PE 加上区域内背景的 `penalty_weight * Σ max(W - penalty_threshold, 0)²` |
| `EPE` | 区域内 EP 上，原 C++ `Loss::epe_loss()` 返回的绝对 EPE 之和 |
| `PE_EPE` | `pe_weight * PE + epe_weight * EPE` |

混合权重直接乘入 loss 和梯度，不自动归一化。背景为 `target < 0.5`。

EPE 的 EP 按当前代码使用 `EpSelect::select_eps_via(2)` 从全局目标轮廓选取。
全芯片适配筛选所选区域内的 EP，并减去局部矩阵原点，将全局 EP 转成局部坐标，原 EP 权重保持不变。
Core 模式下每个 EP 归属于一个 Core；CoreGuard 模式允许重叠评价，并为芯片边界外的周期 Guard 提供对应 EP 副本。
随后直接调用已有 C++ 接口：

```cpp
Loss::epe_loss(aerial_image, local_eps, threshold, options.epe_pixel_size);
gradient.epe_gradient(aerial_image, electric_field, local_eps, threshold, options.epe_pixel_size);
```

`Loss::epe_loss()` 的公式为 `Σ |I - threshold| / (|gradient(I)| + 1e-12) * dx`，不对 EP 加权。
`epe_pixel_size` 直接作为 `dx` 传入；1 表示像素，设为像素的 nm 尺度时 loss 单位为 nm。
梯度沿用现有 `Gradient::epe_gradient(aerial, electric_field, eps, threshold, dx)`：按 `weight_epe` 加权、采用平方误差形式并将梯度模视为常量。
该原接口的当前实现未把 `dx` 乘入梯度。此模块保持原有 loss 和梯度计算规则，不另外定义平方 EPE 或平滑分母。

## CEPT

对每条本块 Guard→邻居 Core 约束，比较两个独立局部 mask 在同一全局位置的值：

```text
diff = 本块 Guard mask - 邻居 Core mask
CEPT = Σ diff²
```

总 loss 为 `data_loss + cept_weight * cept_loss`。本块 Guard 的导数为 `2 * diff`。
默认 `cept_guard_only=false`，同时收集邻居 Guard 约束对本块 Core 的另一端贡献；此模式的 CEPT 梯度对应全部独立局部 mask 的总损失。
`cept_guard_only=true` 时将邻居 Core 作为冻结参考，仅写本块 Guard，不给 Core 添加 CEPT 梯度；这与 Python 的局部协调流程一致。
LSM 固定使用 Guard-only 模式，CTM 保留默认双端模式。

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

自动测试覆盖 Abbe/SOCS、两种损失区域的 PE/PEPenalty 有限差分、双端/Guard-only CEPT、EPE 与原 C++ 接口的一致性、Core EP 归属和周期 Guard EP、两种融合模式及串行/并行一致性。
真实输入检查比较逐块串行与 Micro 并行的结果，报告各一次预热后的计算耗时；使用两张 1024×1024 版图，启用 PE+EPE、CEPT 和融合，划分为 8×8 Micro、Halo 64、Guard 30。
