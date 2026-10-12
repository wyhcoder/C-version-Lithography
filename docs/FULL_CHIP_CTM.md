# 全芯片 CTM 固定步长优化

接口：`include/optimizer/full_chip_ctm.h`。实现：`source/optimizer/full_chip_ctm.cpp`。
入口：`demo/demo_full_chip_ctm.cpp`。独立配置：`config_full_chip_ctm.yaml`。

## 运行

从项目根目录执行：

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target demo_full_chip_ctm --parallel 4
OMP_NUM_THREADS=4 ./build-release/demo_full_chip_ctm config_full_chip_ctm.yaml 20 0.05
```

命令行参数依次为 YAML、更新次数、theta 域固定步长、可选目标 BMP。
不传参数时使用项目根目录的 `config_full_chip_ctm.yaml`，迭代数和固定步长读取该 YAML。
迭代次数和步长可以直接通过第二、第三个参数覆盖，覆盖值也写入输出配置。

例如执行 contact：

```bash
OMP_NUM_THREADS=4 ./build-release/demo_full_chip_ctm config_full_chip_ctm.yaml 20 0.05 assets/target_pattern/contact_1_0_only_1024.bmp
```

YAML 的输入/输出路径以及第四个命令行参数均相对于 YAML 所在目录，绝对路径也可使用。
传入第四个参数时，输出目录使用目标文件名，避免两张版图写到同一目录。
默认划分为 8×8 Micro，Core 128×128、每侧 Halo 64、每侧 Guard 30，完整局部矩阵为 256×256。
外部 `Grid/Pupil/Source/LithoPrepare` 按该局部尺寸初始化一次；所有 Micro 和所有轮次共享只读光学 cache。
当前 `Grid` 会将尺寸向上取为 2 的幂，修改分块参数时须保证 `grid.size() == meta.tile_size`。

## 每轮做什么

沿用原 C++ CTM 初始化：目标非零像素对应 `theta = 0.3π`，其余对应 `theta = 0.7π`。
初始化接口 `CTM_Optimizer::initialize_theta()` 改为静态函数，单块与全芯片流程共用该初始化规则。

```text
初始化各块独立 theta 和 mask
    ↓
并行成像及梯度 → 等待所有块完成 → 串行交换、CEPT 和可选融合
    ↓
记录第 0 次评价（初始状态）
    ↓
重复指定次数：
    并行更新各块 Core + Guard 的 theta 和 mask
    等待所有更新完成
    Core + Guard 之外的 theta 保持本块初值，不从 Core 回拼图刷新
    并行成像及梯度 → 等待所有块完成 → 串行交换、CEPT 和可选融合
    记录本次更新后的评价；交换后的梯度供下一次更新使用
    ↓
返回历史最佳轮的局部 theta/mask，以及同一轮各 Core 回拼的全图 mask
```

所有 Micro 的一次更新都使用同一轮已收集完整的梯度。更新公式为：

```text
M = 0.5 * (1 + cos(theta))
G_theta = -0.5 * G_mask * sin(theta)
theta_new = theta - learning_rate * G_theta
```

`G_mask` 包含所选 PE/EPE 数据梯度、可选 Guard 梯度融合和加权 CEPT 梯度。
只在 `tile.update_local`（Core + Guard）执行 theta 更新。
外层 Halo 的 theta 始终保留本块初始化时的值；其 mask 提供下一轮成像上下文，不通过全局回拼刷新。
各块 Guard 独立保存，不用邻居 Core 覆盖，所以更新后能产生像素差异，交由 CEPT 协调。
四邻域 CEPT 比较更新后的 A Guard 与 B Core 在同一全局位置的 mask 像素值，与梯度差异无关。
CTM 内部固定 `cept_guard_only=true`：B Core 是冻结参考，只将 `2*(A_Guard-B_Core)` 写入 A Guard，
不额外加入反向邻居损失对 A Core 的导数；A Core 仍正常接收数据项梯度。

`iterations` 是实际更新次数，日志/history 共 `iterations + 1` 项；最后一项评价的是末次更新后的状态。
每次评价后以总 `loss = data_loss + cept_weight * cept_loss` 比较历史最佳值，并同步保存该轮全部 Micro 的完整 theta/mask。
初始状态也参与比较；损失相等时保留首次达到该值的状态。返回值的 `best_iteration` 和 `best_loss` 标识最终选择。
支持 0 次更新以查看初始状态。每轮步长固定，不做线搜索或基于梯度范数的提前停止。
固定步长的损失不保证逐轮下降；融合及原 C++ EPE 近似梯度的语义见 `MICRO_GRADIENT_EXCHANGE.md`。

## C++ 调用

```cpp
FullChipCTMOptimizer optimizer(meta, cache, threshold, alpha);
FullChipCTMOptions options;
options.iterations = 20;
options.learning_rate = 0.05;
options.gradient_options.loss_type = MicroLossType::PE;
options.gradient_options.enable_cept = true;
options.gradient_options.enable_grad_fusion = false;
auto result = optimizer.optimize(global_target, options);
```

`parallel_micro=false` 将成像/反传和参数更新改为逐块串行，可与默认并行路径比较。
交换、融合、CEPT 和最终 Core 回拼串行；每轮不刷新外围 Halo。
线程数在构造 optimizer 前通过 `OMP_NUM_THREADS` 或 `omp_set_num_threads()` 设置。
对象持有可变工作缓存，供一个调用者逐轮使用；meta/cache 须比 optimizer 存活更久。

## 输出

启动时打印实际损失类型、数据评价区域、权重和损失公式。CTM 默认评价 Core，可通过 C++ 的 `loss_region` 选择 Core + Guard。
每轮日志与全芯片 LSM 使用相同的分项名称：

- `Total-loss`：所选数据损失加上加权 CEPT，各项系数在启动配置中显示。
- `PE-all(raw)` / `PE-main(raw)` / `PE-bg(raw)`：PE 全区域、PEPenalty 原主图、PE_EPE 膨胀主图外背景的原始 PE。
- `BG(raw)`：原背景的阈值平方惩罚，只在 PEPenalty 展示。
- `EPE(raw)`：主图轮廓 EP 的原始绝对 EPE，仅 EPE/PE_EPE 展示，后者使用膨胀窗口筛选。
- `CEPT(raw)`：本块 Guard 与四邻域 Core 的 mask 像素平方差总和，仅开启 CEPT 时展示。
- `theta_step` 在配置行显示：theta 域固定学习率，与 LSM 的 CFL 时间步不同。

`BG(raw) = Σ[target<0.5] max(WI-penalty_threshold,0)^2`，PEPenalty 不另加普通背景 PE。
四种类型的启用项均使用 YAML 对应权重，损失与梯度同权重，不自动归一化。
梯度融合改变更新方向，不额外加入一个损失项。

默认输出到 `result/full_chip_ctm/<目标文件名>/`：

| 文件 | 内容 |
| --- | --- |
| `gray_mask.txt` | 历史最佳轮 Core 回拼的连续 mask，含行列数注释 |
| `gray_mask.bmp` | 同一 mask 映射到 0～255 的预览 |
| `binary_mask.bmp` | 原 C++ `Imbinarize::binarize_advanced()` 返回的 `final_mask` 预览 |
| `binary_mask.txt` | 同一 `final_mask` 的 0/1 矩阵，供后续 Level Set 使用 |
| `target.bmp` | 本次输入目标图 |
| `history.csv` | 原列和已加权贡献保留，追加 `pe_loss_raw,bg_penalty_raw,epe_loss_raw` 原始分项 |
| `summary.csv` | 最佳轮编号/损失、末轮编号/损失；导出 mask 对应最佳轮 |
| `config_used.yaml` | 实际参数，包括命令行覆盖后的更新次数、步长和路径 |

返回的 `local_masks` / `local_theta` 保留同一最佳轮的各块完整独立状态，便于后续接 Level Set 或继续研究边界协调。
`global_mask` 只回拼 Core；history 的 data_loss 是各块所选区域的局部成像损失汇总，loss 还包含加权 CEPT。
追加的 CSV 列为 `pe_contribution`、`penalty_contribution`、`epe_contribution`、`cept_contribution`，未启用的项为 0。
满足 `data_loss = pe_contribution + penalty_contribution + epe_contribution`，`loss = data_loss + cept_contribution`。

## 灰度 mask 的 Micro 成像对照

`demo_full_chip_ctm` 结束后自动生成 `imaging_comparison/`。成像输入为最佳轮各块的连续 `local_masks`，
不使用 `binary_mask`；生成的 WI 也保留 sigmoid 连续值。新增接口声明在
`include/utils/full_chip_imaging_compare.h`，实现在 `source/utils/full_chip_imaging_compare.cpp`。

对照包含原始 `target` 和 CTM 最佳轮的局部灰度版图。`compare_micro_imaging()` 接收 `PartitionMeta`、
优化时已经准备的 `ImagingCache`、全局 target 和完整 `local_masks`；不会再初始化 1024×1024 光学系统。
target 按原分区规则提取周期 Halo；优化 mask 保留每块独立的 Guard/Halo，不从回拼 mask 重新提取。
各 Micro 在完整局部范围内成像，然后只取 Core 回拼 AI 和连续 WI，无重叠平均。
当前 64 Micro 使用同一套 256×256 光学缓存，输出仍为 1024×1024、8 nm/像素。
沿用 `parallel_micro` 并行计算各块，每个线程独占成像缓冲区，最后串行拼接。

输出直接放在 `imaging_comparison/`：`target_mask.txt`、`optimized_gray_mask.txt`、`target_ai.txt`、`target_wi.txt`、
`optimized_ai.txt`、`optimized_wi.txt`。全部保留 double 数值，优化灰度 mask 和 WI 都不做二值化。
`metrics.csv` 只记录两个全图 PE，均使用原 `Loss::pe_loss(WI, target)`，不包含 CEPT。

`sampling.csv` 中 `imaging_mode=micro_core`，`grid_size` 为局部光学网格尺寸；
`image_rows/image_cols` 为回拼矩阵尺寸，另记录 Micro 数、光源点数及相干核数。
`imaging_source_grid_size` 只用于独立整图评价工具，本 demo 的最终成像直接复用优化缓存。

绘制 demo 已导出的 Micro 成像结果：

```bash
MPLCONFIGDIR=/tmp/litho-matplotlib .venv/bin/python scripts/plot_full_chip_imaging_compare.py result/full_chip_ctm/contact_1_0_only_1024/imaging_comparison
```

绘图只保存 `gray_ctm_comparison.png` 和中心 256 像素窗口的 `gray_ctm_comparison_zoom.png`。
上排为 target、下排为优化灰度版图，三列依次为输入 mask、Core 回拼 AI、Core 回拼连续 WI。

原 `demo_full_chip_imaging_compare [config.yaml] [mask.txt] [output_dir]` 保留为独立整图直接成像工具。
它按保存的全局 mask 重建整图光学系统，输出 `imaging_mode=full_frame`，与本 demo 的独立 Guard 局部成像口径不同。
后两个相对路径相对于工作目录；评价旧实验时使用该实验的 `config_used.yaml`，输出目录应另指定以保留 Micro 结果。

二值化在最佳轮的全图 Core 回拼后执行，使用原 C++ 接口；当前全芯片入口参数为
`Imbinarize::binarize_advanced(global_mask, global_target, 0.05, 0.05, 5, 5)`。
沿用原接口的峰值自适应阈值、小连通域过滤、SRAF 提取和最小间距约束，输出直接取 `final_mask`。
原单块 `demo_CTM` 的参数为 0.1、0.15，面积/间距默认 50/15 像素；当前全芯片入口保留用户设置的 5/5。

## 验证

```bash
cmake --build build-release --target test_full_chip_ctm test_micro_gradient_exchange --parallel 4
OMP_NUM_THREADS=4 OMP_MAX_ACTIVE_LEVELS=2 ctest --test-dir build-release -R '^(full_chip_ctm|micro_gradient_exchange)$' --output-on-failure
```

测试覆盖 Abbe/SOCS 单块原接口对照、非零 CEPT 下的 Guard-only 两轮独立轨迹、Core + Guard 更新、
外围 theta 与初值完全相同、最佳状态恢复、四种损失和融合开关、串行/并行一致性及零次更新。
轨迹测试故意传入 `cept_guard_only=false`，确认全芯片 CTM 仍强制采用 Guard-only 规则。

成像对照另有 `test_full_chip_imaging_compare`，验证完整/压缩 SOCS 的特征值和 AI/WI 一致，
以及整图接口保留灰度输入、Micro 串并行一致、非整除/非方形 Core 回拼、独立 Guard 参与成像，
和 Core 边界 EP 使用所属块完整 AI 求斜率。运行方式为：

```bash
cmake --build build-release --target test_full_chip_imaging_compare --parallel 4
ctest --test-dir build-release -R '^full_chip_imaging_compare$' --output-on-failure
```

下列历史结果采用修改前的双端 CEPT 和外围 Halo 刷新，不能与当前 Guard-only、冻结 Halo 的数值直接比较。
2026-10-04 实际执行了两张 1024×1024 版图，4 线程、64 Micro、20 次更新、步长 0.05，PE + 1.1×CEPT、关闭融合：

| 输入 | 初始总 loss | 第 20 次更新后总 loss | 优化循环耗时（含初始及末轮评价） |
| --- | ---: | ---: | ---: |
| metal_3_0_only_1024 | 26975.42 | 6296.86 | 31.74 s |
| contact_1_0_only_1024 | 57172.22 | 10839.42 | 32.02 s |

两组 history 均包含 0～20 的 21 个状态；灰度/二值 BMP 和 TXT 尺寸均已核对，灰度值有限且位于 [0,1]。
这些数值来自局部成像 Core loss 的汇总，不代表整图直接成像的 EPE 或二值化后的误差。
