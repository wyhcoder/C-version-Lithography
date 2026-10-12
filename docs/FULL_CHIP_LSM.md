# 全芯片 LSM：CTM 二值化之后的水平集优化

声明在 `include/optimizer/full_chip_lsm.h`，实现在 `source/optimizer/full_chip_lsm.cpp`。
运行入口为 `demo/demo_full_chip_lsm.cpp`，独立配置为 `config_full_chip_lsm.yaml`。

## 运行

CTM 已有的 `binary_mask.txt` 是 LSM 的输入，目标仍为原始 target，不把 CTM 结果当作目标。
默认版图由 `config_full_chip_lsm.yaml` 指定。由已有 CTM 结果接着运行，无须再次执行 CTM：

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target demo_full_chip_lsm --parallel 4
OMP_NUM_THREADS=4 ./build-release/demo_full_chip_lsm config_full_chip_lsm.yaml
```

支持 `[config.yaml] [iterations] [binary_mask.txt] [output_dir]`，路径相对于 YAML 所在目录。
例如运行 20 次更新：

```bash
OMP_NUM_THREADS=4 ./build-release/demo_full_chip_lsm config_full_chip_lsm.yaml 20
```

换版图时同时修改 `target_path`、`initial_mask_path`、`output_dir` 和 `epe_sampling`；光学参数应与生成 CTM 输入的实验一致。

## PE/EPE 梯度量级诊断

独立入口 `demo_full_chip_gradient_stats` 在 LSM 的第 0 轮分别评价纯 PE 和纯 EPE，不更新 phi，不改变优化配置：

```bash
cmake --build build-release --target demo_full_chip_gradient_stats --parallel 4
OMP_NUM_THREADS=4 ./build-release/demo_full_chip_gradient_stats config_full_chip_lsm.yaml
```

支持 `[config.yaml] [output_dir]`；默认输出在配置中的 LSM 结果目录下 `gradient_stats_initial/`。
复用实际 Micro 分区、局部 CTM 二值输入、原 SDF 初始化、mask 渲染、PE 背景窗口及 EPE 选点接口。
诊断采用 PE_EPE 的两个分项，分别设为单位权重提取背景 PE 与主图 EPE，不含背景惩罚或 CEPT。
分别保留融合前梯度 `raw` 与按配置进行 Guard 融合后的梯度 `fused`；此处背景 PE 不等同于纯 PE 类型的全区域 PE。
诊断直接使用当前 PE/EPE mask 梯度接口，不归一化；本诊断测量数值尺度，不验证它是否为统计 EPE 的严格导数。

`gradient_stats.csv` 含每块和全体汇总（`micro_id=-1`）的 RMS、平均绝对值、P99 绝对值、最大绝对值和方向余弦。
三个统计区域为完整 `CoreHalo`、更新窗口 `CoreGuard`、更新窗口内 `|phi|<=pixel_size_nm` 的 `Boundary`。
统计使用同一组 mask 像素，计入零值；RMS 为 `sqrt(sum(g*g)/count)`，P99 使用最近秩。
全体汇总按 Micro 独立驱动统计，Guard/周期重叠像素允许重复；没有先把梯度回拼成全局矩阵。
`mixed_*` 列仅表示假设启用混合损失时配置的 `pe_weight/epe_weight` 产生的贡献；当前 `PEPenalty` 中 EPE 有效系数仍为 0。
`linearity_check.csv` 核对拆项加权与原 `PE_EPE` 评价一致，包括 Guard 融合前后；`config_used.yaml` 保存输入和参数快照。

2026-10-10 **修正 EPE 梯度前**，contact 1024×1024、64 Micro、局部 256×256、8 nm/像素、Gray 4 倍渲染的初始状态：

| 统计范围 | PE RMS | EPE RMS | EPE/PE |
| --- | ---: | ---: | ---: |
| 融合前 Core+Guard | 1.986166 | 11.693101 | 5.887274 |
| 融合后 Core+Guard | 2.009956 | 11.694297 | 5.818185 |
| 融合后 Boundary | 2.243218 | 13.837031 | 6.168384 |

融合后 Boundary 的 PE/EPE 平均绝对值为 1.563008/0.261783，P99 为 6.338083/1.148136，最大绝对值为 11.175249/1878.181831。
EPE RMS 主要受到局部峰值影响；Micro 0 和 56 占该边界带 EPE 梯度平方和的 92.901%。
假设混合权重为配置中的 PE=2、EPE=0.5，边界带加权 RMS 分别为 4.486436/6.918516，比值为 1.542096。
拆项与原混合计算的融合前误差为 0，融合后最大绝对误差小于 1e-13；390 行 CSV 的样本数及 RMS/均值/最大值汇总独立核对通过。
这是第 0 轮梯度诊断，没有执行更新，不能代表后续全部迭代的梯度比例。

同日修正平方 EPE 驱动，补入 `dx^2` 和光学反传系数 2 后，相同输入的融合后 Boundary RMS 为
PE **2.243218**、EPE **1771.140012**，EPE/PE 为 **789.553185**；EPE 最大绝对值为 **240407.274432**。
完整新统计位于 `result/full_chip_lsm/contact_1_0_only_1024/gradient_stats_initial_squared_epe_dx2/`，
修正前的 `gradient_stats_initial/` 保留作为对照。390 行统计已确认 PE 不变、EPE 的各幅值统计均变为旧值的 `2*8^2=128` 倍。
新增 Abbe/SOCS 冻结分母平方 EPE 的 mask 有限差分检查，覆盖 `dx=1/4/8`、重复 EP 和 EP 权重；
梯度交换、全芯片 CTM、LSM 共 3 项 CTest 通过。本次仍未加入梯度归一化，也没有执行完整优化。

## phi 到灰度 mask

在 `full_chip_lsm` 中选择渲染方式，当前 YAML 已启用与 Python 相同的子像素块平均：

```yaml
  levelset_update_mask_type: gray # binary 则使用原像素 phi >= 0。
  levelset_phi_renderer_scale: 4  # 每方向放大倍数，须为正整数。
```

实现声明在 `include/utils/levelset_mask_renderer.h`，实现在 `source/utils/levelset_mask_renderer.cpp`。
`Gray` 模式先把完整局部 phi 转成 float32，用 OpenCV `INTER_LINEAR` 双线性插值放大，
然后按 `phi_highres >= 0` 得到子像素二值图，每个 `scale × scale` 块平均回原网格。
放大倍数为 4 时，一个原像素中有 16 个子像素：若其中 10 个在图形内，mask 值为 `10/16=0.625`。
灰度是零水平集图形的覆盖率，不是 sigmoid，也不是独立优化的 CTM 灰度变量。
渲染后矩阵尺寸、光学 FFT 网格和 nm/像素均不变；`scale=1` 退化为像素中心的硬二值化。

**同一个渲染 mask 实际用于成像、数据损失、完整 Halo 的光学反传和 Guard CEPT**，不只用于图片展示。
仍将 `∂loss/∂mask` 作为原 C++ 水平集 PDE 的法向速度；不通过硬阈值 renderer 求 `∂mask/∂phi`。
RK2 和 SDF 重初始化只写 Core + Guard 内的 phi，窗口外 phi 保持初始值。
灰度像素依赖双线性插值的相邻 phi，因此紧贴更新窗口外侧的覆盖率可能随窗口内 phi 改变，这不等于窗口外 phi 被更新。
最终先渲染每块完整局部 phi，再只回拼 Core；不在拼好的全局 phi 上重新渲染，以保留各块自己的 Halo 上下文。
库接口默认 `Binary`，项目 YAML 当前选择 `gray`；改回 `binary` 可使用原来的渲染流程。

## 损失选择与 EP 选点

在 `full_chip_lsm` 中设置 `loss_type`。只支持四种模式，CEPT 由独立开关控制：

| loss_type | data_loss |
| --- | --- |
| `PE` | `pe_weight * PE-all`，主图形和背景全部像素 |
| `EPE` | `epe_weight * EPE`，原 target 主图轮廓 EP |
| `PEPenalty` | `pe_weight * PE-main + penalty_weight * BG_raw`，未膨胀主图 PE 与原背景惩罚 |
| `PE_EPE` | `pe_weight * PE-bg + epe_weight * EPE`，膨胀主图内 EPE 与外部背景 PE |

LSM 的评价范围固定为 **Core + Guard**（含角区）；仅 `PE_EPE` 按膨胀主图分区：

```text
W = Core + Guard 的完整矩形窗口（含角区）
M = dilate(target > 0.5, 3×3 全 1 邻域, main_region_dilation_pixels 次) ∩ W
B = W \ M
EPE = 原 C++ EPE 接口在 M 内的 EP 上求和
PE = Σ[B] (WI-target)^2
BG_raw = Σ[W 且 target<0.5] max(WI-penalty_threshold,0)^2，独立于膨胀，仅 PEPenalty 优化它
```

PE_EPE 先膨胀完整局部 target，再与 W 相交；膨胀带属于 EPE 区域，不再计入背景 PE。
`main_region_dilation_pixels: 2` 与 Python 默认一致，每次膨胀向外扩 1 像素，包含对角方向；0 表示不膨胀。
EPE 仍只在原始目标轮廓选出的 EP 上评价，膨胀用于划分区域，不把膨胀后的边界重新选成 EP。
PEPenalty 的 PE 只评价未膨胀主图形，背景只加阈值平方惩罚，不额外加一份普通背景 PE。
所有启用项的配置权重分别乘入损失和梯度，单项也生效，不自动归一化；PE/背景惩罚共用一次光学反传。
例如使用背景 PE 和主图 EPE 的加权组合：

```yaml
  loss_type: PE_EPE
  pe_weight: 0.5
  epe_weight: 0.5
  main_region_dilation_pixels: 2
  epe_sampling: contact
  epe_all_eps: true
```

EP 始终从原始 **全局 target** 选取，随后按各块区域转换为局部像素坐标；不在裁切出来的轮廓上重新选点。

- `epe_sampling: contact`：调用原 `select_eps_via(0)`，仅保留 `weight_epe=1` 的点，即每个矩形 contact 的四边中点。
  `r=0` 本身只控制角点偏移，原函数还返回角点和其他边上采样点，所以需要这次筛选。
  边长小于 4 像素的片段仍按原函数跳过；芯片边界裁断的 contact 不补造四个点，实际 EP 总数以 CSV 为准。
- `epe_sampling: metal`：调用原 `select_eps_others(epe_interval_line, epe_interval_corner)`，保留全部返回坐标。
  两个间隔单位都是像素，仅 metal 使用。

`epe_all_eps: true` 在选点结束后执行 `selected.weight_epe.setOnes()`，与 MEEF 的 `wepe_all_eps` 操作一致。
LSM 默认开启，metal 图中原先蓝色的点也与红色点一样参与 EPE 梯度；不改变选点坐标、点数或 MEEF 权重。
contact 先筛选四边中点，再统一权重，因此不会恢复已经排除的角点和其他边上点。
设为 `false` 则保留原选点函数返回的 EPE 权重。
MEEF 根配置目前 `interval_line=5`，LSM 为 6；同一 target 的点位置要完全相同，还需使用相同采样间隔。

EPE 数值继续使用 `Loss::epe_loss`，即各 EP 的 `|AI-threshold|/|∇AI|*pixel_size_nm` 之和，单位 nm。
原 `Loss` 接口对传入的全部点求和，不读取 `weight_epe`；原 `Gradient::epe_gradient` 读取该权重。
因此统一权重会改变 EPE 更新驱动，不改变同一个 mask 上报告的 EPE 或 Core 点数。
仅在 `epe_all_eps=false` 时，metal 的零权重点参与报告 EPE、但不参与 EPE 梯度。
contact 筛选后全部四边中点权重为 1，其他点不参与损失或梯度。
EPE 驱动采用平方 EPE 的冻结光强斜率近似，含物理尺度 `dx^2` 和强度光学反传的系数 2；报告仍为绝对 EPE。

## 每轮流程

1. 将原 C++ 二值化得到的全局 0/1 mask 提取为各块 Core + Halo，用 `LevelSetUtils::reinit_sdf()` 初始化 phi，再按配置渲染局部 mask。
2. 冻结所有 Micro 的局部状态，并行完成 Core + Halo 成像；在 Core + Guard 内按 YAML 选择的四种损失语义评价，反传完整局部 mask 梯度。
3. 等待所有 Micro 完成，再串行交换四邻域条带：CEPT 比较本块 Guard 与邻居 Core 的 mask，邻居 Core 作为冻结参考，只给本块 Guard 加梯度；按开关融合 Guard 数据梯度。
4. 将总梯度作为法向速度，乘 Core + Guard 更新窗口；各 Micro 并行计算 WENO/Godunov、曲率项和自己的 CFL dt。
5. 按原 C++ 两阶段 RK2 推进 phi，两阶段使用同一轮梯度和第一阶段的 dt，仅更新 Core + Guard。
6. 每到重初始化间隔，使用原 C++ SDF 接口重新计算距离，仅提交 Core + Guard。所有块完成后统一提交新 phi。
7. 按 Binary/Gray 配置渲染局部 mask，不从邻居或 Core 回拼图刷新 phi。Core + Guard 窗口之外的 phi 始终保留本块初始化时的值。
8. 评价更新后的全部局部状态，记录历史最佳轮。最终只回拼同一最佳轮的 Core，返回 mask 和 phi。

调度顺序参照 Python `ilt_macro_serial.py` 的 LEVEL_SET 阶段和 `levelset_micro_parallel.update_micro()`。
数值部分采用当前 C++：内正外负的 SDF、WENO/Godunov、曲率和 RK2；Gray 渲染采用 Python 的双线性放大和二值块平均。
窗口外 phi 不更新，Guard 始终保留各块独立状态；这两条规则与 Python 当前 staged LSM 一致。
仅 PE_EPE 的膨胀区域划分与 Python 的 `_build_levelset_objective_regions` 一致；其他类型按用户指定的窗口评价。
PE/背景窗口乘在输出误差信号上，然后经过光学伴随；反传后不再把 PE 梯度裁成背景窗口。
因此主图形和外围 Halo 的 mask 像素也可以通过光学传播影响背景损失，其梯度仍保留；演化时只使用 Core + Guard 内的总梯度。
EPE 由 `epe_sampling` 控制原 C++ 全局选点，PE_EPE 再用膨胀窗口筛选；包含 Guard 重叠点与周期副本。
CEPT 仍单独评价 Guard 与邻居 Core 的 mask 差异；CEPT 和梯度融合只处理四侧有实际邻居的 Guard 条带，不处理角区。
Python 的 TV、进程池/MPI 参数不属于此接口；C++ 使用 OpenMP，原单块 LSM 接口的渲染规则保持原样。
全芯片场景下关闭 WENO 的嵌套行/列并行；单块运行时仍可使用其内部并行。

`dx=dy=pixel_size_nm`，当前为 8 nm；phi 的长度单位为 nm。原 SDF 实现采用各向同性网格。
`cfl` 决定稳定步长，`curvature_weight` 对应原单块 LSM 的 b；`reinit_interval=0` 关闭重初始化。
固定执行指定更新次数，不额外加入提前停止或线搜索。

## C++ 接口

```cpp
FullChipLSMOptimizer optimizer(meta, cache, threshold, alpha);
FullChipLSMOptions options;
options.iterations = 200;
options.dx = options.dy = pixel_size_nm;
options.mask_type = LevelSetMaskType::Gray; // Binary 保留原像素中心的 0/1 渲染。
options.render_scale = 4; // 每方向 4 个子像素，覆盖率步进为 1/16。
options.gradient_options.loss_type = MicroLossType::PE_EPE;
options.gradient_options.pe_weight = 0.5;
options.gradient_options.epe_weight = 0.5;
options.gradient_options.main_region_dilation_pixels = 2; // 单位：像素；膨胀后划分主图形与背景评价区域。
options.gradient_options.epe_sampling = MicroEpeSampling::Contact; // metal 改成 Metal。
options.gradient_options.epe_all_eps = true; // 全部已选 EP 参与 EPE 梯度，与 MEEF 的 wepe_all_eps 同义。
options.gradient_options.epe_pixel_size = pixel_size_nm;
options.gradient_options.enable_cept = true;
options.gradient_options.enable_grad_fusion = true;
auto result = optimizer.optimize(ctm_binary_mask, global_target, options);
```

`meta/cache` 在外部初始化，须比 optimizer 存活更久，且 `cache.N == meta.tile_size`。
`Grid(N, pixel_size_nm, NA, wavelength_nm)` 直接使用输入的 N，不扩成 2 的幂或额外加一，自动处理奇偶尺寸。
单块入口的 N 来自 `system.grid_size_pixel`；全芯片优化和最终对比都使用 `meta.tile_size` 的局部成像缓存。
空间像素中心为 `(i-(N-1)/2.0)*pixel_size_nm`，原点位于几何中心；偶数时处于像素之间，奇数时落在中心像素上。
中心化频率为 `(i-N/2)/(N*pixel_size_nm)`，其中 N/2 是整数除法，再归一化到 NA/λ；奇偶尺寸都包含零频。
数据项和梯度采用同一损失选项；CEPT 比较当前各块 mask 像素，而非梯度或 phi 数值。
LSM 内部固定 `loss_region=CoreGuard`、`cept_guard_only=true`、`collect_core_metrics=true`，窗口完全由 `loss_type` 决定。
全芯片 CTM 和 LSM 均固定 Guard-only CEPT；共享模块仍保留独立调用时的可选双端规则。
梯度融合改变数据更新方向，不修改 loss 标量；EPE 沿用原 C++ 梯度公式。
曲率是 PDE 的平滑项，不计入用于最佳轮选择的 `data_loss + cept_weight * cept_loss`。

## 输出

启动日志打印实际 mask 渲染方式和倍数、损失类型、评价区域、膨胀次数、梯度范围、更新窗口以及各项权重。
每轮仅输出一行对齐的表格：

```text
LSM    Iter   PE-main(raw)        BG(raw)     Total-loss
```

- `Total-loss`：本轮全部 Micro 的加权总损失，用于历史最佳轮选择。
- `PE-all(raw)` / `PE-main(raw)` / `PE-bg(raw)`：分别对应 PE、PEPenalty、PE_EPE 的原始 PE，只展示所选类型的一列。
- `EPE(raw)`：所选 EPE 窗口内 EP 的绝对 EPE 总和，未乘 EPE 系数，仅 EPE/PE_EPE 展示。
- `BG(raw)`：未膨胀背景的原始阈值平方惩罚，仅 PEPenalty 展示。
- `CEPT(raw)`：本块 Guard 与四邻域 Core 的 mask 平方差总和；仅开启 CEPT 时展示。
- `EPE-mean`：`EPE(raw) / 实际评价 EP 数`，仅 EPE/PE_EPE 展示，含 Guard 重叠点和周期副本。

每轮终端仅显示所选类型包含的原始分项；其他诊断值仍保存在 CSV，不能当成当前优化项。
例如 PEPenalty 满足 `Total-loss = pe_weight*PE-main(raw) + penalty_weight*BG(raw) + 可选的 cept_weight*CEPT(raw)`。
每轮日志不打印 Core 指标或 CFL 步长。

背景惩罚为 `penalty_weight * BG_raw`，仅 PEPenalty 加入；该模式使用 pe_weight 和 penalty_weight，不使用 epe_weight。
梯度融合只改变更新驱动，曲率只用于 PDE 平滑，两者不另外加入 `total_loss`。

`lsm_mask.txt/bmp` 保存最佳轮实际渲染 mask 的 Core 回拼结果：Gray 时保留覆盖率，Binary 时为 0/1；`lsm_phi.txt` 保存对应 phi。
`lsm_binary_mask.txt/bmp` 额外保存同一最佳轮 `phi >= 0` 的硬二值 Core 回拼图，未用它重新选择最佳轮或替代灰度成像输入。
TXT 保留原矩阵 double 值，BMP 将 `[0,1]` 映射到 `[0,255]` 并取最近整数。
`initial_mask.txt/bmp` 保留输入的 CTM 二值版图，`initial_phi.txt` 保存其局部 SDF 的 Core 回拼。
`initial_rendered_mask.txt/bmp` 保存第 0 轮实际用于成像和损失评价的渲染 mask。
`history.csv` 保留原六列及四列已加权分项 `pe_contribution`、`penalty_contribution`、`epe_contribution`、`cept_contribution`。
满足 `data_loss = pe_contribution + penalty_contribution + epe_contribution`，`loss = data_loss + cept_contribution`。
原 `cept_loss` 列继续保存未加权值；记录覆盖 0～iterations 的所有状态。
另追加 `total_pe`、`total_bg_penalty_raw`、`total_epe_sum_nm`、`total_ep_count`、`epe_mean_nm`，对应每轮日志的原始总值和点数。
其中 EPE 均值即 `total_epe_sum_nm / total_ep_count`；即使 EPE 系数为 0，也报告真实 EPE。
`summary.csv` 区分最佳轮和末轮，`config_used.yaml` 保存实际更新次数及输入/输出路径。
返回的 `local_masks/local_phi` 保留最佳轮的完整独立局部状态。

### Core 区域统计

优化使用 **Core + Guard 上的所选数据项**；已有 `core_*` 指标作为独立诊断保留在 CSV，不重复打印到每轮日志。
各 Core 是不重叠的全局半开区间 `[y0,y1) × [x0,x1)`；EP 落在分界线上时由包含该像素的唯一 Core 统计。
Core 统计没有 Guard 周期副本，也没有 CEPT，所以汇总时不会重复计入相邻块的重叠部分。
Core PE 使用所选类型的 PE 窗口；背景惩罚始终统计未膨胀背景，EPE 只取评价窗口内属于 Core 的 EP。
每次局部成像后直接复用该 AI/WI，先在完整局部 AI 上计算 EPE 空间导数，再筛选 Core 所属 EP。
不能先把 AI 裁成 Core 后求导，否则 Core 边界上的差分会改变。

无论当前选择哪种优化损失，每轮都统计以下全部指标，追加到 `history.csv`：

| 列名 | 含义 |
| --- | --- |
| `core_pe` | 所选 PE 窗口与各 Core 交集内的原始 PE；纯 PE 模式包含全部 Core 像素 |
| `core_bg_penalty_raw` | Core 内未膨胀背景的 `BG_raw`，不乘 `penalty_weight` |
| `core_pe_penalty` | `pe_weight * Core原主图PE + penalty_weight * core_bg_penalty_raw`，独立 PEPenalty 诊断 |
| `core_epe_sum_nm` | 唯一归属的全局 EP 在其所属 Micro 成像上的原 EPE 之和，单位 nm |
| `core_ep_count` | 主图形窗口内唯一归属于 Core 的全局 EP 数；原始选点另存 `ep_points.csv` |
| `core_epe_mean_nm` | `core_epe_sum_nm / core_ep_count`；没有 EP 时为 0 |
| `core_data_loss` | 选定模式只在 Core 上的加权数据损失，不含 CEPT |

Core EPE 均值同样按 **Core 总 EPE / Core 总点数** 计算，不平均各 Micro 的均值。
`total_*` 包含 Guard 重叠区域，`core_*` 不包含；最佳轮按实际优化目标 `分区 data_loss + 加权 CEPT` 选择。
`core_metrics_best.csv` 保存最佳轮每个 Micro 的 Core 范围和上述指标，可以定位各块损失。
它们来自各块局部成像。最终对比同样使用这些局部模型，但对比 PE 在全部 Core 像素统计，
Core 日志中的 PE 只在背景窗口统计；两者评价区域不同。

### 保存 EP 选点图

入口在优化开始前自动保存：

- `ep_points.csv`：实际全局 EP 的 `(y,x)` 像素坐标、所属 `micro_id`、生效的 EPE 权重及 MEEF 权重。
- `core_regions.csv`：Core 的全局半开区间，用于叠加分区线。
- `ep_points.png`：原始 target 上的全图 EP，青色线表示 Core；默认全部 EP 的权重为 1，统一显示红色。
  `epe_all_eps=false` 时仍可用蓝色显示原零权重点；红蓝两类都是 EP，颜色只表示实际 EPE 梯度权重。
- `ep_points_zoom.png`：靠近中心 EP 的 128×128 像素窗口，仍显示全局坐标；点数少时标注 `(y,x)`。

Python 仅绘制 C++ 导出的点，不另外计算或选择 EP。可以单独重画：

```bash
MPLCONFIGDIR=/tmp/litho-matplotlib XDG_CACHE_HOME=/tmp/litho-matplotlib .venv/bin/python scripts/plot_full_chip_eps.py result/full_chip_lsm/contact_1_0_only_1024
```

## 原始版图与 LSM 最佳版图的 Micro 成像对比

`demo_full_chip_lsm` 结束后自动对原始 `target` 的各块和最佳轮完整 `local_masks` 分别成像。
复用优化时已准备的 Micro 光学缓存，当前为 256×256；不会额外初始化 1024×1024 光学模型。
优化 mask 保留各块独立 Guard/Halo，各块成像后只将 Core 的 AI/WI 写入 1024×1024 全局矩阵。
沿用 `parallel_micro` 并行成像，串行回拼；两组输入使用相同局部光学和显影参数。
`imaging_source_grid_size` 仅供独立整图直接成像工具使用。
WI 保留 sigmoid 连续值；LSM 成像输入直接使用最佳轮实际渲染 mask，Gray 模式保留覆盖率。

输出到当前 LSM 结果目录的 `imaging_comparison/`：

| 文件 | 含义 |
| --- | --- |
| `lsm_comparison.png` | 全图对比，上排未优化 target，下排 LSM 最佳版图；三列为 mask、AI、WI |
| `lsm_comparison_zoom.png` | 同一对比的中心 256×256 像素窗口，切片区间不包含结束索引 |
| `target_mask.txt` / `optimized_lsm_mask.txt` | 原始 target 和最佳局部 mask 的 Core 回拼预览，保留 double 原值 |
| `target_ai.txt` / `optimized_ai.txt` | 各 Micro 局部空中像的 Core 回拼结果 |
| `target_wi.txt` / `optimized_wi.txt` | 各 Micro 连续显影 WI 的 Core 回拼结果 |
| `metrics.csv` | 两个全图 PE，均相对原始 target 计算，不含 CEPT |
| `sampling.csv` | `micro_core` 模式、局部 FFT 尺寸、全局输出行/列数、Micro 数、照明采样及 nm/像素 |

成像和 PE 由 C++ 完成，入口使用项目 `.venv/bin/python` 调用绘图脚本保存 PNG；绘图依赖 NumPy/Matplotlib。
对比 PE 在全部 Core 像素上统计，不重复计入 Guard/Halo；与迭代日志的分区加权损失/CEPT 分别保存。

### 每个 EP 的 EPE

`demo_full_chip_lsm` 完成最佳局部 mask 的成像后，自动在结果目录保存 `ep_epe.csv`。
使用原 `ep_points.csv` 的全部坐标，调用现有 `Loss::evaluate_epe()`，不重新选点或改变 EPE 公式。
每行保留 `ep_id,y,x,micro_id,weight_epe,weight_meef`，追加：

- `aerial_intensity`：该点在所属 Micro 最佳局部 mask 的完整 AI 上的光强。
- `epe_nm`：该点的未加权绝对 EPE，单位 nm；没有乘入 EPE 优化权重或 EP 权重。

`y/x` 是从 0 开始的全局像素行/列索引。一个 EP 只输出一行，无 Guard 重叠点或周期副本。
空间斜率在所属 Micro 的完整 AI 上计算，保留 Core 边缘的 Guard/Halo 光强，不在回拼 AI 的接缝上差分。
每个点只有一个 Core 归属，因此不包含迭代日志 `total_*` 中的 Guard 重叠副本。
声明在 `include/utils/full_chip_epe.h`，实现在 `source/utils/full_chip_epe.cpp`。

独立工具查看已有逐点结果，无需重新优化或成像：

```bash
cmake --build build-release --target demo_full_chip_epe --parallel 4
OMP_NUM_THREADS=4 ./build-release/demo_full_chip_epe result/full_chip_lsm/contact_1_0_only_1024
```

`micro_core` 结果直接读取 demo 已导出的 `ep_epe.csv`，汇总点数、均值和最大值；
旧整图直接成像结果仍按保存的 `optimized_ai.txt` 和 `config_used.yaml` 重新导出逐点 EPE。

切换 Micro 回拼前，2026-10-10 对已有 contact 1024×1024 整图成像结果（最佳轮 98）导出 3706 个 EP，平均 EPE 为 1.340828 nm，
最大 EPE 为 56.858976 nm；全部逐点值已与独立 NumPy 空间差分核对，未重新运行该版图的优化。
另用 5×6 解析线性光强场核对坐标行序、8 nm 单位换算及零权重点的未加权输出；
64×64 contact 的 0 次更新入口验证自动生成了 4 个 EP 的 CSV。全芯片 LSM 和全图成像两项测试通过。

以下工具保留为独立整图直接成像入口，不能从保存的全局 mask 恢复各块独立 Guard。
如需对历史结果补整图成像，应单独指定输出目录，避免覆盖当前 Micro 回拼结果：

```bash
cmake --build build-release --target demo_full_chip_imaging_compare --parallel 4
OMP_NUM_THREADS=4 ./build-release/demo_full_chip_imaging_compare result/full_chip_lsm/contact_1_0_only_1024_validation20_pe_epe/config_used.yaml result/full_chip_lsm/contact_1_0_only_1024_validation20_pe_epe/lsm_mask.txt /tmp/litho_full_frame_compare
MPLCONFIGDIR=/tmp/litho-matplotlib XDG_CACHE_HOME=/tmp/litho-matplotlib .venv/bin/python scripts/plot_full_chip_imaging_compare.py /tmp/litho_full_frame_compare
```

## 验证

```bash
cmake --build build-release --target test_full_chip_lsm test_levelset_mask_renderer test_micro_gradient_exchange test_level_set_godunov --parallel 4
OMP_NUM_THREADS=4 OMP_MAX_ACTIVE_LEVELS=2 ctest --test-dir build-release -R '^(full_chip_lsm|levelset_mask_renderer|micro_gradient_exchange|level_set_godunov)$' --output-on-failure
```

`test_levelset_mask_renderer` 核对原 Python renderer 生成的非方形样例、1/2/4 倍采样、直线覆盖率、零点和边界。
`test_full_chip_lsm` 在 Binary/Gray 下用原单块数值接口独立执行轨迹，核对 Abbe/SOCS 的 RK2、CFL 和重初始化；
另核对窗口外 phi 与初始局部 SDF 完全相同、独立 Guard、四种损失、两种融合模式、串行/并行一致性，以及返回状态能复现最佳 loss。
`test_micro_gradient_exchange` 独立按像素邻域核对膨胀、互斥的 PE/EPE 窗口和先膨胀再截窗口，
检查背景损失的完整局部梯度有限差分、Guard/周期 EPE 选点及 Guard-only CEPT。
`test_full_chip_lsm` 确认 LSM 固定 Core + Guard 评价，纯 PE 与原单块全区域 PE 的完整 RK2 轨迹对照。
新增覆盖 contact 四边中点、metal 原选点坐标/权重、分界 EP 的唯一 Core 归属、逐像素独立 Core 统计、
PEPenalty 主图 PE/背景惩罚独立系数的有限差分，以及 PE_EPE 与原 EPE 驱动的加权组合一致性。
日志总量口径另核对全部 Micro 的区域分项、实际评价 EP 数与 EPE 均值，并保留唯一 Core EP 的诊断；
在纯 PE 模式下核对 EPE/背景仍正常统计，并验证这些未启用项不改变实际损失或梯度。

### 2026-10-09 灰度渲染验证

渲染器、全芯片 LSM、梯度交换、全芯片 CTM、WENO/Godunov 共 5 项测试通过。
原 Python renderer 的非方形样例在 1/2/4 倍采样下与 C++ 输出完全一致。
使用当前 contact 配置（PE/EPE/BG/CEPT 权重为 1/0.8/2/1，膨胀 1 次、CFL=0.9），
完成 1024×1024、64 Micro、4 线程、Gray 4 倍渲染的 2 次更新入口验证，最佳轮为 2。
总损失从 1004258.244838 降至 514784.469477；输出含 133260 个非 0/1 像素，取值均为 1/16 的整数倍。
已核对最佳 phi 的硬二值输出、BMP 灰度量化、加权分项和全图成像输入；各 Core 内排除一圈边界后的
1016064 个像素与 Python renderer 完全一致，分界像素按各块自己的 Guard 上下文渲染。
结果位于 `/private/tmp/lsm_gray_20261009_ScJb9R/result/`，实际配置见 `config_used.yaml`。
这是 2 次更新的流程验证；全图 PE 为 target 53547.167659、LSM 102214.014249，
这次短跑的全图成像指标尚未优于原始 target，不能据此宣称灰度模式改善了最终优化效果或已收敛。

2026-10-06 主图形/背景分区改动通过梯度交换、全芯片 LSM、全芯片 CTM 三项测试。
使用当时的 contact 二值渲染配置（权重 PE/EPE/BG/CEPT 为 0.5/0.8/0.5/1.0），完成 1024×1024、64 Micro、4 线程的一次更新入口验证：
总损失从 347717.248538 降至 347694.518577，最佳轮为 1，优化循环耗时约 6.95 s。
已独立核对分区内实际评价 EP 数为 7846（含 Guard 重叠/周期副本），唯一 Core EP 数为 3706，
并检查加权分项、EPE 均值、phi/mask 一致性及全图成像输出。
结果位于 `/private/tmp/lsm_split_regions_20261006_9Fmpw3/result/`，实际配置见其中 `config_used.yaml`。
这是一次更新的流程验证，没有执行默认 100 次完整优化。

### 历史验证（主图形/背景分区改动前）

下列 2026-10-06 的历史版图实验采用修改前未区分主图形/背景的 Core + Guard 数据损失，与当前分区损失不能直接比较。
另验证 `epe_all_eps` 不改变坐标、数量或 MEEF 权重，新增梯度等于原蓝色 EP 的原 C++ 梯度贡献。

开启 `epe_all_eps=true` 后，用真实 metal 1024×1024、64 Micro、4 线程执行 2 次更新。
全部 11482 个 EP 的 EPE 权重均为 1，其中包括原先权重为 0 的 2440 个点；已逐点核对与旧导出坐标完全相同。
输出在 `result/full_chip_lsm/metal_3_0_only_1024_all_eps_check_20261006_3lf8vhge/`，包含新的选点全图/放大图、CSV 和实际配置。
三项混合及 CEPT 系数为 `0.5/0.5/1.0/0.5`，最佳轮为第 2 次更新，Core EPE 均值为 15.942500 nm。
这是两次更新的运行验证，不代表完整优化或收敛。

本次三项混合/Core 统计改动通过 LSM、梯度交换、CTM、WENO/Godunov 四项测试。
另用真实 1024×1024、64 Micro、4 线程验证 `0.5*PE + 0.5*EPE + 1*BG_raw + 0.5*CEPT`，开启 Average 融合：

| 版图 | 更新次数 | 全局 EP 数 | 最佳轮 | 最佳轮 Core EPE 均值 (nm) |
| --- | ---: | ---: | ---: | ---: |
| contact_1_0_only_1024 | 2 | 3706 | 2 | 5.135243 |
| metal_3_0_only_1024 | 1 | 11482 | 0 | 16.105092 |

结果在 `result/full_chip_lsm/epe_core_validation_20261006__gkhs2jf/` 下的两个版图目录，实际配置见 `config_used.yaml`。
这次历史 metal 验证尚未统一 EPE 权重，11482 个 EP 中仅 9042 个参与梯度；其选点图和日志保留原始实验状态。
已逐项核对 EP 的唯一 Core 归属、全局点数、加权分项相加、Core 均值和最佳轮每块指标的汇总，并检查选点放大图。
这是指定短轮数的入口验证；metal 一次更新尚未改变二值 mask，最佳轮保留初始化状态，不表示完成了 100 次优化或收敛。

2026-10-06 对齐上述三条区域规则后，LSM、梯度交换、CTM、WENO/Godunov 和全图成像的 5 项测试全部通过。
另调用 Python 原函数对照 4 个 Micro 的完整 CEPT 与最终梯度矩阵，Average/Linear 两种模式均一致。
使用现有 metal CTM 二值结果，完成真实 1024×1024、64 Micro、4 线程、3 次更新的入口验证：
PEPenalty + 1.1×CEPT、Average 融合下，总 loss 从 319175 降至 249580.830497，最佳轮为 3，优化循环耗时 6.34 s。
已核对 1024×1024 输出、phi/mask 对应关系、最佳轮记录及全图成像/绘图；这是 3 次更新的运行验证，不代表 100 次已执行或已收敛。

以下 2026-10-05 的历史数值使用修改前的 Core-only loss、双端 CEPT 和 Halo 刷新，不能与当前区域规则的日志直接比较。
当时使用现有 contact CTM 二值结果，完成了真实 1024×1024、64 Micro、4 线程、20 次更新验证。
两次均开启 CEPT（权重 1.1）和 Average 梯度融合，最佳轮均为第 20 次更新：

| 数据项配置 | 初始总 loss | 最佳总 loss | 最佳数据 loss | 未加权 CEPT | 优化耗时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PE | 182768.729863 | 23522.515509 | 11530.315509 | 10902 | 38.44 s |
| 0.5 PE + 0.5 EPE | 101373.174374 | 24963.830863 | 17411.230863 | 6866 | 57.16 s |

结果分别位于 `result/full_chip_lsm/contact_1_0_only_1024_validation20/` 和
`result/full_chip_lsm/contact_1_0_only_1024_validation20_pe_epe/`，实际参数见各目录的 `config_used.yaml`。
这是指定 20 次更新的验证，不代表默认 200 次已执行或已收敛。
已检查输出的 1024×1024 尺寸、phi 有限值、`mask == (phi >= 0)`、初始 SDF 符号、加权损失及最佳轮记录对应关系。

同一 PE/EPE 混合损失实验的最佳 mask 已补做 1024×1024 全图成像：原始 target 的全图 PE 为
53489.618457，LSM 最佳 mask 的全图 PE 为 26656.775517。两张对比 PNG 已生成并检查标注和显示；
导出的 LSM 成像输入与 `lsm_mask.txt` 完全相同。另以 128×128、2 次更新验证了 LSM 入口自动成像并绘图的流程。
