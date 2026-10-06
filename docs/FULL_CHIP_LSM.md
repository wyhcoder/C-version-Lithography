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

换版图时同时修改 `target_path`、`initial_mask_path` 和 `output_dir`；光学参数应与生成 CTM 输入的实验一致。

## 每轮流程

1. 将原 C++ 二值化得到的全局 0/1 mask 提取为各块 Core + Halo，再用 `LevelSetUtils::reinit_sdf()` 初始化 phi。
2. 冻结所有 Micro 的局部状态，由 `MicroGradientExchange` 并行成像、计算 Core + Guard loss 和原 C++ PE/EPE 梯度。
3. 等待所有 Micro 完成，再串行交换四邻域条带：CEPT 比较本块 Guard 与邻居 Core 的 mask，邻居 Core 作为冻结参考，只给本块 Guard 加梯度；按开关融合 Guard 数据梯度。
4. 将总梯度作为法向速度，乘 Core + Guard 更新窗口；各 Micro 并行计算 WENO/Godunov、曲率项和自己的 CFL dt。
5. 按原 C++ 两阶段 RK2 推进 phi，两阶段使用同一轮梯度和第一阶段的 dt，仅更新 Core + Guard。
6. 每到重初始化间隔，使用原 C++ SDF 接口重新计算距离，仅提交 Core + Guard。所有块完成后统一提交新 phi。
7. 按 `phi >= 0` 渲染局部二值 mask，不从邻居或 Core 回拼图刷新 phi。Core + Guard 窗口之外的 phi 始终保留本块初始化时的值。
8. 评价更新后的全部局部状态，记录历史最佳轮。最终只回拼同一最佳轮的 Core，返回 mask 和 phi。

调度顺序参照 Python `ilt_macro_serial.py` 的 LEVEL_SET 阶段和 `levelset_micro_parallel.update_micro()`。
数值部分采用当前 C++：内正外负的 SDF、`phi >= 0` 二值渲染、WENO/Godunov、曲率和 RK2。
窗口外 phi 不更新，Guard 始终保留各块独立状态；这两条规则与 Python 当前 staged LSM 一致。
损失区域固定为 Core + Guard 的完整矩形，包含角区；CEPT 和梯度融合只处理四侧有实际邻居的 Guard 条带，不处理角区。
PE/PEPenalty 的输出误差窗口和 EPE 的 EP 范围都使用该矩形，光学反传仍得到完整局部梯度。
EPE 保留当前 C++ 的全局 `select_eps_via(2)` 选点规则，芯片外侧 Guard 对应周期 EP；重叠区域的 EP 可由多个 Micro 评价。
Python 的可选灰度渲染、TV、进程池/MPI 参数不属于此接口；C++ 使用 OpenMP，LSM 单块算法无需完整重跑。
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
options.gradient_options.loss_type = MicroLossType::PE;
options.gradient_options.enable_cept = true;
options.gradient_options.enable_grad_fusion = true;
auto result = optimizer.optimize(ctm_binary_mask, global_target, options);
```

`meta/cache` 在外部初始化，须比 optimizer 存活更久，且 `cache.N == meta.tile_size`。
`Grid(N, pixel_size_nm, NA, wavelength_nm)` 直接使用输入的 N，不扩成 2 的幂或额外加一，自动处理奇偶尺寸。
单块入口的 N 来自 `system.grid_size_pixel`；全芯片局部成像来自 `meta.tile_size`，最终整图成像来自实际矩阵边长。
空间像素中心为 `(i-(N-1)/2.0)*pixel_size_nm`，原点位于几何中心；偶数时处于像素之间，奇数时落在中心像素上。
中心化频率为 `(i-N/2)/(N*pixel_size_nm)`，其中 N/2 是整数除法，再归一化到 NA/λ；奇偶尺寸都包含零频。
数据项和梯度采用同一损失选项；CEPT 比较当前各块 mask 像素，而非梯度或 phi 数值。
LSM 内部固定 `loss_region=CoreGuard`、`cept_guard_only=true`，无需在 YAML 里另外设置。
共享梯度模块的默认 Core 损失区域和双端 CEPT 继续供 CTM 使用。
梯度融合改变数据更新方向，不修改 loss 标量；EPE 沿用原 C++ 梯度公式。
曲率是 PDE 的平滑项，不计入用于最佳轮选择的 `data_loss + cept_weight * cept_loss`。

## 输出

启动日志打印实际损失类型、Core + Guard 评价区域以及各项权重。每轮明确区分：

- `total_loss`：本轮全部 Micro 的总损失，用于历史最佳轮选择。
- `data_loss(类型)`：所选数据项；`PEPenalty` 为普通 PE 加背景显影惩罚，`PE_EPE` 为加权 PE 加加权 EPE。
- `PE_weighted`、`BG_penalty_weighted`、`EPE_weighted`：实际加入数据损失的各项贡献。
- `CEPT_raw`：本块 Guard 与四邻域 Core 的 mask 平方差之和，尚未乘 `cept_weight`。
- `CEPT_weighted`：`cept_weight * CEPT_raw`，实际加入总损失的贡献；关闭时显示 `CEPT=off`。
- `dt[min,max]`：各 Micro 的数值时间步范围，不是损失，也不是计算耗时。

背景惩罚为 `penalty_weight * Σ[target<0.5] max(WI-penalty_threshold,0)^2`。
`PEPenalty` 模式不使用配置中的 `pe_weight/epe_weight`；这两个参数仅控制 `PE_EPE`。
梯度融合只改变更新驱动，曲率只用于 PDE 平滑，两者不另外加入 `total_loss`。

`lsm_mask.txt/bmp` 保存最佳轮的二值 Core 回拼结果，`lsm_phi.txt` 保存对应 phi。
`initial_mask.txt/bmp` 与 `initial_phi.txt` 保存实际 LSM 初值。
`history.csv` 保留原六列，追加 `pe_contribution`、`penalty_contribution`、`epe_contribution`、`cept_contribution`，均为已加权贡献，未启用的项为 0。
满足 `data_loss = pe_contribution + penalty_contribution + epe_contribution`，`loss = data_loss + cept_contribution`。
原 `cept_loss` 列继续保存未加权值；记录覆盖 0～iterations 的所有状态。
`summary.csv` 区分最佳轮和末轮，`config_used.yaml` 保存实际更新次数及输入/输出路径。
返回的 `local_masks/local_phi` 保留最佳轮的完整独立局部状态。

## 原始版图与 LSM 最佳版图的全图成像对比

`demo_full_chip_lsm` 结束后自动对原始 `target` 和最佳轮的 `global_mask` 分别成像。
按全图尺寸重新准备光瞳、SOCS 核和 FFT，当前为 1024×1024；两张版图共用同一套光学和显影参数。
`imaging_source_grid_size=256` 仅控制照明积分采样，与全图 FFT 尺寸独立。
WI 保留 sigmoid 连续值；LSM 成像输入直接使用最佳二值 mask。

输出到当前 LSM 结果目录的 `imaging_comparison/`：

| 文件 | 含义 |
| --- | --- |
| `lsm_comparison.png` | 全图对比，上排未优化 target，下排 LSM 最佳版图；三列为 mask、AI、WI |
| `lsm_comparison_zoom.png` | 同一对比的中心 256×256 像素窗口，切片区间不包含结束索引 |
| `target_mask.txt` / `optimized_lsm_mask.txt` | 两张成像输入版图的原始 double 矩阵 |
| `target_ai.txt` / `optimized_ai.txt` | 两张完整版图各自的全图空中像 |
| `target_wi.txt` / `optimized_wi.txt` | 两张完整版图各自的连续显影结果 |
| `metrics.csv` | 两个全图 PE，均相对原始 target 计算，不含 CEPT |
| `sampling.csv` | 全图 FFT 尺寸、照明采样、光源点数、SOCS 核数和 nm/像素 |

成像和 PE 由 C++ 完成，入口使用项目 `.venv/bin/python` 调用绘图脚本保存 PNG；绘图依赖 NumPy/Matplotlib。
全图 PE 与迭代日志中的局部 Core + Guard loss/CEPT 是不同评价，分别保存。

已有 LSM 结果可以单独补成像和画图，无须重新优化，例如对前面的 20 轮验证结果：

```bash
cmake --build build-release --target demo_full_chip_imaging_compare --parallel 4
OMP_NUM_THREADS=4 ./build-release/demo_full_chip_imaging_compare result/full_chip_lsm/contact_1_0_only_1024_validation20_pe_epe/config_used.yaml
MPLCONFIGDIR=/tmp/litho-matplotlib XDG_CACHE_HOME=/tmp/litho-matplotlib .venv/bin/python scripts/plot_full_chip_imaging_compare.py result/full_chip_lsm/contact_1_0_only_1024_validation20_pe_epe/imaging_comparison
```

## 验证

```bash
cmake --build build-release --target test_full_chip_lsm test_micro_gradient_exchange test_level_set_godunov --parallel 4
OMP_NUM_THREADS=4 OMP_MAX_ACTIVE_LEVELS=2 ctest --test-dir build-release -R '^(full_chip_lsm|micro_gradient_exchange|level_set_godunov)$' --output-on-failure
```

`test_full_chip_lsm` 用原单块数值接口独立执行轨迹，核对 Abbe/SOCS 下的 RK2、CFL 和重初始化；
另核对窗口外 phi 与初始局部 SDF 完全相同、独立 Guard、四种损失、两种融合模式、串行/并行一致性，以及返回状态能复现最佳 loss。
`test_micro_gradient_exchange` 另覆盖 Core + Guard 的 PE/PEPenalty 有限差分、Guard/周期 EPE 选点和仅作用于 Guard 的 CEPT。

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
