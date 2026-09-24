# Litho_cpp

面向计算光刻实验的 C++20 项目，包含 Abbe/SOCS 成像、CTM、Level Set、MEEF 主图形控制点优化，以及基于骨架中心线的 SRAF 宽度优化。

本文重点说明当前 YAML 驱动的两个入口：

- `demo_MEEF_Optimizer_init`：初始化、构建 MEEF 矩阵或执行完整 MEEF 优化。
- `demo_SRAF_Optimizer_init`：读取 LSM 和 MEEF 主图形控制点，执行共享或独立 SRAF 半宽优化。

## 依赖

### macOS（Apple Silicon + Homebrew）

```bash
brew install cmake eigen fftw opencv yaml-cpp libomp
```

### Ubuntu / Debian

```bash
sudo apt install build-essential cmake libeigen3-dev libfftw3-dev \
    libopencv-dev libyaml-cpp-dev libomp-dev
```

SRAF 独立宽度模式使用 `libcmaes`。系统中找不到该库时，CMake 会从项目固定的 Git 提交下载源码，因此第一次配置需要网络。
CTM 的 L-BFGS 模式使用头文件库 LBFGS++，CMake 同样会在系统未安装时下载固定版本。

自动绘图脚本需要 NumPy 和 Matplotlib。建议安装到项目虚拟环境：

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install numpy matplotlib
```

## 编译

在项目根目录执行：

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release \
    --target demo_MEEF_Optimizer_init demo_SRAF_Optimizer_init \
    --parallel
```

根目录的 `config.yaml` 和 `assets/target_pattern/` 会在 **CMake 配置阶段**复制到 `build-release/`。修改根目录配置或目标图后，可以重新执行：

```bash
cmake -S . -B build-release
```

也可以在运行时给 demo 传入根目录配置的绝对路径，这样不依赖构建目录中的 YAML 副本。

两个主 demo 都应从 `build-release/` 目录启动，因为程序用当前目录的父目录作为项目根目录：

```bash
cd build-release
```

## CTM 优化

`demo_CTM` 可用 `config.yaml` 的 `ctm` 段选择固定步长梯度下降或 L-BFGS：

```yaml
ctm:
  optimizer: "lbfgs"           # gradient_descent / lbfgs
  max_iteration: 50
  learning_rate: 0.9            # 仅 gradient_descent 使用
  lbfgs_history_size: 10       # 仅 lbfgs 使用
  gradient_tolerance: 1.0e-2   # ||dPE/dtheta||_2 阈值；0 不设置正的梯度阈值
```

两种算法都优化相位变量 `theta`，并通过 `mask = (1 + cos(theta)) / 2` 生成灰度掩模。固定步长模式使用 `learning_rate` 更新 `theta`；L-BFGS 使用线搜索确定每步长度。两种模式都在梯度的整图 L2 范数不大于 `gradient_tolerance` 时停止，最多执行 `max_iteration` 轮。L-BFGS 日志中的 `eval` 包含线搜索试探点，因此可能多于迭代次数。

在项目根目录构建，然后从构建目录运行：

```bash
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target demo_CTM --parallel
cd build-release
./demo_CTM ../config.yaml
```

## 完整工作流

```text
target BMP + LSM mask
        │
        ▼
MEEF 主图形控制点优化
        │
        ├── best_wepe/control_points.txt
        └── best_epe/control_points.txt
        │
        ▼
SRAF 骨架提取、分叉图拟合和宽度优化
        │
        ├── shared：全部 SRAF 共用一个半宽
        └── independent：每个 SRAF 连通块一个半宽
```

MEEF 只更新主图形控制点，SRAF 在这一步保持固定。SRAF 宽度优化读取 MEEF 保存的主图形控制点，固定主图形形状，然后改变 SRAF 半宽。

---

## MEEF 优化

### 1. 准备输入

在根目录的 `config.yaml` 中首先保持以下三者属于同一个版图：

```yaml
PATTERN_NAME: &PATTERN_NAME "工字型"

mask:
  image_name: *PATTERN_NAME

meef:
  pattern_name: *PATTERN_NAME
  lsm_mask_path: "result/LSM_result/LSM_mask.txt"
  output_dir: "result/MEEF_result/test"
```

对应的目标图应存在于：

```text
assets/target_pattern/<PATTERN_NAME>.bmp
```

`meef.lsm_mask_path` 与 `meef.output_dir` 的相对路径按**项目根目录**解析。

### 2. 选择运行模式

```yaml
meef:
  run_mode: "--init-only"   # --init-only / --build-meef / --optimize
```

| 模式 | 用途 | 主要结果 |
|---|---|---|
| `--init-only` | 检查输入、控制点、EP 点和参数化 mask | `main_cps.txt`、`sraf_cps.txt`、`eps.txt`、`main_mask.txt`、`sraf_mask.txt` |
| `--build-meef` | 初始化后构建一次 X/Y MEEF 矩阵 | `meef_mx.txt`、`meef_my.txt` |
| `--optimize` | 完整迭代优化并保存最优快照 | `errors.csv`、`best_wepe/`、`best_epe/`、曲线图和 EPE 直方图 |

建议先运行 `--init-only`，确认数量和初始 mask，再运行 `--build-meef` 检查矩阵维度，最后使用 `--optimize`。

### 3. 选择主图形控制点来源

```yaml
meef:
  main_cp_mode: "lsm_interval"   # target_interval / lsm_interval / npy
  main_cp_interval: 7
  main_symmetry: "none"          # none / center / left-right / diagonal
```

| `main_cp_mode` | 含义 |
|---|---|
| `target_interval` | 从 target 轮廓按间隔选点 |
| `lsm_interval` | 从 LSM 主图形轮廓按间隔选点，通常更接近当前待优化形状 |
| `npy` | 从 `main_cps_npy_path` 导入 Python 保存的不等长轮廓控制点 |

使用 NPY 时设置：

```yaml
meef:
  main_cp_mode: "npy"
  main_cps_npy_path: "/absolute/path/to/control_points.npy"
```

demo 会调用 `scripts/convert_npy_control_points.py`，把 NPY 转成可检查的 `imported_main_cps.txt`。项目中的控制点文本统一使用 `y x` 顺序。

### 4. 选择固定 SRAF 来源

```yaml
meef:
  sraf_mask_mode: "lsm"          # lsm / fitted_txt
  fitted_sraf_txt_path: "/absolute/path/to/fitted_sraf.txt"
  sraf_cp_interval: 7
  sraf_min_cps: 8
  sraf_min_aera: 50               # 当前配置字段保留此拼写
```

- `lsm`：从 `lsm_mask_path` 中分离 SRAF，再提取其轮廓控制点。
- `fitted_txt`：读取已经拟合的 SRAF mask；该文件可以是完整 mask，也可以仅包含 SRAF。

### 5. 曲线、光栅化和 EP 权重

```yaml
meef:
  curve_type: "BS"               # 主图形：OA / BZ / BS
  sraf_curve_type: "CR"          # 固定 SRAF：OA / BS / CR
  rasterizer: "msaa"             # msaa / dirac
  msaa_level: 16                 # 4 / 16 / 64
  delta: 0.15
  interval_line: 5
  interval_corner: 2
  mid_weight: 4.0
  other_weight: 4.0
  wepe_all_eps: true
  optimize_wepe_only: false
  epe_histogram_bin_width_nm: 0.25
```

- `wepe_all_eps: true`：所有已选择 EP 的 `weight_epe=1`。
- `wepe_all_eps: false`：只有关键 EP 的 `weight_epe=1`，其他 EP 权重为 0。
- `optimize_wepe_only: true`：MEEF 方程只使用计入 WEPE 的 EP 行。
- `optimize_wepe_only: false`：MEEF 方程使用全部 EP 行。
- `epe_histogram_bin_width_nm`：`best_wepe/` 和 `best_epe/` 中 EPE 直方图的柱宽，单位为 nm。

### 6. MEEF 构建、更新和停止条件

```yaml
meef:
  meef_builder: "finite_difference"       # finite_difference / analytic
  meef_matrix_update_mode: "periodic"     # every_iteration / periodic / initial_only
  meef_rebuild_interval: 5
  move_strategy: "xy"
  iter: 40
  stop_mode: "small_step"                 # fixed_iterations / small_step
  step_tol: 0.05
  patience: 2
```

- `finite_difference`：分别扰动控制点的 `x+ / x- / y+ / y-`，用中心差分构建 `Mx/My`。
- `analytic`：使用解析方向导数；当前要求 `curve_type: BS` 与 `rasterizer: dirac`。
- `every_iteration`：每轮重建 MEEF 矩阵，计算量最大。
- `periodic`：按 `meef_rebuild_interval` 周期重建，其余轮次复用矩阵。
- `initial_only`：只在第一轮构建，以后一直复用。
- `fixed_iterations`：固定运行 `iter` 轮。
- `small_step`：最大控制点位移连续 `patience` 轮小于 `step_tol` 时提前停止；`iter` 仍是最大迭代数。

### 7. 运行 MEEF

使用构建目录中的配置副本：

```bash
cd build-release
./demo_MEEF_Optimizer_init
```

直接使用刚修改的根目录配置：

```bash
cd build-release
./demo_MEEF_Optimizer_init "$(cd .. && pwd)/config.yaml"
```

`--optimize` 完成后会自动调用 `scripts/plot_meef_parametric_curves.py`，保存 `best_wepe_parametric_curves.png` 和 `best_epe_parametric_curves.png`，并打开 LSM 与最终结果对比窗口；关闭窗口后程序退出。终端输出同步保存到 `console_output.log`。

### 8. MEEF 主要输出

默认目录为 `result/MEEF_result/test/`：

| 文件或目录 | 说明 |
|---|---|
| `config_used.yaml` | 本次运行的配置快照 |
| `console_output.log` | 完整终端日志 |
| `main_cps.txt`、`sraf_cps.txt` | 初始化控制点，坐标顺序为 `y x` |
| `eps.txt`、`eps_weights.txt` | EP 坐标及对应的 EPE/MEEF 权重 |
| `main_mask.txt`、`sraf_mask.txt` | 初始化后渲染的两部分 mask |
| `meef_mx.txt`、`meef_my.txt` | `--build-meef` 模式生成的矩阵 |
| `errors.csv` | 每轮 PE、EPE、wEPE 和累计时间 |
| `iterations/` | 最后一轮 mask、aerial、wafer、MEEF 矩阵和位移 |
| `curves/` | 每轮控制点与参数化曲线历史 |
| `best_wepe/` | 最小 wEPE 对应的控制点、mask、aerial、wafer、指标和直方图 |
| `best_epe/` | 最小普通 EPE 对应的同类结果 |

`best_wepe/control_points.txt` 与 `best_epe/control_points.txt` 可以作为下一阶段 SRAF 宽度优化的主图形输入。

---

## SRAF 宽度优化

### 1. 输入关系

```yaml
sraf_optimizer:
  paths:
    lsm_mask_path: "result/LSM_result/LSM_mask.txt"
    main_control_points_path: "result/MEEF_result/test/best_wepe/control_points.txt"
    output_dir: "result/SRAF_result"
```

三个关键输入必须对应同一版图：

1. `PATTERN_NAME` 对应的 target BMP；
2. `lsm_mask_path` 对应的完整 LSM mask；
3. `main_control_points_path` 对应的 MEEF 主图形控制点。

### 2. 骨架和分叉 SRAF

```yaml
sraf_optimizer:
  geometry:
    foreground_threshold: 1.0e-6
    target_threshold: 0.0
    overlap_ratio: 0.05
    fallback_dilate_radius: 3
    opening_radius: 0
    minimum_component_area: 3
    control_point_interval: 10
```

处理流程为：完整 LSM → 分离主图形/SRAF → 清理小连通块 → Zhang-Suen 骨架化 → 控制点采样。

一个 SRAF 连通块如果包含分叉，会按“端点或分叉点之间的路径”拆成多条图边。每条边分别用开放式 Catmull-Rom 曲线拟合，但所有边仍保留相同 `component_id`，共享该连通块的同一个宽度变量。

### 3. 半宽和 MSAA 渲染

```yaml
sraf_optimizer:
  rendering:
    curve_type: "BS"              # 主图形：OA / BZ / BS
    initial_half_width: 2.0
    maximum_half_width: 5.0
    samples_per_axis: 4
    main_mask_msaa_level: 16
```

优化变量是**半宽** `w`，完整线宽为 `2w`。若像素尺寸为 `4 nm`，`w=2 px` 对应约 `16 nm` 完整线宽。

程序先在曲线包围框外扩 `maximum_half_width + 1` 像素形成 ROI，并缓存每个子像素到该连通块所有中心曲线边的最短距离。评价新宽度时只判断 `distance <= w`，无需重复拟合曲线和计算距离。

`maximum_half_width` 必须不小于初始半宽以及搜索上界。

### 4. 评价函数

```yaml
sraf_optimizer:
  evaluation:
    epe:
      interval_line: 5
      interval_corner: 2
      mid_weight: 4.0
      other_weight: 1.0
    pv_band:
      dose_margin: 0.05
      defocus_range: 100.0
      defocus_step: 3
      mode: "full"                # dose_only / defocus_only / full

  objective:
    pv_band_weight: 0.90
    weighted_epe_weight: 0.05
    pixel_error_weight: 0.05
```

候选宽度的代价为：

```text
cost = pv_band_weight       × PV/PV_initial
     + weighted_epe_weight  × mean_wEPE/mean_wEPE_initial
     + pixel_error_weight   × PE/PE_initial
```

三项先除以初始值归一化，再按权重求和。权重通常设为和为 1，方便解释各项贡献。

### 5. 共享宽度模式

```yaml
sraf_optimizer:
  width_optimization:
    mode: "shared"
    shared:
      min_half_width: 1.0
      max_half_width: 4.0
      coarse_step: 0.5
      fine_step: 0.1
```

所有 SRAF 连通块共用一个半宽。算法先遍历完整区间做粗搜索，再在最优粗搜索点附近做细搜索。结果保存在：

- `shared_width_history.csv`
- `optimized_shared_sraf_mask.png`
- `optimized_shared_mask.png`
- `optimized_shared_pv_band_map.png`
- `optimized_shared_pv_band_map_comparison.png`

### 6. 独立宽度模式

```yaml
sraf_optimizer:
  width_optimization:
    mode: "independent"
    independent:
      min_half_width: 1.0
      max_half_width: 4.0
      cma_es:
        initial_sigma: 0.375
        population_size: 12
        max_evaluations: 1200
        tolerance_x: 1.0e-2
        tolerance_fun: 1.0e-3
        seed: 1
```

每个 SRAF **连通块**对应一个 CMA-ES 变量。分叉后的一组图边仍是一个连通块，因此仍只对应一个宽度。

- `population_size`：每一代评价的完整宽度向量数量。
- `max_evaluations`：候选宽度向量的最大评价次数。
- `initial_sigma`：CMA-ES 搜索分布的初始标准差，不是 SRAF 宽度。
- `seed > 0`：固定随机种子以复现实验；`seed: 0` 使用随机种子。

结果保存在：

- `independent_width_history.csv`：每次评价的 cost、归一化指标和完整宽度向量。
- `optimized_independent_widths.csv`：`sraf_index`、原 `component_id` 与最优半宽的对应关系。
- `cma_es_loss_curve.png`：候选 cost 与 best-so-far 曲线。
- `optimized_independent_sraf_mask.png`
- `optimized_independent_mask.png`
- `optimized_independent_pv_band_map.png`
- `optimized_independent_pv_band_map_comparison.png`

### 7. 运行 SRAF 宽度优化

```bash
cd build-release
./demo_SRAF_Optimizer_init "$(cd .. && pwd)/config.yaml"
```

虽然可执行文件名称中保留了 `_init`，当前入口在构造完优化器后会直接调用 `optimize()`，因此运行的是完整共享搜索或独立 CMA-ES 优化。

### 8. SRAF 几何输出

默认输出目录为 `result/SRAF_result/`：

| 文件 | 说明 |
|---|---|
| `main_control_points_loaded_yx.txt` | 实际载入的 MEEF 主图形控制点 |
| `sraf_control_points_yx.txt` | 按 `component_id` 分组的 SRAF 骨架控制点 |
| `sraf_graph_edges_yx.txt` | 分叉连通块每条边的原始骨架像素链 |
| `sraf_fitted_graph_edges_yx.txt` | 每条分叉边拟合后的密集曲线点 |
| `initial_main_mask.png` | 固定主图形的 MSAA mask |
| `initial_sraf_mask.png` | 初始半宽恢复的 SRAF mask |
| `initial_combined_mask.png` | 主图形与初始 SRAF 的并集 |

这些文本文件使用 `y x` 坐标。独立宽度结果中的 `component_id` 用来对应原 SRAF 连通区，`sraf_index` 是宽度向量下标。

### 9. 绘制独立宽度几何图

独立宽度优化完成后运行：

```bash
cd ..
MPLCONFIGDIR=.cache/matplotlib ./.venv/bin/python \
    scripts/plot_sraf_width_geometry.py \
    --result-dir result/SRAF_result \
    --target-mask result/MEEF_result/test/target_mask.txt
```

脚本读取逐边拟合结果，可以绘制分叉 SRAF；同一个 `component_id` 的所有边使用 `optimized_independent_widths.csv` 中的同一个半宽。默认输出为：

```text
result/SRAF_result/optimized_independent_parametric_geometry.png
```

该图是曲线和线宽的几何示意。实际进入成像计算的 MSAA 灰度结果应查看 `optimized_independent_sraf_mask.png`。

---

## 测试

```bash
cmake --build build-release \
    --target test_sraf_geometry test_sraf_curve test_parametric \
    --parallel
ctest --test-dir build-release --output-on-failure
```

## 常见问题

### 修改 `config.yaml` 后运行结果没有变化

如果使用 `./demo_*` 的默认配置，demo 读取的是 `build-release/config.yaml`。重新执行 `cmake -S . -B build-release` 刷新副本，或者把根目录 YAML 的绝对路径作为命令行参数传入。

### 提示找不到 LSM、控制点或 target

- 相对的 LSM、控制点和输出路径按项目根目录解析。
- target 从 `build-release/target_pattern/<PATTERN_NAME>.bmp` 读取。
- 从 `build-release/` 目录启动两个主 demo。
- 检查 target、LSM 和 MEEF 控制点是否来自同一个版图。

### Python 绘图提示缺少 Matplotlib

```bash
./.venv/bin/python -m pip install numpy matplotlib
```

C++ demo 优先使用项目的 `.venv/bin/python`，否则回退到系统 `python3`。

### 独立宽度大量卡在上下界

先检查搜索范围、初始半宽、骨架拟合形状和目标函数各归一化项。增大 `population_size` 只会增加每代候选数量，不能修复不合理的中心线形状或宽度边界。

## 目录结构

```text
Litho_cpp/
├── assets/
│   ├── target_pattern/          # target BMP
│   └── lsm_mask/                # 示例 LSM mask
├── config.yaml                  # 成像、MEEF 和 SRAF 宽度配置
├── demo/                        # YAML 驱动的完整流程入口
├── examples/                    # 小型算法示例
├── include/                     # 公共头文件
├── source/
│   ├── litho_model/             # 成像模型
│   ├── optimizer/               # CTM、LSM、MEEF、SRAF 优化器
│   └── utils/                   # 曲线、MSAA、损失、骨架等工具
├── scripts/                     # 结果绘图和格式转换脚本
├── tests/                       # 回归测试
└── result/                      # 默认运行结果
```
