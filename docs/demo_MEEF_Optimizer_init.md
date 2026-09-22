# demo_MEEF_Optimizer_init 使用指南

`demo/demo_MEEF_Optimizer_init.cpp` 是 MEEF_Optimizer 的集成测试 demo，支持三种运行模式：初始化测试、MEEF 矩阵构建、完整优化流程。

# 1. 编译

在项目根目录下执行（任选一个 build 目录）：

```bash
cmake -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target demo_MEEF_Optimizer_init -j
```

产物：`build-release/demo_MEEF_Optimizer_init`

## 2. 运行方式

运行参数均从 YAML 的 `meef:` 节读取。先在 `config.yaml` 中设置 `run_mode`、
`iter`、`output_dir` 和 EP 权重，再从 `build-release` 目录执行：

```bash
cd build-release
./demo_MEEF_Optimizer_init ../config.yaml
```

省略路径时读取 `build-release/config.yaml`；修改根目录的 YAML 后需重新运行
`cmake -S . -B build-release` 刷新该副本。命令行只接受一个可选的 YAML 路径。

## 3. 三种运行模式

### 3.1 `run_mode: "--init-only"`

只构造 `MEEF_Optimizer`，测试初始化是否正常，不跑光刻仿真。会输出 `main_cps.txt` / `sraf_cps.txt` / `eps.txt` / `sraf_mask.txt` / `main_mask.txt` 并校验文件完整性。

### 3.2 `run_mode: "--build-meef"`

构建 X/Y 方向 MEEF 矩阵（每个控制点扰动 ±delta 共 4 次光刻仿真，OpenMP 并行），保存为 `meef_mx.txt` / `meef_my.txt`，并打印矩阵尺寸与范数。

### 3.3 `run_mode: "--optimize"`

执行完整 MEEF 优化流程，最多迭代 YAML 中 `iter` 指定的次数。结束后自动保存参数化曲线图和 EP 选点图，并尝试打开四图对比窗口。

`optimize_wepe_only: false` 时，MEEF 矩阵的所有 EP 行都参与求解，并乘各自的
`weight_meef`。设为 `true` 时，矩阵行还会乘 `weight_epe`。`wepe_all_eps: true`
会把全部已选 EP 的 `weight_epe` 设为 1，因此全部 EP 都计入 WEPE、图上全部为红色；
此时 WEPE 总和等于 EPE 总和。`mid_weight` 和 `other_weight` 控制的是另一组
`weight_meef`，不会被 `wepe_all_eps` 改动。

## 4. YAML 配置

`config.yaml` 中的 `meef:` 节用于配置 `MEEFPipelineConfig`。运行时会将完整的
YAML 复制到 `meef.output_dir / meef.config_snapshot_name`，默认即
`result/MEEF_result/test/config_used.yaml`，以便结果可复现。

`output_dir` 的相对路径以项目根目录为基准；`config_snapshot_name` 和
`console_log_name` 必须是不含目录的文件名。程序会继续向终端输出，并同时把
`std::cout` / `std::cerr` 保存到日志文件。

例如只修改配置即可指定输出目录、快照名称与迭代次数：

```yaml
meef:
  output_dir: "result/MEEF_result/run_001"
  config_snapshot_name: "run_001_config.yaml"
  console_log_name: "run_001_console.log"
  iter: 50
  main_cp_mode: "target_interval"
  main_cps_npy_path: ""
  main_cp_interval: 9
  sraf_mask_mode: "lsm"
  fitted_sraf_txt_path: ""
  delta: 0.10
```

可设置字段如下：


| 字段               | 值     | 含义                         |
| ------------------ | ------ | ---------------------------- |
| `console_log_name` | console_output.log | 终端输出日志名称       |
| `pattern_name`     | 工字型 | 图案名                       |
| `move_strategy`    | xy     | X/Y 双向扰动                 |
| `main_cp_mode`     | target_interval | `target_interval` 从目标轮廓取点；`lsm_interval` 从 LSM 主图形轮廓取点；`npy` 导入控制点 |
| `main_cps_npy_path` | 空    | `npy` 模式的控制点文件，点为 `(y,x)` |
| `main_cp_interval` | 7      | 两种 interval 模式每取一点后跳过的轮廓点数；0 表示逐点选取 |
| `main_symmetry`    | none   | 不使用对称                   |
| `sraf_mask_mode`   | lsm    | 从原 LSM 提取；也可设为 `fitted_txt` |
| `fitted_sraf_txt_path` | 空 | 拟合后的完整 mask 或 SRAF-only 文本 |
| `sraf_cp_interval` | 5      | SRAF 控制点间隔，单位为原图像素；外边界和孔洞边界均采样 |
| `sraf_min_cps`     | 8      | SRAF 最少控制点数            |
| `sraf_min_aera`    | 50     | SRAF 最小连通面积            |
| `msaa_level`       | 16     | MSAA 采样数                  |
| `curve_type`       | BS     | 主图形周期 B 样条，输入点为控制点 |
| `sraf_curve_type`  | 同 `curve_type` | 固定 SRAF 单独选择 `BS`、`OA` 或 `CR`；`CR` 为经过输入点的周期向心 Catmull-Rom |
| `delta`            | 0.15   | 中心差分步长                 |
| `dilate_radius`    | 2      | 主/SRAF 分离膨胀半径         |
| `interval_line`    | 5      | 直线段 EP 间隔               |
| `interval_corner`  | 2      | 拐角 EP 间隔                 |
| `mid_weight`       | 4.0    | 中点 EP 权重                 |
| `other_weight`     | 1.0    | 其他 EP 权重                 |
| `wepe_all_eps`     | false  | `true` 时所有已选 EP 的 `weight_epe=1`，全部计入 WEPE |
| `optimize_wepe_only` | false | `true` 时 MEEF 矩阵行额外乘 `weight_epe` |
| `stop_mode`        | fixed_iterations | 跑满 `iter`；`small_step` 可按最大控制点位移提前停止 |
| `step_tol`         | 0.02   | `small_step` 的位移阈值，单位 pixel |
| `patience`         | 3      | `small_step` 需连续满足阈值的轮数 |

`main_cp_mode: npy` 支持 Python 版 `np.save(..., dtype=object)` 产生的不等长多轮廓；
demo 会用项目 `.venv/bin/python`（不存在时用 `python3`）将其转换为输出目录中的
`imported_main_cps.txt`，随后由 C++ 校验边界并读取。`sraf_mask_mode: lsm`
会从 LSM 分离出 SRAF，`fitted_txt` 会从拟合文件中分离出 SRAF。`lsm` 配合 `rasterizer: msaa` 时提取 SRAF 的外边界和孔洞边界控制点，按 `sraf_curve_type` 生成曲线并以奇偶规则渲染，在 MEEF 迭代中固定渲染结果。`OA` 是折线，`CR` 是周期向心 Catmull-Rom 插值曲线；两者都经过输入点，`CR` 在点间可能改变细 SRAF 的宽度，需检查空中像是否超过显影阈值。`fitted_txt` 或 `rasterizer: dirac` 则固定使用分离后的原始 SRAF 灰度 mask。

修改根目录 `config.yaml` 后，需要重新执行 CMake 配置以刷新 build 目录副本；
也可以运行 demo 时显式传入根目录配置文件路径。

## 5. 输出文件

所有输出写入 `save_path`：


| 文件                   | 模式           | 说明                      |
| ---------------------- | -------------- | ------------------------- |
| `main_cps.txt`         | 全部           | 主图形控制点              |
| `imported_main_cps.txt`| NPY 导入       | `.npy` 的可审计转换结果   |
| `sraf_cps.txt`         | 全部           | SRAF 控制点               |
| `eps.txt`              | 全部           | edge points               |
| `eps_weights.txt`      | 全部           | 与 `eps.txt` 逐行对应的 EPE/MEEF 权重 |
| `sraf_mask.txt`        | 全部           | 实际用于优化的固定 SRAF 掩模 |
| `main_mask.txt`        | 全部           | 参数化重建的主图形掩模   |
| `meef_mx.txt`          | `--build-meef` | X 方向 MEEF 矩阵          |
| `meef_my.txt`          | `--build-meef` | Y 方向 MEEF 矩阵          |
| `lsm_mask.txt`         | `--optimize`   | 初始 LSM 掩模（用于对比） |
| `lsm_wafer.txt`        | `--optimize`   | 初始 LSM 成像结果         |
| `initial_mask.txt`、`initial_aerial.txt`、`initial_wafer.txt` | `--optimize` | 参数化初始状态的掩模、空中像和显影图 |
| `iterations/mask.txt`  | `--optimize`   | 优化后掩模（每轮覆盖）    |
| `iterations/wafer.txt` | `--optimize`   | 优化后成像（每轮覆盖）    |
| `best_wepe/`、`best_epe/` | `--optimize` | 两种最优结果快照 |
| `errors.csv`           | `--optimize`   | 每轮 EPE/WEPE 历史        |
| `best_wepe_parametric_curves.png` | `--optimize` | 最优 WEPE 参数化曲线、控制点和目标轮廓 |
| `best_epe_parametric_curves.png` | `--optimize` | 最优 EPE 参数化曲线、控制点和目标轮廓 |
| `ep_selection.png`     | `--optimize`   | 目标版图上的 EP 选点，区分计入和未计入 WEPE 的点 |
| `console_output.log`   | 全部           | 本次运行的终端输出副本    |

> `--optimize` 模式下每轮迭代会**覆盖** `iterations/`，不保留中间轮次；最优结果分别存入 `best_wepe/` 和 `best_epe/`。
> 日志中的 `SRAF exposure` 统计固定 SRAF 掩模覆盖的非目标区域内，空中像达到显影阈值的像素数和最大强度。

## 6. 优化后可视化

`--optimize` 保存最优结果后，会自动运行 `scripts/plot_meef_parametric_curves.py --best both`，
生成两张参数化曲线图和 `ep_selection.png`。EP 图以 target 为底图，红色为
`weight_epe=1`、计入 WEPE 的点，蓝色为未计入 WEPE 的点；设置 `wepe_all_eps: true`
后所有 EP 都显示为红色。坐标按文件中的 `(y,x)` 绘为图上的 `(x,y)`。
脚本当前支持 `curve_type: BS`，优先使用项目 `.venv/bin/python`；绘图失败会打印 warning，
已保存的数值结果仍保留。

随后 demo 会尝试打开四图对比窗口，关闭窗口后退出：

```bash
python3 scripts/show_multi.py \
    <save_path>/lsm_mask.txt gray \
    <save_path>/lsm_wafer.txt gray \
    <save_path>/iterations/mask.txt gray \
    <save_path>/iterations/wafer.txt gray \
    --titles 'LSM Baseline Mask|LSM Baseline Wafer|MEEF Optimized Mask|MEEF Optimized Wafer'
```

在同一个 matplotlib 窗口显示 4 张图：LSM mask / LSM wafer / 优化后 mask / 优化后 wafer。关闭窗口后 demo 退出。

依赖：`matplotlib` + `numpy`。若四图窗口脚本退出状态异常，demo 会打印 warning 但不中断。

## 7. 常见问题

**Q: 如何让全部 EP 都计入 WEPE？**
在 YAML 的 `meef:` 节设置 `wepe_all_eps: true`。此时 `eps_weights.txt` 的第一列全为 1，EP 图全部显示红色。

**Q: `--build-meef` 很慢？**
MEEF 矩阵构建需要 `4 × 控制点数` 次光刻仿真，OpenMP 并行可加速约 4–5 倍。确保 Release 编译且未限制线程数。

**Q: `optimize_wepe_only` 和 `wepe_all_eps` 有什么区别？**
前者决定 MEEF 求解时是否再按 `weight_epe` 缩放各 EP 行；后者决定哪些已选 EP 的 `weight_epe` 为 1。启用 `wepe_all_eps` 后，即使前者为 `true`，所有已选 EP 行也会保留。
