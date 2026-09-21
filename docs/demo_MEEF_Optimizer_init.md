# demo_MEEF_Optimizer_init 使用指南

`demo/demo_MEEF_Optimizer_init.cpp` 是 MEEF_Optimizer 的集成测试 demo，支持三种运行模式：初始化测试、MEEF 矩阵构建、完整优化流程。

# 1. 编译

在项目根目录下执行（任选一个 build 目录）：

```bash
cmake -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target demo_MEEF_Optimizer_init -j
```

产物：`build-release/demo_MEEF_Optimizer_init`

## 2. 命令行参数

程序采用**位置参数**解析，顺序固定：

```
./demo_MEEF_Optimizer_init [config] [lsm_mask] [save_path] [mode] [iter] [--wepe-only]
```


| 位置    | 参数        | 默认值                               | 说明                                  |
| ------- | ----------- | ------------------------------------ | ------------------------------------- |
| argv[1] | config      | `config.yaml`                        | 仿真参数 YAML                         |
| argv[2] | lsm_mask    | `assets/lsm_mask/ls_image工字型.txt` | 初始 LSM 掩模                         |
| argv[3] | save_path   | `result/MEEF_result/test`            | 输出目录                              |
| argv[4] | mode        | `--init-only`                        | 运行模式（见下）                      |
| argv[5] | iter        | `100`                                | 优化迭代次数（仅`--optimize` 生效）   |
| argv[6] | --wepe-only | （不启用）                           | WEPE-only 开关（仅`--optimize` 生效） |

> ⚠️ 因为是位置参数，`mode` 必须出现在第 4 个位置。不能跳过前三个参数直接传 `--optimize`，否则 `--optimize` 会被当成 `config` 路径，报「配置文件不存在」。

## 3. 三种运行模式

### 3.1 `--init-only`（默认）

只构造 `MEEF_Optimizer`，测试初始化是否正常，不跑光刻仿真。会输出 `main_cps.txt` / `sraf_cps.txt` / `eps.txt` / `sraf_mask.txt` / `main_mask.txt` 并校验文件完整性。

```bash
./build-release/demo_MEEF_Optimizer_init
# 或显式指定前三个参数
./build-release/demo_MEEF_Optimizer_init config.yaml assets/lsm_mask/ls_image工字型.txt result/MEEF_result/test
```

### 3.2 `--build-meef`

构建 X/Y 方向 MEEF 矩阵（每个控制点扰动 ±delta 共 4 次光刻仿真，OpenMP 并行），保存为 `meef_mx.txt` / `meef_my.txt`，并打印矩阵尺寸与范数。

```bash
./build-release/demo_MEEF_Optimizer_init config.yaml assets/lsm_mask/ls_image工字型.txt result/MEEF_result/test --build-meef
```

### 3.3 `--optimize [iter] [--wepe-only]`

执行完整 MEEF 优化流程：构建 MEEF 矩阵 → 截断 SVD 求解 → 更新控制点 → 渲染评估，循环 `iter` 次。优化结束后自动调用 `scripts/show_multi.py` 弹出 4 联图窗口。

```bash
# 全 EP 点优化，50 次迭代
./build-release/demo_MEEF_Optimizer_init config.yaml assets/lsm_mask/ls_image工字型.txt result/MEEF_result/test --optimize 50

# WEPE-only 优化，50 次迭代
./build-release/demo_MEEF_Optimizer_init config.yaml assets/lsm_mask/ls_image工字型.txt result/MEEF_result/test --optimize 50 --wepe-only
./demo_MEEF_Optimizer_init ../config.yaml ../assets/lsm_mask/ls_image工字型.txt ../result/MEEF_result/test --optimize 50 --wepe-only
```

两种 EP 模式的区别（对应 `MEEFPipelineConfig.optimize_wepe_only`）：


| 模式             | MEEF 矩阵列来源  | 说明                       |
| ---------------- | ---------------- | -------------------------- |
| 全 EP 点（默认） | 所有 edge points | 覆盖完整轮廓               |
| `--wepe-only`    | 仅 WEPE 关键点   | 只在加权关键点上优化，更快 |

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
| `main_cp_mode`     | target_interval | 从目标间隔取点；也可设为 `npy` |
| `main_cps_npy_path` | 空    | `npy` 模式的控制点文件，点为 `(y,x)` |
| `main_cp_interval` | 7      | 主图形控制点采样间隔（像素） |
| `main_symmetry`    | none   | 不使用对称                   |
| `sraf_mask_mode`   | lsm    | 从原 LSM 提取；也可设为 `fitted_txt` |
| `fitted_sraf_txt_path` | 空 | 拟合后的完整 mask 或 SRAF-only 文本 |
| `sraf_cp_interval` | 5      | SRAF 控制点间隔              |
| `sraf_min_cps`     | 8      | SRAF 最少控制点数            |
| `sraf_min_aera`    | 50     | SRAF 最小连通面积            |
| `msaa_level`       | 16     | MSAA 采样数                  |
| `curve_type`       | BS     | B-Spline 插值                |
| `delta`            | 0.15   | 中心差分步长                 |
| `dilate_radius`    | 2      | 主/SRAF 分离膨胀半径         |
| `interval_line`    | 5      | 直线段 EP 间隔               |
| `interval_corner`  | 2      | 拐角 EP 间隔                 |
| `mid_weight`       | 4.0    | 中点 EP 权重                 |
| `other_weight`     | 1.0    | 其他 EP 权重                 |
| `stop_mode`        | fixed_iterations | 跑满 `iter`；`small_step` 可按最大控制点位移提前停止 |
| `step_tol`         | 0.02   | `small_step` 的位移阈值，单位 pixel |
| `patience`         | 3      | `small_step` 需连续满足阈值的轮数 |

`main_cp_mode: npy` 支持 Python 版 `np.save(..., dtype=object)` 产生的不等长多轮廓；
demo 会用项目 `.venv/bin/python`（不存在时用 `python3`）将其转换为输出目录中的
`imported_main_cps.txt`，随后由 C++ 校验边界并读取。`sraf_mask_mode: fitted_txt`
会相对 target 分离出 SRAF，并把该灰度 mask 原样固定用于每次 MEEF 扰动。

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
| `sraf_mask.txt`        | 全部           | 分离出的 SRAF 掩模        |
| `main_mask.txt`        | 全部           | 分离出的主图形掩模        |
| `meef_mx.txt`          | `--build-meef` | X 方向 MEEF 矩阵          |
| `meef_my.txt`          | `--build-meef` | Y 方向 MEEF 矩阵          |
| `lsm_mask.txt`         | `--optimize`   | 初始 LSM 掩模（用于对比） |
| `lsm_wafer.txt`        | `--optimize`   | 初始 LSM 成像结果         |
| `iterations/mask.txt`  | `--optimize`   | 优化后掩模（每轮覆盖）    |
| `iterations/wafer.txt` | `--optimize`   | 优化后成像（每轮覆盖）    |
| `best/`                | `--optimize`   | 最优结果快照              |
| `errors.csv`           | `--optimize`   | 每轮 EPE/WEPE 历史        |
| `meta.json`            | 全部           | 元信息                    |
| `console_output.log`   | 全部           | 本次运行的终端输出副本    |

> `--optimize` 模式下每轮迭代会**覆盖** `iterations/`，不保留中间轮次；最优结果单独存 `best/`。

## 6. 优化后可视化

`--optimize` 模式结束时会自动执行（`demo_MEEF_Optimizer_init.cpp:179`）：

```bash
python3 scripts/show_multi.py \
    <save_path>/lsm_mask.txt gray \
    <save_path>/lsm_wafer.txt seismic \
    <save_path>/iterations/mask.txt gray \
    <save_path>/iterations/wafer.txt seismic \
    --title_prefix 'LSM vs Optimized'
```

在同一个 matplotlib 窗口显示 4 张图：LSM mask / LSM wafer / 优化后 mask / 优化后 wafer。关闭窗口后 demo 退出。

依赖：`python3` + `matplotlib` + `numpy`。若脚本退出状态异常，demo 会打印 warning 但不中断。

## 7. 常见问题

**Q: 只传 `--optimize 50 --wepe-only` 不带前三个参数行吗？**
不行。`--optimize` 会被当成 `config` 路径，报「配置文件不存在: --optimize」。必须按位置补齐前三个参数（或省略全部走默认）。

**Q: `--build-meef` 很慢？**
MEEF 矩阵构建需要 `4 × 控制点数` 次光刻仿真，OpenMP 并行可加速约 4–5 倍。确保 Release 编译且未限制线程数。

**Q: WEPE-only 和全 EP 的结果差异？**
WEPE-only 只优化加权关键点，迭代更快但覆盖范围小；全 EP 覆盖完整轮廓，更全面但每轮 MEEF 矩阵更大。
