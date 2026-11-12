# SRAF printing：直接检测 AI，逐块优化控制点

入口 `demo_sraf_printing_repair` 读取 MEEF 快照，直接检查背景范围内的 `AI>resist.threshold`，匹配对应 SRAF。
每次只锁定一块 SRAF，主图形和其他 SRAF 保持固定。YAML 可选择：

- `single_sraf_normal_gradient`：本次新增，全部有效控制点沿法线做中心差分，再联合向内梯度下降。
- `single_sraf_multi_pair`：原几何方法，每轮重新选择宽点对联合内缩；省略 mode 时仍使用原方法。

## 代码与运行

调用顺序：[配置](../config_sraf_printing_repair.yaml) → [入口](../demo/demo_sraf_printing_repair.cpp) →
[接口](../include/utils/sraf_printing_repair.h) / [几何实现](../source/utils/sraf_printing_repair.cpp)。
[测试](../tests/test_sraf_printing_repair.cpp)负责检测、联合位移和几何保护；[绘图](../scripts/plot_sraf_printing_repair.py)读取本方法的输出。

在项目根目录执行：

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target demo_sraf_printing_repair test_sraf_printing_repair -j 4
ctest --test-dir build-release --output-on-failure -R '^sraf_printing_repair$'
OMP_NUM_THREADS=8 ./build-release/demo_sraf_printing_repair --config config_sraf_printing_repair.yaml
```

Demo 写完全部数值记录后自动调用 `scripts/plot_sraf_printing_repair.py`，一次运行即可生成 `comparison.png`。
图片直接保存在 YAML 的 `output_dir` 中，重复运行覆盖本次生成的同名文件，不自动追加版本后缀。
自动调用优先使用项目 `.venv/bin/python`，不存在时使用 `python3`，需要已安装 NumPy 和 Matplotlib。
绘图失败会保留数值结果并返回非零退出码，日志会给出原因。
光学参数和显影阈值来自 `source_dir/config_used.yaml`，不读取当前根目录的 `config.yaml`。
入口使用 CR SRAF、MSAA 光栅化和 PyTorch EPE。

## AI 检测阈值与安全目标

printing 检测、统计和选块直接使用源配置的 `resist.threshold`，判据为严格 `AI>threshold`。
本块锁定后的停止目标为 `safe_threshold=threshold-intensity_margin`，余量直接作用于 AI。
例如 `intensity_margin=0.02`、源 `threshold=0.25` 时，AI>0.25 触发处理，本块 AI 峰值≤0.23 时结束。
实际目标以运行时 YAML 的余量为准。
未发生 AI 超阈值的其他 SRAF 不会仅因未达到安全目标而主动处理。

`wi_limit`、WI 反解 AI 的函数及相关输出已移除。WI 仅用于保存、统计峰值和绘图，不参与修复决策。
成像和主图形 EPE 继续使用源配置中的 threshold 和 alpha。

## 原几何方法

1. 由源快照分离固定主图形，重绘 SRAF 并完整成像，检查与保存快照的一致性。
2. 按连续轮廓包含关系，将外轮廓及其一层孔洞分为一块，多层包含输入会报告错误。
3. 直接扫描连续 AI 的超阈值位置，按初始材料覆盖匹配尚未完成的 SRAF；有多块时优先选择可归属峰值最高者。
4. 锁定本块，在 1/8 pixel 网格上通过最大内切圆估计最宽截面的尺度和位置。
5. 用本轮控制点拟合周期向心 CR，求各控制点处的解析切线及材料内法线，检测法线厚度。
6. 筛选相对的宽点对，要求法线朝向彼此、连接线位于材料内部，排除长轴误判。组间不共享控制点，中点接近且方向夹角小于 30 度的截面不重复选取。
7. 所有点对基于同一份移动前控制点联合内缩，每组两点等距；几何失败时，各组允许独立减半或暂缓。试探不提前改变实际状态。
8. 对最终联合候选完整成像，几何、强度和 EPE 均满足条件后接受一次更新。下一轮重新拟合、重新测宽、重新选组。
9. 本块 AI 峰值达到 `threshold-intensity_margin` 后结束本块；也可能因上限、无有效组或所有步长失败而停止，再检测下一块。

未发生 printing 的其他 SRAF 不会主动选择；无法归属的超阈值位置报告未解决。
本方法是几何启发式修复，不计算 AI 损失对控制点的梯度。

## 法向有限差分梯度方法

入口中的代码按四个步骤展开，不使用新的优化器类或第三方优化库：

1. 从当前锁定 SRAF 的全部控制点计算周期 CR 单位内法线，固定本轮法线和本阶段背景窗口 W。
2. 每个点分别移动 `+fd_step_px*n`（向内）和 `-fd_step_px*n`（向外），重新拟合、完整光栅化并成像。
3. 计算中心差分，再基于同一份旧坐标联合向内更新。第一版差分串行执行。
4. 候选只检查周期 CR 闭合和采样曲线自交，并要求本块损失严格下降；不通过则整体减半实际位移。

```text
L = mean_W(max(AI-safe_threshold, 0)^2)
gradient_i = (L_plus_i-L_minus_i)/(2*fd_step_px)
requested_i = min(max(-learning_rate*gradient_i, 0), step_px)
P_new_i = P_old_i + step_scale*requested_i*inward_normal_i
```

W 在锁定本块后保持固定，包含本块附近的背景，不只包含初始超阈值像素。
差分扰动只检查被扰动曲线不自交；向外扰动不能套用“面积必须减小”的接受规则。
若双侧曲线无效，则将该点的差分距离减半重试；两侧都有效后才记录导数。
实际更新仅保留内向下降分量，每点按 `step_px` 限幅，不再施加厚度、面积、孔洞数或栅格连通性限制。
全局回退只缩放实际位移，不重新计算本轮法线及梯度；提交后下一轮才重新计算。
梯度方法要求损失下降，不要求每次 AI 峰值单调下降；本块停止仍检查 AI 峰值，避免均值掩盖单点超限。

每块有 m 个有效控制点时，通常每轮需要 2m 次差分成像，候选评价和无效扰动重试另计。
MSAA 使数值目标存在离散台阶，不能把差分距离设置得越小越好。
周期 CR 使用循环索引自动闭合，输出采样点列表不重复首点，检查时不能要求采样首尾点恰好相等。
梯度方法的 EPE 和全局 printing 仅统计，不作为回退条件；损失不下降或曲线自交时仍会回退。

## 检测范围与接受条件

printing 判据为严格 `AI>threshold`，选块接口直接接收 AI、源阈值和背景检测范围。
WI 不参与二值化检测，仅保存连续矩阵并统计背景峰值用于核查。
背景范围是主图形 Guard 外，另补入 Guard 内不与 target 重叠的独立 AI 超阈值连通域。
因此全背景统计不等于全部 `target<=0.5` 像素；后者单独记录为 `raw_background_printed_px`。

本块评价窗口为初始材料按每轴 `[-3,3]` pixel 外扩，再排除 Guard；本块内独立超阈值区域仍保留。
窗口只用于评价，成像输入始终是完整主图形与全部 SRAF。

两种方法均不检查 mask 二值化后的材料块数、孔洞数或 4/8 邻域连接，也不因这些数量变化回退。
原几何方法检查采样曲线自交、灰度覆盖面积、EPE 和全局 printing；选组与内缩步长规则继续沿用。
新梯度方法按用户要求仅以周期 CR 闭合和采样曲线不自交作为几何条件，并要求平方超限损失下降。
厚度、面积、栅格连通性、EPE 和全局 printing 不作为新梯度方法的回退条件，其中展示指标继续记录。
用于寻找独立 printing 区域的 AI 超阈值连通域分析继续使用，它不约束 SRAF mask 的连接情况。

## YAML 参数

所有轮廓和控制点编号从 0 开始；坐标为整图 `(y,x)` 浮点像素坐标。本例 1 pixel = 6 nm。

| 参数 | 含义 |
| --- | --- |
| `source_dir` / `snapshot` | 原始 MEEF 目录及使用的 `best_wepe` 或 `best_epe` 快照 |
| `output_dir` | 固定输出目录，重复运行覆盖同名文件；相对路径基于项目根目录 |
| `mode` | `single_sraf_normal_gradient` 为新方法，`single_sraf_multi_pair` 为原方法；省略时保留原方法 |
| `normal_gradient.fd_step_px` | 双侧法向中心差分距离，单位 pixel，当前 YAML 为 0.05，省略时默认 0.2 |
| `normal_gradient.learning_rate` | 损失梯度转换成内向请求位移的系数，当前为 1000；随后仅按 step_px 限幅 |
| `pair_groups.maximum_pairs` | 仅几何方法：每轮联合更新的点对数上限，本例为 4 |
| `pair_groups.minimum_center_spacing_px` | 仅几何方法：重复截面距离阈值，配合 30 度方向条件去重，单位 pixel |
| `max_iterations` | 全局循环上限，包含结束块和选块检查的轮次 |
| `max_iterations_per_sraf` | 每块接受的更新轮数上限 |
| `step_px` | 每个控制点的最大法向位移，单位 pixel |
| `max_backtracks` | 全局及每组局部试探各自最多减半次数；配置 6 表示每层最多试 7 个步长 |
| `width_threshold_px` | 仅几何方法：最宽截面直径、控制点法线厚度及点对距离的筛选阈值，单位 pixel |
| `minimum_width_px` | 仅原几何方法：剩余厚度保护，单位 pixel；新梯度方法不使用 |
| `intensity_margin` | 本块 AI 停止目标相对源显影阈值降低的余量，默认 0.01；须非负且小于源 threshold |
| `target_guard_px` | 主图形排除带的正方形膨胀半径，单位 pixel；3 对应每轴含端点的 `[-3,3]` 邻域 |
| `max_mean_epe_increase_nm` / `max_max_epe_increase_nm` | 仅原几何方法：允许的平均/最大绝对 EPE 增量，单位 nm |
| `max_added_sraf_area_px` | 仅原几何方法：新增灰度覆盖面积上限，单位 pixel² |
| `snapshot_tolerance` | 初始化重绘和重新成像与源快照的最大绝对差值容差 |

每组初始请求位移取 `min(本轮名义步长, 0.45*(有效宽度-minimum_width_px))`。
有效宽度是点对距离及两端法线厚度的最小值；几何回退还可进一步减小该组位移。

## 原几何方法的历史结果（余量 0.01）

以下为原几何方法在 `intensity_margin=0.01` 时的历史数值，不是新梯度方法的结果。
只处理 SRAF18，三轮分别接受 4、3、2 组，即 8、6、4 个控制点的联合更新。

| 指标 | 初始 | 最终 |
| --- | ---: | ---: |
| 检测范围内 AI>0.25 的背景像素 | 57 | 0 |
| SRAF18 本块 AI 峰值 | 0.329906316621 | 0.233928539500 |
| 检测范围内背景 AI 峰值 | 0.329906316621 | 0.246720201837 |
| 平均绝对 EPE（nm） | 0.146878506985 | 0.146978073476 |
| 最大绝对 EPE（nm） | 0.856976781512 | 0.856373480963 |

本块达到 0.24 的 AI 安全目标，全背景低于显影阈值 0.25，`printing_cleared=true`。
其他未发生 printing 的位置仍有 0.246720 峰值，因此全背景的 `safe_threshold_reached=false`。
程序退出码 0 表示数值记录和自动绘图已完成，是否满足修复目标应查看 summary 中的状态。

## 此前梯度方法的验证结果（余量 0.02）

输出固定为 `result/printing_repair/matrix_0003_normal_gradient`。
本次使用 `fd_step_px=0.05`、`learning_rate=1000`、`intensity_margin=0.02`。
按用户要求移除了梯度方法中的栅格连接、孔洞数、厚度、面积和 EPE 拒绝条件，只检查周期 CR 闭合、自交及损失下降。

- 只锁定 SRAF18，38 轮更新全部以系数 1 接受，没有因几何条件缩小位移。
- 第 32 轮检测范围内 AI>0.25 的像素清零，初始为 57，最终为 0。
- 损失从 2.29508336943e-4 降至 1.43743877050e-6；最终本块 AI 峰值为 0.246576697745。
- 第 39 轮所有试探步长都不能继续降低损失，因此结束本块；全背景 AI 峰值为 0.246715340689。
- `printing_cleared=true`，但尚未达到 0.23 的安全目标，`safe_threshold_reached=false`。
- 变化的 CR 采样曲线闭合且无严格交叉；像素材料块数可变化，本次 8 邻接块数为 30→31，该指标仅统计。

第一轮 26 个法向导数逐点打印到控制台，同时保存到 `first_gradient.csv`。
原始梯度不包含学习率或回退系数，正方向为材料内法线，负梯度对应内向下降。

## 输出文件

- `initial/`、`final/`：控制点、实际拟合曲线采样、SRAF/完整 mask、AI、WI、EPE 和指标。`sraf_curves.txt` 显式重复首点闭合。
- `sraf_regions.csv`、`sraf_stages.csv`：轮廓归属、逐块处理顺序、峰值及停止原因。
- `history.csv`、`trials.csv`、`movements.csv`：每轮状态、步长试探及逐点真实位移。
- `group_history.csv`：每组的两点、宽度、请求/实际步长；0 表示暂缓。组编号仅在本轮有效。
- `pair_history.csv`：实际移动组的点对与本块峰值，同一轮可有多行。
- `summary.yaml`、`config_used.yaml`、`source_config_used.yaml`、`console_output.log`：状态、配置及运行记录。
- `summary.yaml`：`threshold` 为源 AI 检测阈值，`intensity_margin` 为直接 AI 余量，`safe_threshold` 为本块 AI 停止目标。
  `resist_alpha` 和初始/最终背景 WI 峰值仅记录成像参数及展示指标。
  材料块数、孔洞数及 4/8 邻域连接是否保持的五项字段已删除，不再执行这类栅格检查。
- `metrics.txt` 新增 `background_wi_peak`；`background_printed_px` 等超阈值统计采用 `AI>threshold` 的口径。
- `comparison.png`：全图连续 WI、热点局部 WI、WI 剖面及实际控制点曲线前后对照。前后固定 `[0,1]` 色阶，红线 `WI=0.5` 仅为显示参考。
- `convergence.png`：背景 AI 峰值、AI 检测阈值/安全目标和主图形 EPE 的全局收敛过程。
- `wafer_comparison.png`：初始热点附近真实连续 WI 对照，前后使用相同 `[0,1]` 色阶，并标出 `WI=0.5` 参考线和剖面。
- `final_parametric_curves.png`、`sraf_hotspot_parametric_curve.png`：完整曲线和热点曲线对照，直接读取 C++ 导出的拟合采样点。
- `initial_pair_groups.png`、`group_updates.png`、`pair_updates.png`：首轮选组及逐轮联合更新过程。
- `gradient_history.csv`：新方法逐点的法线、实际差分距离、双侧损失、导数、请求/实际位移和有效性。
- `first_gradient.csv`：第一轮原始法向导数和请求位移，控制点编号从 0 开始；控制台也逐点打印。
- `loss_history.csv`：新方法每轮损失、整体回退系数、实际移动点数、评价峰值和停止原因，失败轮也记录。
- `loss_windows/stage_N.txt`：新方法每阶段固定的整图 0/1 评价窗口，阶段编号从 1 开始，与 CSV 对应。
- `gradient_updates.png`：新方法按 SRAF 分阶段展示损失下降及每轮实际移动点数。
- `normal_gradient_verification.json`：梯度、位移、损失和原几何方法兼容性核查。

兼容性和验证边界见 [兼容风险](sraf_printing_repair_compatibility.md)。
