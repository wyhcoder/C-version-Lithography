# CTM 梯度融合与 CEPT 对照实验

入口为 `demo/demo_ctm_boundary_ablation.cpp`，配置为 `config_ctm_boundary_ablation.yaml`。四组从同一 Metal target 初始化，分别关闭两项、只开梯度融合、只开 CEPT、同时开启两项。每组独立执行 100 次 theta 更新，不以上一组的结果初始化。

固定参数：1024×1024 target、4×4 Micro、Core 256×256、每侧 Halo 128 px、Guard 30 px、局部光学网格 512×512、8 nm/px、theta 步长 0.9、Core PE 数据损失、CEPT 权重 1、Linear 梯度融合。未开启的损失权重不参与 PE 单项优化。四组共用一次光学准备，`compact_socs` 只裁去 SVD 中间矩阵的零行，不改变局部光学网格或核。成像仍是完整局部 mask 分块成像，之后只回拼 Core。

## 运行

```sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-release --target demo_ctm_boundary_ablation -j4
OMP_NUM_THREADS=4 ./build-release/demo_ctm_boundary_ablation config_ctm_boundary_ablation.yaml
```

可选第二参数覆盖更新次数，第三参数覆盖输出目录，例如：

```sh
OMP_NUM_THREADS=4 ./build-release/demo_ctm_boundary_ablation config_ctm_boundary_ablation.yaml 1 /private/tmp/ctm_ablation_smoke
```

默认输出到 `result/ctm_boundary_ablation/metal_4x4_100`。`final/` 保存各组相同更新次数的结果，作为主对照；`best/` 另存各组总损失最低的状态。开启 CEPT 后总损失含额外项，因此各组最佳轮可能不同；不能直接用各组最佳总损失的大小给优化质量排序。原生产 CTM demo 仍返回并保存最佳结果。

## 图和统计

- `final_mask_comparison.png`：四组灰度 mask 放大图，固定色阶 [0,1]。
- `final_ai_comparison.png`：同一窗口的局部成像 Core 回拼结果，橙色为 AI=0.25 轮廓。
- `final_on_off.png`：同时开启与同时关闭的左右对照，上排 mask，下排 AI。
- `best_*.png`：各组最佳轮的对应图，标题标注实际迭代编号。
- `core_pe_history.png`：共同数据项 Core PE 的迭代曲线，不包含 CEPT。
- `roi_location.png`：在全版图 target 上标出 Core 分界和固定放大窗口。
- `metrics.csv`：全芯片连续性统计和独立 Core PE。
- `roi_continuity.csv`：接缝两侧主线的上下轮廓位置，以及最大边缘错位（px、nm）。
- `REPORT.md`：自动汇总末轮与最佳轮的实际指标，链接图和原始数据。

放大窗口预先固定为全局像素半开区间 y∈[391,409)、x∈[235,277)，参考图中的横向金属线穿过该窗口。右块首列为 x=256，所以黄色虚线画在 x=255.5；青色为 target=0.5 边缘。mask 边缘错位使用 0.5 灰度等值线，AI 边缘错位使用实际曝光阈值 0.25。两侧列索引为 255、256，沿 y 线性插值得到亚像素边缘；轮廓缺失或被窗口截断时记为 nan，不记为零。

全芯片统计区分两件事：

1. **同位置重叠一致性**：完整局部 mask 的本块 Guard 与邻居 Core 在同一全局位置的差异；四邻域按有向像素对统计，包含双方方向，不含角区。RMSE=√(Σ差²/像素对数)。即使关闭 CEPT，也独立测量这些差异；优化日志中的 `cept_loss=0` 只表示关闭该项。
2. **实际回拼接缝跳变**：回拼后的两个相邻 Core 边缘像素之差；公共边只统计一次。原始跳变可能包含 target 的真实变化，因此同时输出相对 Δtarget 的误差、target 两侧相同的跳变，以及两侧均为前景的跳变。

Guard–Core RMSE 降低表示重叠区更一致，不能直接推出回拼 Core 的线条更连续。灰度相邻像素差也不能完全代表几何轮廓连续性，需结合固定窗口的边缘错位和图形观察。`core_pe` 在保存快照时用同一局部成像独立计算，便于核对优化历史中的 PE。

每组保存 `config_used.yaml`、`history.csv`，两个快照各保存全图灰度 `mask.txt`/`mask.bmp` 及 ROI mask/AI/WI 矩阵。全图 TXT 保留双精度；BMP 是 8 位预览，连续性指标使用双精度原矩阵。绘图可独立重跑：

```sh
MPLCONFIGDIR=/private/tmp/litho_mpl .venv/bin/python scripts/plot_ctm_boundary_ablation.py result/ctm_boundary_ablation/metal_4x4_100
```
