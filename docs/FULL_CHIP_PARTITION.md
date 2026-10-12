# 在完整版图上显示 Micro 分区

`demo_full_chip_partition` 读取 CTM 或 LSM 配置，调用优化流程相同的 `Partition::build()`，
导出真实 `PartitionMeta` 的坐标并画在完整 target 上。这个入口只生成分区图，不执行光学初始化或优化。

```bash
cmake --build build-release --target demo_full_chip_partition --parallel 4
./build-release/demo_full_chip_partition config_full_chip_lsm.yaml
```

默认输出目录为 `result/partition/<target名称>_<分块行数>x<分块列数>/`：

- `partition.png`：整张版图上显示所有 Micro 的三个区域边界和编号。
- `partition_micro_36.png`：8×8 配置默认在同一张完整版图上高亮 Micro 36，保留其余 Core 分区。
- `micro_regions.csv`：每块的全局 Core、Core＋Guard、Core＋Halo 坐标。
- `partition_meta.csv`：全图尺寸、分块数、Halo/Guard 宽度、局部网格尺寸和 nm/像素。
- `target.bmp`：本次配置的原始目标版图。

绿色实线是 Core 边界，橙色虚线是 Core＋Guard 外边界，蓝色虚线是完整 Core＋Halo 成像边界。
高亮图中橙色环带表示 Guard，蓝色环带表示 Guard 外侧的 Halo 部分；**Guard 本身也属于 Halo**。
Core 只由本块拥有，最终回拼仅使用 Core；各块的 Guard/Halo 可以覆盖邻居 Core。

所有坐标均为全局像素索引，矩形为 `[y0,y1) × [x0,x1)`，不包含结束索引。
像素中心位于整数坐标，绘图边界位于 `x0-0.5` 和 `x1-0.5` 等像素外缘。
图中芯片外侧的 Halo 保留越界坐标；实际数据由 `extract_local_tile()` 周期取值，边缘没有额外 Micro 邻居。

当前 1024×1024、8×8 配置，每块 Core 为 128×128，Core＋Guard 为 188×188（Guard=30），
完整局部成像范围为 256×256（每侧 Halo=64）。非整除/非方形版图使用实际导出的 tile 范围，
也会显示为了统一偶数网格而增加的成像上下文。

指定输出目录和高亮编号：

```bash
./build-release/demo_full_chip_partition config_full_chip_lsm.yaml result/partition/metal_micro0 0
```

已有导出数据可直接切换高亮的 Micro，无须重新构建或优化：

```bash
MPLCONFIGDIR=/tmp/litho-matplotlib .venv/bin/python scripts/plot_full_chip_partition.py result/partition/metal_3_0_only_1024_8x8 --micro-id 27
```
