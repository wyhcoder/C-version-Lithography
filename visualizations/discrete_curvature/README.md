# 离散轮廓三点曲率可视化

这个目录用于单独演示：轮廓只有离散点时，怎样利用相邻三个点估计中间点曲率，以及怎样让高曲率区域获得更密集的采样点。

该脚本不会修改项目现有算法。当前项目的 `EpSelect::extract_mask_control_points()` 使用固定间隔抽样；这里的“按曲率取点”是一个独立、可验证的示例。

## 数学定义

对轮廓中间点 `P[i]`，取前后两个邻点：

```text
P_prev = P[i - s]
P      = P[i]
P_next = P[i + s]
```

三角形边长为：

```text
a = |P - P_prev|
b = |P_next - P|
c = |P_next - P_prev|
```

设三角形面积为 `A`，则外接圆半径和曲率为：

```text
R       = a*b*c / (4*A)
|kappa| = 1/R = 4*A / (a*b*c)
```

利用二维叉积可以直接计算带符号曲率：

```text
kappa = 2 * cross(P - P_prev, P_next - P) / (a*b*c)
```

逆时针轮廓通常得到正号，顺时针轮廓通常得到负号；取点密度一般使用 `|kappa|`。

这里的 `s` 是 `--neighbor-step`。`s=1` 使用紧邻点，位置最局部但对像素噪声敏感；增大 `s` 会更平滑，但可能抹掉很小的拐角。

## 直接运行内置示例

从项目根目录运行：

```bash
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python \
  visualizations/discrete_curvature/visualize_discrete_curvature.py
```

脚本使用一个闭合圆角矩形，生成：

- `output/discrete_curvature_visualization.png`：四幅联图；
- `output/curvature_values.csv`：每个轮廓点的坐标、曲率、曲率半径和是否被选中。

图中四个部分分别表示：

1. 整个离散轮廓的曲率分布；
2. 一个点及其前后邻点确定的三角形和外接圆；
3. 曲率随轮廓点编号的变化；
4. 一个简单的曲率加权取点结果，高曲率弯曲区域取点更密。

## 读取项目轮廓

输入文件需要至少三行、每行两个数。默认按照本项目的 `(y, x)` 顺序读取：

```bash
MPLCONFIGDIR=.cache/matplotlib .venv/bin/python \
  visualizations/discrete_curvature/visualize_discrete_curvature.py \
  --input path/to/contour.txt \
  --coordinate-order yx \
  --neighbor-step 3 \
  --selected-count 28
```

如果文件列顺序是 `(x, y)`，使用 `--coordinate-order xy`。对于首尾不相连的开放曲线，再增加 `--open`。

## 取点策略说明

示例采用曲率加权的累计采样：

```text
weight[i] = 1 + strength * |kappa[i]| / max(|kappa|)
```

然后在累计权重轴上等间隔取点。因此直线段仍会保留少量点，弯曲区域会自动获得更多点。这只是方便理解的策略，并不是唯一方案。工程中还可以加入：最小点间距、强制保留角点、曲率阈值、非极大值抑制以及轮廓长度约束。

## 数值边界

- 三点共线时 `A=0`、`kappa=0`、`R=inf`。
- 重复点会导致边长乘积接近零，本脚本把该位置曲率置零。
- 像素轮廓可能有锯齿，建议比较多个 `neighbor-step`，不要只看单一结果。
- 三点曲率是分配给中间点的局部估计，不代表孤立离散点本身存在严格微分曲率。
