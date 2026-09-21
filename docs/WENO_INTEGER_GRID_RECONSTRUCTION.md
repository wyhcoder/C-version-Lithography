# WENO5 在整数网格上的重构

本文只使用整数数组下标，解释当前项目中的 WENO5 如何从 `phi[i]` 得到整数
节点 `i` 上的左右单边导数。

对应源码：

- [`source/utils/level_set_utils.cpp`](../source/utils/level_set_utils.cpp)
- `LevelSetUtils::weno5_1d()`
- `LevelSetUtils::der_weno5()`

## 1. 输入数据

假设一维 Level Set 数据存放为：

```text
phi[0], phi[1], phi[2], ..., phi[N-1]
```

这里每个 `phi[i]` 都位于整数网格点 `i`。如果网格间距是一个像素，则相邻
节点的距离就是：

```text
dx = 1 pixel
```

WENO5 最终要在每个整数节点 `i` 上计算两个导数：

```text
D_minus[i]：从左侧信息重构得到的导数
D_plus[i] ：从右侧信息重构得到的导数
```

## 2. 先计算相邻整数节点的差分

首先建立差分数组 `q`：

```text
q[j] = (phi[j+1] - phi[j]) / dx
```

例如：

```text
q[0] = (phi[1] - phi[0]) / dx
q[1] = (phi[2] - phi[1]) / dx
q[2] = (phi[3] - phi[2]) / dx
```

`q[j]` 始终由整数节点 `j` 和 `j+1` 计算，仍然使用普通整数数组保存。

源码中的 `D1` 就是这里的差分数组：

```cpp
D1(i) = (v(i + 1) - v(i)) / dx;
```

## 3. 在整数节点 `i` 上选择五个差分

计算左侧单边导数 `D_minus[i]` 时，选择：

```text
q[i-3], q[i-2], q[i-1], q[i], q[i+1]
```

计算右侧单边导数 `D_plus[i]` 时，选择镜像排列：

```text
q[i+2], q[i+1], q[i], q[i-1], q[i-2]
```

可以直观地画成：

```text
phi[i-3] phi[i-2] phi[i-1] phi[i] phi[i+1] phi[i+2] phi[i+3]
    o---------o---------o-------o-------o---------o---------o
       q[i-3]    q[i-2]  q[i-1]  q[i]    q[i+1]    q[i+2]
                                  ↑
                              目标节点 i
```

因此：

- `minus` 模板整体更偏向目标节点左侧；
- `plus` 模板整体更偏向目标节点右侧；
- 两个结果都对应同一个整数节点 `i`。

## 4. 五个差分组成三个候选模板

把某一方向选出的五个差分依次记为：

```text
v1, v2, v3, v4, v5
```

然后构造三个候选导数：

```text
d1 =  1/3*v1 - 7/6*v2 + 11/6*v3
d2 = -1/6*v2 + 5/6*v3 +  1/3*v4
d3 =  1/3*v3 + 5/6*v4 -  1/6*v5
```

三个候选模板分别使用：

```text
d1 使用 v1, v2, v3
d2 使用     v2, v3, v4
d3 使用         v3, v4, v5
```

也就是把五个连续差分拆成三个互相重叠的三点模板。

## 5. 根据光滑程度计算权重

代码分别计算三个候选模板的光滑性指标：

```text
S1：模板 v1, v2, v3 的不光滑程度
S2：模板 v2, v3, v4 的不光滑程度
S3：模板 v3, v4, v5 的不光滑程度
```

理想权重为：

```text
g1 = 0.1
g2 = 0.6
g3 = 0.3
```

非线性权重为：

```text
a1 = g1 / (S1 + epsilon)^2
a2 = g2 / (S2 + epsilon)^2
a3 = g3 / (S3 + epsilon)^2

w1 = a1 / (a1 + a2 + a3)
w2 = a2 / (a1 + a2 + a3)
w3 = a3 / (a1 + a2 + a3)
```

最后得到整数节点 `i` 上的 WENO 导数：

```text
D_weno[i] = w1*d1 + w2*d2 + w3*d3
```

如果某个候选模板跨过尖角或数值突变，它的 `S` 会变大，最终权重会自动
减小。光滑区域中三个模板按接近理想权重组合，从而获得高阶精度。

## 6. 具体例子：计算节点 `i = 10`

先计算整数差分：

```text
q[7]  = (phi[8]  - phi[7])  / dx
q[8]  = (phi[9]  - phi[8])  / dx
q[9]  = (phi[10] - phi[9])  / dx
q[10] = (phi[11] - phi[10]) / dx
q[11] = (phi[12] - phi[11]) / dx
q[12] = (phi[13] - phi[12]) / dx
```

计算 `D_minus[10]` 时：

```text
v1 = q[7]
v2 = q[8]
v3 = q[9]
v4 = q[10]
v5 = q[11]
```

计算 `D_plus[10]` 时：

```text
v1 = q[12]
v2 = q[11]
v3 = q[10]
v4 = q[9]
v5 = q[8]
```

然后分别把两组 `v1...v5` 送入相同的候选模板、光滑性指标和非线性权重
公式，即可得到：

```text
D_minus[10]
D_plus[10]
```

## 7. 为什么源码中的 `D1` 下标看起来有所不同

为了防止边界越界，`der_weno5()` 会在两端各复制三个节点。原始数据：

```text
phi[0], phi[1], ..., phi[N-1]
```

扩展后变成：

```text
phi[0], phi[0], phi[0], phi[0], phi[1], ..., phi[N-1], phi[N-1], phi[N-1], phi[N-1]
```

因此原始 `phi[i]` 在扩展数组中的位置是 `v[i+3]`，源码里的差分数组满足：

```text
D1[p] 对应原始差分 q[p-3]
```

所以源码中的：

```text
minus: D1[i], D1[i+1], D1[i+2], D1[i+3], D1[i+4]
plus : D1[i+5], D1[i+4], D1[i+3], D1[i+2], D1[i+1]
```

换回原始整数网格就是：

```text
minus: q[i-3], q[i-2], q[i-1], q[i],   q[i+1]
plus : q[i+2], q[i+1], q[i],   q[i-1], q[i-2]
```

## 8. 二维矩阵中的计算

对于二维 Level Set 矩阵 `phi(row, col)`：

- x 方向：固定一行，对这一行使用上面的一维 WENO5；
- y 方向：固定一列，对这一列使用同样的一维 WENO5；
- 每个整数像素 `(row, col)` 最终得到四个单边导数。

```text
Dx_minus(row, col)
Dx_plus(row, col)
Dy_minus(row, col)
Dy_plus(row, col)
```

随后 Godunov 格式根据法向速度和导数符号组合这四个值。WENO 重构完成后，
输出矩阵的尺寸和整数像素坐标都与输入 `phi` 完全一致。

## 9. 整数网格版伪代码

```text
for j = 0 ... N-2:
    q[j] = (phi[j+1] - phi[j]) / dx

for i = 0 ... N-1:
    left  = [q[i-3], q[i-2], q[i-1], q[i],   q[i+1]]
    right = [q[i+2], q[i+1], q[i],   q[i-1], q[i-2]]

    D_minus[i] = WENO_combine(left)
    D_plus[i]  = WENO_combine(right)
```

边界处缺少的 `q` 由三层边界复制补齐。

## 10. 一句话理解

在整数网格上，WENO5 先计算相邻整数节点的差分 `q[j]`，再围绕目标整数节点
`i` 选择左右偏置的五个差分，通过三个候选模板和非线性权重，最终得到同一
整数节点上的左右单边导数 `D_minus[i]` 和 `D_plus[i]`。
