# WENO5 与 Godunov 在 Level Set 优化中的用法

## 1. 功能概述

本项目使用 WENO5 计算 Level Set 函数 `phi` 的高阶单边导数，再使用
Godunov 数值 Hamiltonian 组合这些导数，求解法向演化方程：

```text
phi_t + Vn * |grad(phi)| = 0
```

其中：

- `phi` 是 Level Set 函数，零水平集 `phi = 0` 表示掩模轮廓；
- `Vn` 是标量法向速度，本项目中由光刻成像误差梯度 `Gm` 提供；
- WENO5 负责得到高精度、低振荡的单边导数；
- Godunov 格式负责根据局部传播方向组合单边导数；
- CFL 条件负责限制时间步长；
- 二阶 Runge-Kutta 负责时间推进。

相关源码：

- [`source/utils/level_set_utils.cpp`](../source/utils/level_set_utils.cpp)
- [`include/utils/level_set_utils.h`](../include/utils/level_set_utils.h)
- [`source/optimizer/LSM_Optimizer.cpp`](../source/optimizer/LSM_Optimizer.cpp)
- [`tests/test_level_set_godunov.cpp`](../tests/test_level_set_godunov.cpp)

## 2. 为什么使用 WENO5

Level Set 法向演化属于 Hamilton-Jacobi 型问题，信息具有明确的传播方向。
中心差分同时使用上游和下游信息，在拐角、梯度突变和拓扑变化附近容易产生
数值振荡。一阶迎风差分虽然稳定，但数值耗散较大，可能导致拐角变圆、细小
结构消失以及零水平集位置误差。

WENO5 的特点是：

- 在光滑区域接近五阶精度；
- 在不光滑区域自动降低跨越突变模板的权重；
- 能同时构造正、负方向的高阶单边导数；
- 相比一阶迎风格式，对轮廓细节的数值耗散更小。

WENO只负责空间导数重构，整体稳定性还依赖正确的Godunov组合、CFL时间步长
和时间积分格式。

## 3. 三个主要接口

### 3.1 `weno5_1d`

```cpp
static Eigen::VectorXd weno5_1d(
    const Eigen::VectorXd& v,
    int N_orig,
    double dx,
    std::string dir);
```

用于在一个已经扩展边界的一维向量上重构WENO5单边导数。

参数：

| 参数 | 含义 |
|---|---|
| `v` | 已进行边界扩展的一维数据 |
| `N_orig` | 需要输出的原始点数 |
| `dx` | 网格间距 |
| `dir` | `"minus"` 或 `"plus"` |

返回值是长度为 `N_orig` 的一维导数向量。

通常不需要在LSM优化器中直接调用这个函数，而是通过 `der_weno5()` 使用。

### 3.2 `der_weno5`

```cpp
static Eigen::MatrixXd der_weno5(
    const Eigen::MatrixXd& M,
    double dx,
    std::string direction,
    int axis = -1);
```

用于沿矩阵的指定方向计算WENO5单边导数。

参数：

| 参数 | 含义 |
|---|---|
| `M` | 输入矩阵 |
| `dx` | 当前方向的网格间距 |
| `direction` | `"minus"` 表示负方向单边导数，`"plus"` 表示正方向单边导数 |
| `axis` | `0` 表示沿行方向，`1` 表示沿列方向 |

例如，计算x方向的两个单边导数：

```cpp
Eigen::MatrixXd phi_x_minus =
    LevelSetUtils::der_weno5(phi, dx, "minus", 1);

Eigen::MatrixXd phi_x_plus =
    LevelSetUtils::der_weno5(phi, dx, "plus", 1);
```

计算y方向时，可以直接使用 `axis=0`，也可以像当前项目一样先转置，再沿
`axis=1` 计算：

```cpp
Eigen::MatrixXd phi_T = phi.transpose();

Eigen::MatrixXd phi_y_minus =
    LevelSetUtils::der_weno5(phi_T, dy, "minus", 1).transpose();

Eigen::MatrixXd phi_y_plus =
    LevelSetUtils::der_weno5(phi_T, dy, "plus", 1).transpose();
```

### 3.3 `evolve_normal_WENO_godunov`

```cpp
static Evolve_Params evolve_normal_WENO_godunov(
    const Eigen::MatrixXd& phi,
    const Eigen::MatrixXd& Vn,
    double dy,
    double dx);
```

这是LSM优化中推荐直接调用的接口。它完成：

1. 边界扩展；
2. 四个WENO5单边导数的计算；
3. 完整Godunov组合；
4. 计算 `Vn * |grad(phi)|`；
5. 返回CFL所需的速度上界。

输入要求：

- `phi` 和 `Vn` 的行列数必须一致；
- `phi` 不能为空；
- `dx > 0` 且 `dy > 0`。

返回结构：

```cpp
struct Evolve_Params {
    Eigen::MatrixXd delta;
    Eigen::MatrixXd H1_abs;
    Eigen::MatrixXd H2_abs;
};
```

其中：

- `delta = Vn * |grad(phi)|_G`；
- `H1_abs` 和 `H2_abs` 是CFL时间步长使用的保守速度上界；
- 返回矩阵与输入 `phi` 尺寸一致。

## 4. WENO5单边模板

首先计算相邻点的一阶差分：

```text
D1(i) = (v(i+1) - v(i)) / dx
```

对输出点 `i`：

```text
minus 使用 D1(i)   ... D1(i+4)
plus  使用 D1(i+1) ... D1(i+5)，然后反向排列
```

`plus` 模板必须相对 `minus` 向右偏移一个差分位置，才能形成真正的镜像单边
模板。两种方向在反向排列后使用相同的理想权重：

```text
gamma1 = 0.1
gamma2 = 0.6
gamma3 = 0.3
```

三个候选导数为：

```text
d1 =  1/3*v1 - 7/6*v2 + 11/6*v3
d2 = -1/6*v2 + 5/6*v3 +  1/3*v4
d3 =  1/3*v3 + 5/6*v4 -  1/6*v5
```

随后计算三个光滑性指标 `S1`、`S2`、`S3`，并得到非线性权重：

```text
alpha_k = gamma_k / (S_k + epsilon)^2
omega_k = alpha_k / sum(alpha)
```

最终导数为：

```text
D_weno = omega1*d1 + omega2*d2 + omega3*d3
```

某个候选模板越不光滑，它的 `S_k` 越大，最终权重越小。

## 5. Godunov组合

WENO5输出四个单边导数：

```text
Dx-、Dx+、Dy-、Dy+
```

`Vn` 是法向速度，不是x或y方向速度，因此不能简单使用：

```text
Vn > 0：x、y全部选择minus
Vn < 0：x、y全部选择plus
```

必须使用Godunov格式，根据 `Vn` 和各单边导数自身的符号进行组合。

当 `Vn > 0`：

```text
Gx^2 = max(Dx-, 0)^2 + min(Dx+, 0)^2
Gy^2 = max(Dy-, 0)^2 + min(Dy+, 0)^2
```

当 `Vn < 0`：

```text
Gx^2 = min(Dx-, 0)^2 + max(Dx+, 0)^2
Gy^2 = min(Dy-, 0)^2 + max(Dy+, 0)^2
```

然后计算：

```text
|grad(phi)|_G = sqrt(Gx^2 + Gy^2)
delta = Vn * |grad(phi)|_G
```

这样，同一个网格点的x方向可能使用 `minus`，而y方向使用 `plus`，不会把
法向速度的正负错误地当成笛卡尔坐标方向。

## 6. 在LSM优化器中的实际用法

项目在 `_caculate_evolution_term()` 中调用WENO-Godunov：

```cpp
Evolve_Params params =
    LevelSetUtils::evolve_normal_WENO_godunov(
        phi, Gm, _dy, _dx);
```

这里：

```text
phi = 当前Level Set函数
Gm  = 光刻成像误差梯度，作为法向速度Vn
```

随后计算曲率正则项：

```cpp
Eigen::MatrixXd delta_kappa =
    LevelSetUtils::evolve_kappa(phi, _dx, _dy, _b);
```

构造完整空间演化算子：

```cpp
Eigen::MatrixXd Normal = delta_kappa - params.delta;
```

对应近似方程：

```text
phi_t = b*kappa*|grad(phi)| - Vn*|grad(phi)|_G
```

最后通过CFL条件计算时间步长：

```cpp
double dt = LevelSetUtils::get_dt_normal_kappa(
    _cfl,
    _dx,
    _dy,
    params.H1_abs,
    params.H2_abs,
    _b);
```

当前CFL速度上界使用：

```text
H1_abs = |Vn|
H2_abs = |Vn|
```

这是一个略保守但稳定的估计。

## 7. 完整调用示例

下面展示一次Level Set空间演化和二阶Runge-Kutta更新：

```cpp
Eigen::MatrixXd phi = initial_sdf;
Eigen::MatrixXd Vn = gradient;

auto calculate_rhs = [&](const Eigen::MatrixXd& current_phi) {
    Evolve_Params normal =
        LevelSetUtils::evolve_normal_WENO_godunov(
            current_phi, Vn, dy, dx);

    Eigen::MatrixXd curvature =
        LevelSetUtils::evolve_kappa(
            current_phi, dx, dy, curvature_weight);

    Eigen::MatrixXd rhs = curvature - normal.delta;

    double dt = LevelSetUtils::get_dt_normal_kappa(
        cfl,
        dx,
        dy,
        normal.H1_abs,
        normal.H2_abs,
        curvature_weight);

    return std::pair<Eigen::MatrixXd, double>(rhs, dt);
};

auto [rhs_n, dt] = calculate_rhs(phi);
Eigen::MatrixXd phi_1 = phi + dt * rhs_n;

auto [rhs_1, ignored_dt] = calculate_rhs(phi_1);
Eigen::MatrixXd phi_next =
    0.5 * phi + 0.5 * (phi_1 + dt * rhs_1);

Eigen::MatrixXd binary_mask =
    (phi_next.array() >= 0.0).cast<double>().matrix();
```

项目当前实现还会定期重新初始化符号距离函数，防止多轮演化后
`|grad(phi)|` 严重偏离1。

## 8. 法向速度符号

`Vn` 的正负表示沿所定义法线的正向或反向运动，具体对应轮廓扩张还是收缩，
取决于 `phi` 的正负约定。

本项目从二值掩模初始化：

```cpp
phi = reinit_sdf(mask - 0.5, dx, dy);
```

因此通常有：

```text
掩模内部：phi > 0
掩模外部：phi < 0
轮廓边界：phi = 0
```

判断扩张或收缩时，应同时结合此正负约定、演化方程中的正负号以及
`grad(phi)` 的方向，不能孤立地说“`Vn > 0` 一定向外”。

## 9. 测试方法

重新配置并构建测试：

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON

cmake --build build-release \
  --target test_level_set_godunov \
  --parallel
```

只运行WENO-Godunov测试：

```bash
ctest --test-dir build-release \
  -R '^level_set_godunov$' \
  --output-on-failure
```

运行全部回归测试：

```bash
ctest --test-dir build-release --output-on-failure
```

测试覆盖：

- 正、负斜率平面的导数和Hamiltonian；
- 正、负法向速度；
- 一维极值点处的Godunov选择；
- 圆形零水平集窄带的左右、上下镜像对称性；
- 非方形矩阵；
- 输入尺寸错误。

## 10. 与旧Python实现的区别

当前C++实现已经使用完整Godunov组合，并修正了WENO `plus` 模板的差分位置
和权重镜像问题。旧Python实现仍采用“只根据 `Vn` 选择一整套
`minus/plus`”的简化方式，并且 `plus` 模板与当前C++不同。

因此，两者现在不会逐元素完全一致。如果需要继续进行Python/C++数值对照，
Python端也应同步修改WENO正向模板和Godunov组合。

## 11. 面试简述

可以这样介绍：

> 项目使用Level Set优化掩模轮廓，演化方程属于Hamilton-Jacobi型方程。
> 我使用WENO5分别计算x、y方向的正负单边导数，在光滑区域获得高阶精度，
> 在拐角和梯度突变区域通过非线性权重抑制振荡。由于法向速度不是x或y方向
> 速度，不能只根据速度正负统一选择单边导数，所以进一步采用Godunov数值
> Hamiltonian，对四个单边导数进行局部max/min组合。最后通过CFL条件控制
> 时间步长，并使用二阶Runge-Kutta更新Level Set函数。
