# 二维条件下 PE Loss 对 Mask 的梯度推导

本文推导当前工程中二维 mask 的像素误差（Pixel Error，PE）对 mask 的梯度，并与
Imaging::compute() 和 Gradient::pe_gradient() 的实现逐步对应。

推导目标是得到一个与 mask 尺寸相同的实数矩阵：

$$
\nabla_M L_{\mathrm{PE}}
=
\left[
\frac{\partial L_{\mathrm{PE}}}{\partial M(u,v)}
\right]_{N\times N}.
$$

当前实际输入中 $N=257$，因此：

$$
M\in\mathbb R^{257\times257},
\qquad
\nabla_M L_{\mathrm{PE}}\in\mathbb R^{257\times257}.
$$

---

## 1. 下标、尺寸与运算约定

为避免把空间像素、mask 像素和 SOCS 核混在一起，本文固定使用：

- $(i,j)$：像面坐标，即空中像、wafer 和 target 的行列下标。
- $(u,v)$：mask 平面的行列下标。
- $k$：Abbe 光源点或 SOCS 相干模态的编号。

主要变量如下。

| 符号 | 含义 | 尺寸/类型 |
|---|---|---|
| $M(u,v)$ | 实数 mask | $N\times N$，实数 |
| $\widehat M$ | mask 的二维 FFT | $N\times N$，复数 |
| $K_k$ | 第 $k$ 个频域光学核 | $N\times N$，复数 |
| $E_k(i,j)$ | 第 $k$ 个核产生的复振幅场 | $N\times N$，复数 |
| $I(i,j)$ | 空中像强度 | $N\times N$，实数 |
| $W(i,j)$ | 光刻胶显影结果 | $N\times N$，实数 |
| $T(i,j)$ | target | $N\times N$，实数 |
| $q_k$ | 第 $k$ 个核的权重 | 标量 |
| $S$ | 原始光源权重总和 | 标量 |

本文使用以下运算符。

### 1.1 逐元素乘法

$$
X\odot Y
$$

表示两个同尺寸数组逐位置相乘：

$$
(X\odot Y)(i,j)=X(i,j)Y(i,j).
$$

对应 Eigen 代码：

~~~cpp
X.array() * Y.array()
~~~

本文所有 $N\times N$ 光学核与频谱之间的乘法都是逐元素乘法，不是普通矩阵乘法。

### 1.2 二维复数 Frobenius 内积

$$
\langle X,Y\rangle_F
=
\sum_{i,j}\overline{X(i,j)}Y(i,j).
$$

它对全部行列求和，结果是一个复数标量。代码等价形式是：

~~~cpp
(X.conjugate().array() * Y.array()).sum()
~~~

### 1.3 FFT 与 IFFT

记：

$$
\mathcal F(X)=\operatorname{FFT2}(X),
\qquad
\mathcal F^{-1}(X)=\operatorname{IFFT2}(X).
$$

这里的 $\mathcal F^{-1}$ 包含 $1/N^2$ 归一化。FFTW 的 backward 变换本身不归一化，
因此当前代码在 IFFT 后乘：

$$
\frac{1}{N^2}.
$$

为简化公式，本文把 ifftshift 和 fftshift 包含在变换的坐标约定中。当前代码显式执行：

- 空间域中心布局进入 FFT 前：ifftshift。
- IFFT 回到空间域后：fftshift。
- $K_k$ 以 DC 位于角点的频域布局保存。

---

## 2. 二维前向成像模型

### 2.1 第 $k$ 个复振幅场

第 $k$ 个光学核下：

$$
\widehat M=\mathcal F(M),
$$

$$
\widehat E_k=\widehat M\odot K_k,
$$

$$
\boxed{
E_k
=
\mathcal F^{-1}
\left[
\mathcal F(M)\odot K_k
\right].
}
$$

定义第 $k$ 个前向光学算子：

$$
\boxed{
A_k[X]
=
\mathcal F^{-1}
\left[
\mathcal F(X)\odot K_k
\right].
}
$$

因此：

$$
E_k=A_k[M].
$$

$A_k[X]$ 表示“对二维数组 $X$ 做 FFT、与 $K_k$ 逐元素相乘、再做 IFFT”，
不是 $A_k$ 与 $X$ 的普通二维矩阵乘法。

对应 Imaging::compute()：

~~~cpp
local_temp = _mask_frequency.array() * kernels_frequency[k].array();
fftw_execute_dft(_plan_inv, local_in, local_out);
~~~

### 2.2 空中像强度

定义：

$$
q_k=
\begin{cases}
w_k, & \text{Abbe 模式},\\
\lambda_k, & \text{SOCS 模式}.
\end{cases}
$$

其中 SOCS 的 $\lambda_k$ 是 TCC/SVD 相干模态的特征值。

空中像为：

$$
\boxed{
I(i,j)
=
\frac{1}{S}
\sum_k q_k |E_k(i,j)|^2,
}
$$

其中：

$$
S=\sum_s w_s.
$$

写成二维数组形式：

$$
\boxed{
I
=
\frac{1}{S}
\sum_k q_k
\left(
E_k\odot\overline{E_k}
\right).
}
$$

这里的 $\sum_k$ 只对核编号求和，不对图像行列求和。对于固定的 $(i,j)$，
代码将所有核在该像素处的强度累加起来。

### 2.3 光刻胶模型

$$
\boxed{
W(i,j)
=
\sigma\left(\alpha[I(i,j)-\tau]\right)
=
\frac{1}{1+\exp\{-\alpha[I(i,j)-\tau]\}}.
}
$$

其中 $\tau$ 是 resist threshold，$\alpha$ 是 sigmoid 陡峭程度。

### 2.4 PE loss

$$
\boxed{
L_{\mathrm{PE}}
=
\sum_{i,j}
\left[
W(i,j)-T(i,j)
\right]^2.
}
$$

PE loss 对全部像面行列求和，最终是一个实数标量。

---

## 3. PE loss 对空中像的梯度

虽然 $L_{\mathrm{PE}}$ 是标量，但它对二维空中像 $I$ 的梯度是一张 $N\times N$ 图：

$$
G_I(i,j)
\equiv
\frac{\partial L_{\mathrm{PE}}}{\partial I(i,j)}.
$$

首先：

$$
\frac{\partial L_{\mathrm{PE}}}{\partial W(i,j)}
=
2[W(i,j)-T(i,j)].
$$

sigmoid 导数为：

$$
\frac{\partial W(i,j)}{\partial I(i,j)}
=
\alpha W(i,j)[1-W(i,j)].
$$

因此：

$$
\boxed{
G_I
=
\frac{\partial L_{\mathrm{PE}}}{\partial I}
=
2\alpha
(W-T)\odot W\odot(1-W).
}
$$

定义代码中使用的未缩放中间量：

$$
\boxed{
R
=
(W-T)\odot W\odot(1-W).
}
$$

于是：

$$
G_I=2\alpha R.
$$

当前 Gradient::pe_gradient() 中名为 dLdI 的变量实际保存 $R$，而不是完整的
$G_I$。缺少的 $2\alpha$ 与后续 $|E|^2$ 导数产生的另一个 $2$ 合并为
output_scale = 4 * alpha。

---

## 4. Mask 扰动引起的复振幅变化

令实数 mask 发生任意微小扰动：

$$
M\longrightarrow M+\delta M,
\qquad
\delta M\in\mathbb R^{N\times N}.
$$

由于 FFT、与固定核逐元素相乘、IFFT 都是线性运算：

$$
A_k[M+\delta M]
=
A_k[M]+A_k[\delta M].
$$

因此：

$$
\boxed{
\delta E_k
=
A_k[\delta M]
=
\mathcal F^{-1}
\left[
\mathcal F(\delta M)\odot K_k
\right].
}
$$

这里 $\mathcal F(\delta M)$ 与 $K_k$ 都是 $N\times N$，二者仍然逐元素相乘。

---

## 5. 复振幅变化引起的空中像变化

第 $k$ 个核的强度为：

$$
|E_k|^2=E_k\odot\overline{E_k}.
$$

对其求微分：

$$
\delta |E_k|^2
=
\delta E_k\odot\overline{E_k}
+
E_k\odot\overline{\delta E_k}.
$$

两项互为共轭，因此：

$$
\boxed{
\delta |E_k|^2
=
2\operatorname{Re}
\left(
\overline{E_k}\odot\delta E_k
\right).
}
$$

于是：

$$
\boxed{
\delta I
=
\frac{2}{S}
\sum_kq_k
\operatorname{Re}
\left(
\overline{E_k}\odot\delta E_k
\right).
}
$$

$\delta I$ 是一张 $N\times N$ 实数图。这里仍然只对核编号 $k$ 求和。

---

## 6. 从空中像变化得到 loss 变化

二维实数变量的微分满足：

$$
\delta L_{\mathrm{PE}}
=
\sum_{i,j}
G_I(i,j)\delta I(i,j).
$$

代入 $\delta I$：

$$
\delta L_{\mathrm{PE}}
=
\frac{2}{S}
\sum_kq_k
\sum_{i,j}
G_I(i,j)
\operatorname{Re}
\left[
\overline{E_k(i,j)}\delta E_k(i,j)
\right].
$$

由于 $G_I$ 是实数，可以移入实部：

$$
\delta L_{\mathrm{PE}}
=
\frac{2}{S}
\sum_kq_k
\operatorname{Re}
\sum_{i,j}
\overline{
G_I(i,j)E_k(i,j)
}
\delta E_k(i,j).
$$

使用 Frobenius 内积：

$$
\boxed{
\delta L_{\mathrm{PE}}
=
\frac{2}{S}
\sum_kq_k
\operatorname{Re}
\left\langle
G_I\odot E_k,\,
\delta E_k
\right\rangle_F.
}
$$

这里：

$$
G_I\odot E_k
$$

是两张 $N\times N$ 图的逐元素乘法。它为第 $k$ 个电场的每个像面像素乘上该像素
对 PE loss 的敏感度。

再代入 $\delta E_k=A_k[\delta M]$：

$$
\boxed{
\delta L_{\mathrm{PE}}
=
\frac{2}{S}
\sum_kq_k
\operatorname{Re}
\left\langle
G_I\odot E_k,\,
A_k[\delta M]
\right\rangle_F.
}
$$

---

## 7. 二维光学算子的伴随

### 7.1 伴随的定义

$A_k^*$ 定义为满足：

$$
\boxed{
\langle Z,A_k[X]\rangle_F
=
\langle A_k^*[Z],X\rangle_F
}
$$

的算子。它把像面信号反向传播到 mask 平面。

### 7.2 由 Parseval 定理推导

已知：

$$
A_k[X]
=
\mathcal F^{-1}
\left[
\mathcal F(X)\odot K_k
\right].
$$

设 $P=N^2$。在当前 DFT 归一化约定下，Parseval 关系为：

$$
\langle X,Y\rangle_F
=
\frac{1}{P}
\langle\mathcal F(X),\mathcal F(Y)\rangle_F.
$$

因此：

$$
\begin{aligned}
\langle Z,A_k[X]\rangle_F
&=
\frac{1}{P}
\left\langle
\mathcal F(Z),
\mathcal F(X)\odot K_k
\right\rangle_F\\
&=
\frac{1}{P}
\sum_{i,j}
\overline{\mathcal F(Z)(i,j)}
\mathcal F(X)(i,j)
K_k(i,j).
\end{aligned}
$$

利用：

$$
\overline{
\mathcal F(Z)(i,j)\overline{K_k(i,j)}
}
=
\overline{\mathcal F(Z)(i,j)}K_k(i,j),
$$

得到：

$$
\begin{aligned}
\langle Z,A_k[X]\rangle_F
&=
\frac{1}{P}
\left\langle
\mathcal F(Z)\odot\overline{K_k},
\mathcal F(X)
\right\rangle_F\\
&=
\left\langle
\mathcal F^{-1}
\left[
\mathcal F(Z)\odot\overline{K_k}
\right],
X
\right\rangle_F.
\end{aligned}
$$

与伴随定义比较：

$$
\boxed{
A_k^*[Z]
=
\mathcal F^{-1}
\left[
\mathcal F(Z)\odot\overline{K_k}
\right].
}
$$

这个式子只包含：

1. 对 $Z$ 做二维 FFT。
2. 与 $\overline{K_k}$ 逐元素相乘。
3. 做二维 IFFT。

没有两个 $N\times N$ 数组的普通矩阵乘法。

### 7.3 为什么只对代码中的核逐元素共轭

对于每个频率位置：

$$
\widehat E_k(i,j)
=
\widehat M(i,j)K_k(i,j).
$$

该位置只依赖同一个位置的 $\widehat M(i,j)$，不同频率位置不混合。因此频域乘核对应
一个对角线性操作，其伴随在每个频率位置上独立地乘：

$$
\overline{K_k(i,j)}.
$$

对代码存储的 $N\times N$ 频域核调用二维 transpose() 会交换 $f_x$ 和 $f_y$，
不是该逐元素乘法的伴随。

---

## 8. 用伴随算子读出 Mask 梯度

第 6 节得到：

$$
\delta L_{\mathrm{PE}}
=
\frac{2}{S}
\sum_kq_k
\operatorname{Re}
\left\langle
G_I\odot E_k,\,
A_k[\delta M]
\right\rangle_F.
$$

使用伴随定义：

$$
\left\langle
G_I\odot E_k,\,
A_k[\delta M]
\right\rangle_F
=
\left\langle
A_k^*[G_I\odot E_k],\,
\delta M
\right\rangle_F.
$$

所以：

$$
\delta L_{\mathrm{PE}}
=
\frac{2}{S}
\sum_kq_k
\operatorname{Re}
\left\langle
A_k^*[G_I\odot E_k],\,
\delta M
\right\rangle_F.
$$

因为 mask 是实数变量，允许的 $\delta M$ 也是实数。若 $B=B_R+iB_I$，则：

$$
\operatorname{Re}\langle B,\delta M\rangle_F
=
\sum_{u,v}B_R(u,v)\delta M(u,v).
$$

而实数矩阵梯度的定义是：

$$
\delta L_{\mathrm{PE}}
=
\sum_{u,v}
\frac{\partial L_{\mathrm{PE}}}{\partial M(u,v)}
\delta M(u,v).
$$

比较每个 $\delta M(u,v)$ 的系数，得到：

$$
\boxed{
\nabla_M L_{\mathrm{PE}}
=
\frac{2}{S}
\sum_kq_k
\operatorname{Re}
\left\{
A_k^*
\left[
G_I\odot E_k
\right]
\right\}.
}
$$

代入伴随算子：

$$
\boxed{
\nabla_M L_{\mathrm{PE}}
=
\frac{2}{S}
\sum_kq_k
\operatorname{Re}
\left\{
\mathcal F^{-1}
\left[
\mathcal F
\left(
G_I\odot E_k
\right)
\odot\overline{K_k}
\right]
\right\}.
}
$$

再代入：

$$
G_I=2\alpha R,
\qquad
R=(W-T)\odot W\odot(1-W),
$$

得到当前代码实现的最终公式：

$$
\boxed{
\nabla_M L_{\mathrm{PE}}
=
\frac{4\alpha}{S}
\sum_kq_k
\operatorname{Re}
\left\{
\mathcal F^{-1}
\left[
\mathcal F
\left(
R\odot E_k
\right)
\odot\overline{K_k}
\right]
\right\}.
}
$$

这是一个 $N\times N$ 实数矩阵。具体到当前输入：

$$
\boxed{
\nabla_M L_{\mathrm{PE}}
\in
\mathbb R^{257\times257}.
}
$$

每个元素：

$$
\frac{\partial L_{\mathrm{PE}}}{\partial M(u,v)}
$$

表示只增大 mask 像素 $M(u,v)$ 时，PE loss 的一阶变化率。

---

## 9. 完全展开的二维下标公式

为了同时看清“图像行列求和”和“核求和”，设 $h_k$ 是 $K_k$ 对应的空间域复数核。
前向场可以写成：

$$
E_k(i,j)
=
\sum_{u,v}
h_k(i-u,j-v)M(u,v).
$$

因此：

$$
\frac{\partial E_k(i,j)}{\partial M(u,v)}
=
h_k(i-u,j-v).
$$

空中像对单个 mask 像素的导数为：

$$
\frac{\partial I(i,j)}{\partial M(u,v)}
=
\frac{2}{S}
\sum_kq_k
\operatorname{Re}
\left[
\overline{E_k(i,j)}
h_k(i-u,j-v)
\right].
$$

再使用链式法则：

$$
\frac{\partial L_{\mathrm{PE}}}{\partial M(u,v)}
=
\sum_{i,j}
\frac{\partial L_{\mathrm{PE}}}{\partial I(i,j)}
\frac{\partial I(i,j)}{\partial M(u,v)}.
$$

得到：

$$
\boxed{
\frac{\partial L_{\mathrm{PE}}}{\partial M(u,v)}
=
\frac{2}{S}
\sum_kq_k
\sum_{i,j}
G_I(i,j)
\operatorname{Re}
\left[
\overline{E_k(i,j)}
h_k(i-u,j-v)
\right].
}
$$

使用 $G_I=2\alpha R$：

$$
\boxed{
\frac{\partial L_{\mathrm{PE}}}{\partial M(u,v)}
=
\frac{4\alpha}{S}
\sum_kq_k
\sum_{i,j}
R(i,j)
\operatorname{Re}
\left[
\overline{E_k(i,j)}
h_k(i-u,j-v)
\right].
}
$$

这个式子明确包含两种不同的求和：

- $\sum_k$：累加所有 Abbe 光源点或 SOCS 相干模态。
- $\sum_{i,j}$：累加所有像面像素对 mask 像素 $(u,v)$ 的贡献。

如果直接按此公式计算，需要四重空间循环，并在外层再对核编号 $k$ 循环。当前代码使用：

$$
\mathcal F^{-1}
\left[
\mathcal F(R\odot E_k)\odot\overline{K_k}
\right]
$$

等价地完成 $\sum_{i,j}$ 对所有 $(u,v)$ 的相关运算，将复杂度从直接空间计算降低为
每个核若干次二维 FFT。

等价的直接循环伪代码是：

~~~cpp
for (int u = 0; u < N; ++u) {
    for (int v = 0; v < N; ++v) {
        gradient(u, v) = 0.0;
        for (int k = 0; k < kernel_count; ++k) {
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    gradient(u, v) += 4.0 * alpha / source_weight_sum
                        * q[k] * R(i, j)
                        * std::real(std::conj(E[k](i, j)) * h[k](i - u, j - v));
                }
            }
        }
    }
}
~~~

实际边界和索引采用 FFT 对应的周期坐标约定；上述伪代码用于展示求和维度。

---

## 10. 与当前 C++ 代码逐项对应

### 10.1 构造未缩放的像面敏感度 $R$

数学：

$$
R=(W-T)\odot W\odot(1-W).
$$

代码：

~~~cpp
Eigen::MatrixXd dLdI =
    (wafer_image - target_image).array() *
     wafer_image.array() * (1.0 - wafer_image.array());
~~~

### 10.2 构造第 $k$ 个像面反向信号

数学：

$$
Z_k=R\odot E_k.
$$

代码：

~~~cpp
local_buf = electric_field[k].array() * dLdI.array();
~~~

这是逐元素乘法。$R$ 和 $E_k$ 都是 $257\times257$，相同位置的元素相乘。

### 10.3 应用伴随光学算子

数学：

$$
A_k^*[Z_k]
=
\mathcal F^{-1}
\left[
\mathcal F(Z_k)\odot\overline{K_k}
\right].
$$

代码：

~~~cpp
FFT::ifftshift_inplace(local_buf);
FFT::to_fftw(local_buf, local_in);
fftw_execute_dft(_plan_fwd, local_in, local_out);
FFT::from_fftw(local_out, local_buf);

local_buf.array() *= optical_kernels_frequency[k].conjugate().array();

FFT::to_fftw(local_buf, local_in);
fftw_execute_dft(_plan_inv, local_in, local_out);
FFT::from_fftw(local_out, local_buf);
FFT::fftshift_inplace(local_buf);
~~~

### 10.4 取实部并乘系数

数学：

$$
\displaystyle
4\alpha q_k
\operatorname{Re}\{A_k^*[R\odot E_k]\}.
$$

代码：

~~~cpp
kernel_gradients[k] =
    (output_scale * weight * invN * local_buf.real().array()).matrix();
~~~

其中：

~~~cpp
output_scale = 4.0 * _alpha;
weight = socs_eigenvalues[k];  // SOCS
invN = 1.0 / (N * N);          // 实现数学定义中 IFFT 的归一化
~~~

### 10.5 对核求和并除以光源权重总和

数学：

$$
\nabla_ML_{\mathrm{PE}}
=
\frac{1}{S}
\sum_k
\text{kernel gradient}_k.
$$

代码：

~~~cpp
for (int k = 0; k < K; ++k) {
    gradient += kernel_gradients[k];
}

gradient /= source_weight_sum;
~~~

---

## 11. 计算流程和尺寸

对每个 $k$：

$$
\begin{array}{rcl}
R\odot E_k
&:&
257\times257\\
\mathcal F(R\odot E_k)
&:&
257\times257\\
\mathcal F(R\odot E_k)\odot\overline{K_k}
&:&
257\times257\\
\mathcal F^{-1}[\cdots]
&:&
257\times257\\
\operatorname{Re}\{\mathcal F^{-1}[\cdots]\}
&:&
257\times257
\end{array}
$$

最后对全部 $k$ 的 $257\times257$ 结果逐元素累加，仍得到：

$$
\boxed{
\nabla_ML_{\mathrm{PE}}
\in\mathbb R^{257\times257}.
}
$$

---

## 12. 如果 CTM 优化的是参数 $\theta$

当前 CTM 可通过：

$$
M=\frac{1+\cos\theta}{2}
$$

保证 mask 值处于 $[0,1]$。此时：

$$
\frac{\partial M}{\partial\theta}
=
-\frac{1}{2}\sin\theta.
$$

因此：

$$
\boxed{
\nabla_\theta L_{\mathrm{PE}}
=
-\frac{1}{2}
\sin\theta
\odot
\nabla_ML_{\mathrm{PE}}.
}
$$

这里仍然是 $N\times N$ 数组的逐元素乘法。

---

## 13. 有限差分验证

选择任意实数方向矩阵：

$$
D\in\mathbb R^{N\times N},
\qquad
\|D\|_F=1.
$$

解析方向导数：

$$
d_{\mathrm{analytic}}
=
\sum_{u,v}
\nabla_ML_{\mathrm{PE}}(u,v)D(u,v).
$$

中心有限差分：

$$
d_{\mathrm{numeric}}
=
\frac{
L_{\mathrm{PE}}(M+\varepsilon D)
-
L_{\mathrm{PE}}(M-\varepsilon D)
}{
2\varepsilon
}.
$$

若推导、共轭、shift 和 FFT 归一化正确，应满足：

$$
d_{\mathrm{analytic}}
\approx
d_{\mathrm{numeric}}.
$$

当前真实输入诊断进一步使用 $M=(1+\cos\theta)/2$，在 $\theta$ 参数空间进行方向有限差分。
在 $\varepsilon=10^{-3}$ 时，解析方向导数和中心有限差分均约为：

$$
-41.1178.
$$

这验证了 $\nabla_ML_{\mathrm{PE}}$ 以及
$\nabla_\theta L_{\mathrm{PE}}=-\tfrac12\sin\theta\odot\nabla_ML_{\mathrm{PE}}$
组成的完整反向传播链。

---

## 14. 最终结论

当前二维 PE 对 mask 的梯度为：

$$
\boxed{
\nabla_M L_{\mathrm{PE}}
=
\frac{4\alpha}{S}
\sum_kq_k
\operatorname{Re}
\left\{
\mathcal F^{-1}
\left[
\mathcal F
\left(
\left[(W-T)\odot W\odot(1-W)\right]
\odot E_k
\right)
\odot\overline{K_k}
\right]
\right\}.
}
$$

其中：

- $L_{\mathrm{PE}}$ 是对所有像面行列求和得到的标量。
- $\partial L_{\mathrm{PE}}/\partial I$ 是一张 $N\times N$ 像面梯度图。
- $R\odot E_k$ 是逐元素乘法。
- 与 $\overline{K_k}$ 的频域乘法也是逐元素乘法。
- $\sum_k$ 对光学核或 SOCS 模态求和。
- 像面行列 $(i,j)$ 对每个 mask 像素 $(u,v)$ 的求和由伴随相关运算
  $\mathcal F^{-1}[\mathcal F(\cdot)\odot\overline{K_k}]$ 一次性完成。
- 最终结果是与输入 mask 同尺寸的实数矩阵。
