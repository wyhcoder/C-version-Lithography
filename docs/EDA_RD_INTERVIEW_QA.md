# 计算光刻 / EDA R&D 面试学习题库

> 适用项目：`Litho_cpp`  
> 目标岗位：计算光刻、OPC、Patterning DTCO、EDA 数值算法、EDA C++ / HPC 研发  
> 使用方法：先遮住“参考回答”口述 1～2 分钟，再打开源码核对；不要逐字背诵。

## 0. 岗位要求与项目对应关系

目前公开岗位中，全芯智造的 OPC 研发方向要求制程数值模拟、参数优化、Linux、Python/C/C++、机器学习和半导体制造知识；Patterning DTCO 方向还涉及 OPC 仿真、图像处理、图论和版图数据库。华芯程的软件工程师岗位强调大型 EDA 工业软件和面向对象语言开发。

- 全芯智造 OPC 研发：<https://amedac.zhiye.com/zpdetail/560894155>
- 全芯智造 Patterning DTCO：<https://amedac.zhiye.com/zpdetail/560895625>
- 华芯程招聘岗位：<https://www.seida.tech/zpgw>

本项目已经覆盖 Abbe/SOCS 成像、FFT、SVD、CTM、LSM、MEEF、EPE/WEPE、SRAF、PV Band、OpenMP 和 YAML 配置；主要短板是系统化测试、Linux 工程化、版图数据库和真实工艺标定。

---

## 1. 项目介绍与架构

### Q1：请用一分钟介绍这个项目。

**参考回答：**

这是一个使用 C++20 实现的计算光刻和 OPC 优化原型。成像模块根据光源、光瞳、掩模和光刻胶参数计算 aerial image 与 wafer image，同时支持逐光源点的 Abbe 成像和基于 SVD 降阶的 SOCS 成像。优化模块包含 CTM、Level Set 和基于控制点的 MEEF 优化，并使用 EPE、WEPE、PE 和 PV Band 等指标评价结果。工程上使用 Eigen、FFTW、OpenCV、OpenMP、yaml-cpp 和 CMake，并保存配置快照、迭代历史、计时和可视化结果。

### Q2：项目的数据流是什么？

**参考回答：**

```text
config.yaml
  → Grid / Source / Pupil / Target Mask
  → LithoPrepare 生成 ImagingCache
  → Imaging 计算 aerial image 和 wafer image
  → Loss 计算 EPE / WEPE / PE
  → CTM、LSM 或 MEEF 更新掩模
  → 保存迭代结果、最优结果、日志和可视化
```

其中 `LithoPrepare` 负责把不随迭代变化的光学量预计算并缓存，避免每轮优化重复构造光瞳和相干核。

### Q3：为什么要把 `LithoPrepare` 和 `Imaging` 分开？

**参考回答：**

光源采样、频移光瞳、PSF 和 SOCS 分解代价较大，但在固定光学条件下不会随 mask 变化。把它们放入 `ImagingCache` 后，优化迭代只需要对新 mask 做 FFT、核乘法、逆 FFT 和强度累加。这样既降低重复计算，也使缓存可以在 CTM、LSM、MEEF 和 PV Band 模块间复用。

### Q4：为什么使用 YAML？

**参考回答：**

EDA 数值实验参数多，硬编码会降低可复现性。YAML 将光学参数、光刻胶参数、MEEF 参数、输入路径和输出路径统一放到配置文件。demo 会把实际配置复制到结果目录，并保存终端日志，因此每次实验都能追溯参数。

### Q5：这个项目离工业 EDA 产品还有哪些距离？

**参考回答：**

目前是研究型原型，输入主要是 BMP/TXT 矩阵，规模也较小。工业产品还需要 GDSII/OASIS 层次化版图、分块和边界拼接、大规模并行、严格工艺标定、DRC/MRC 约束、稳定 API、异常恢复、完整回归测试、跨平台部署以及更成熟的日志和版本管理。

---

## 2. 光刻成像基础

### Q6：Abbe 成像的核心思想是什么？

**参考回答：**

部分相干光源可以离散成多个光源点。对每个光源点，掩模频谱与对应的频移光瞳相乘，逆 FFT 得到复电场，再计算模平方。不同光源点彼此非相干，因此最后按光源权重累加强度：

```text
I(x,y) = Σ_s w_s | F⁻¹{ M(f,g) H_s(f,g) } |²
```

项目的 `Imaging::compute` 会先复用一次 mask FFT，再对各个光学核执行逆 FFT 和强度累加。

### Q7：为什么电场相加和强度相加不能混淆？

**参考回答：**

同一相干模式内部应先叠加复振幅，再取模平方；彼此非相干的光源点或相干模式之间应累加强度。如果直接把不同非相干源的复电场相加，会引入不存在的交叉干涉项。

### Q8：SOCS 是什么，为什么比 Abbe 快？

**参考回答：**

SOCS 将部分相干成像算子分解成少量相干核：

```text
I(x,y) ≈ Σ_k λ_k | F⁻¹{ M(f,g) K_k(f,g) } |²
```

项目把加权频移光瞳组成矩阵 `A`，执行薄 SVD，取前 `K` 个左奇异向量作为核，权重为 `σ_k²`。当有效核数 `K` 远小于光源点数 `Ns` 时，每次成像所需的逆 FFT 数量从 `Ns` 降到 `K`。

### Q9：SOCS 的速度和精度如何权衡？

**参考回答：**

增加模态数会提高奇异值能量覆盖率并减小与 Abbe 的误差，但也会增加每次成像的 FFT 数量、内存和运行时间。不能只看奇异值能量，还应在代表性版图上画出 `K—运行时间—Aerial RMS—EPE` 曲线，据此选取满足精度要求的最小 `K`。

### Q10：项目里 `K == source point count` 判断 Abbe/SOCS 是否稳健？

**参考回答：**

不完全稳健。当前实现用“核数是否等于光源点数”推断模式，如果 SOCS 恰好保留全部模态，也可能被误判成 Abbe。更可靠的设计是在 `ImagingCache` 中保存显式枚举，例如 `ImagingMode::Abbe` 和 `ImagingMode::SOCS`，不要从容器长度推断语义。

### Q11：`fftshift` 和 `ifftshift` 为什么重要？

**参考回答：**

数学推导通常把零频放在频谱中心，而 FFTW 的数组约定把零频放在角点。mask、光瞳和核相乘前必须使用一致的频率布局，否则会产生相位错位和空间平移。偶数与奇数尺寸下 shift 的偏移量也不同，因此需要专门测试。

### Q12：项目中的光刻胶模型是什么？

**参考回答：**

项目使用 sigmoid 将 aerial image 转成连续 wafer image：

```text
W = 1 / (1 + exp(-alpha * (I - threshold)))
```

`threshold` 控制显影阈值，`alpha` 控制转变陡峭程度。较大的 `alpha` 更接近硬阈值，但会带来梯度集中、指数溢出和数值敏感性，需要使用稳定 sigmoid 或限制指数范围。

### Q13：NA、波长和 sigma 分别有什么作用？

**参考回答：**

- 波长越短，理论分辨率越高。
- NA 越大，可通过的空间频率范围越宽，但景深通常减小。
- `sigma_in/sigma_out` 描述归一化光源内外半径，改变照明相干性与方向性。
- 它们共同决定光瞳截止频率、成像对比度、分辨率和工艺窗口。

### Q14：Zernike 多项式在项目中的作用是什么？

**参考回答：**

Zernike 多项式用于在单位圆光瞳上描述像差，例如离焦、像散和彗差。项目把 YAML 中的像差系数转成光瞳相位；PV Band 模块还会根据 defocus 更新相应的光瞳或缓存。

### Q15：为什么不能只验证 wafer 二值图？

**参考回答：**

阈值化会掩盖 aerial image 中的连续误差。两个 aerial image 可能存在明显差异，但恰好落在阈值同一侧，得到相同二值结果。验证应同时比较 aerial RMS/max error、wafer mismatch、轮廓 EPE 和工艺窗口。

---

## 3. OPC、EPE 与 MEEF

### Q16：什么是 OPC？

**参考回答：**

OPC 是在 mask 上主动加入边缘偏置、角部修正或 SRAF，使经过衍射和光刻胶非线性后形成的 wafer 轮廓更接近目标版图。它本质上是受制造约束的逆问题。

### Q17：EPE 是什么？

**参考回答：**

EPE 是从目标边缘上的 evaluation point 沿局部法向测量目标轮廓与印刷轮廓之间的有符号距离。它比全图像素误差更接近版图签核关注的边缘偏差，但结果依赖 EP 的采样位置、法向定义和轮廓提取方法。

### Q18：EPE、WEPE 和 PE 有什么区别？

**参考回答：**

- EPE：所有 EP 的边缘误差。
- WEPE：对拐角、线端、中点等关键 EP 加权，突出关键弱点。
- PE：wafer 图案与 target 的像素差异，反映全局区域误差。

优化时应说明目标函数是什么，不能把三者混为同一个指标。

### Q19：MEEF 矩阵是什么？

**参考回答：**

MEEF 矩阵是 EPE 对 mask 控制点位移的局部灵敏度矩阵。项目将 x/y 方向分开：

```text
Mx(i,j) = ∂EPE_i / ∂x_j
My(i,j) = ∂EPE_i / ∂y_j
```

矩阵行对应 EP，列对应主图形控制点。它把“移动哪个控制点会如何影响各 EP”线性化。

### Q20：项目如何构建 MEEF 矩阵？

**参考回答：**

每个控制点执行 `x+、x-、y+、y-` 四次扰动，重新渲染 mask、运行光刻成像并计算 EPE，最后用中心差分：

```text
∂E/∂x ≈ (E(x+δ) - E(x-δ)) / (2δ)
```

因此构建一次矩阵大约需要 `4 × 控制点数` 次成像，是优化中的主要耗时部分。

### Q21：为什么用中心差分而不是前向差分？

**参考回答：**

在函数足够光滑时，前向差分截断误差是一阶 `O(δ)`，中心差分是二阶 `O(δ²)`，精度通常更高；代价是每个方向需要正负两次仿真。

### Q22：差分步长 `delta` 如何选择？

**参考回答：**

太大会破坏局部线性假设，太小会受到浮点误差、栅格量化、MSAA 渲染误差和轮廓跳变影响。合理方法是做 step-size study：尝试多个 `delta`，观察 MEEF 列方向和范数是否稳定，并结合 mask 最小可表达位移选择。

### Q23：为什么 MEEF 求解需要 SVD 或正则化？

**参考回答：**

不同控制点可能对 EP 产生相似影响，MEEF 列会相关；有些方向又几乎不可观测，所以矩阵可能病态或秩亏。直接求逆会放大噪声。SVD 可以识别小奇异值方向，截断 SVD 或 Tikhonov 正则化能在残差和更新幅度之间折中。

### Q24：正则化参数过大或过小会怎样？

**参考回答：**

- 太小：追随 MEEF 和 EPE 中的噪声，控制点移动过大或振荡。
- 太大：更新过度平滑，收敛慢，难以修正局部误差。
- 可通过 L-curve、GCV、经验扫描或 trust-region 思路选择。

### Q25：`optimize_wepe_only` 做了什么？

**参考回答：**

项目先按 `weight_meef` 对 MEEF 的 EP 行加权；启用 `optimize_wepe_only` 后，再乘 `weight_epe`，非关键 EP 对应的行被置零，因此求解只响应 WEPE 关键点。这样可以集中优化弱点，但可能牺牲未选中轮廓的整体质量。

### Q26：为什么 SRAF 在控制点优化中保持固定？

**参考回答：**

当前 MEEF 只描述主图形控制点对 EPE 的灵敏度，因此每次扰动只重新渲染主图形，再叠加固定 SRAF。这样减少变量规模并保证矩阵含义一致。进一步扩展可以加入 SRAF 位置、宽度和长度变量，但必须增加 MRC 和主图形 keep-out 约束。

### Q27：项目如何判断提前收敛？

**参考回答：**

计算每轮控制点位移的最大模长 `|d|max`。当它连续 `patience` 轮小于 `step_tol` 时停止。连续多轮判定可以避免单轮偶然小步长导致误停。但还应同时观察目标函数改善量和约束违例。

### Q28：为什么“EPE 下降”不一定等于 OPC 更好？

**参考回答：**

单一 nominal 条件下 EPE 下降，可能伴随 mask 复杂度增加、局部最小间距违规、MEEF 条件变差，或者在 focus/dose 变化时性能恶化。因此还应检查 PV Band、工艺窗口、MRC、mask 可制造性和 hotspot。

### Q29：CTM、LSM 和 MEEF 方法有什么区别？

**参考回答：**

- CTM：直接优化连续 mask 像素或参数矩阵，变量多，梯度计算规则但可制造性较弱。
- LSM：用符号距离函数演化轮廓，适合拓扑变化和形状优化。
- MEEF：用少量控制点参数化边界，以 EPE 灵敏度建立局部线性模型，可解释性更强、变量更少。

---

## 4. Process Window 与 PV Band

### Q30：什么是 PV Band？

**参考回答：**

在多个 focus/dose 工艺条件下分别得到 wafer 轮廓，取每个位置的最大打印结果与最小打印结果之差，形成工艺变化带。PV Band 越宽，说明图形对工艺扰动越敏感。

### Q31：为什么 nominal EPE 和 PV Band 要一起看？

**参考回答：**

Nominal EPE 衡量标称条件准确度，PV Band 衡量鲁棒性。一个 mask 可能在 nominal 条件下非常准确，但轻微离焦或剂量变化就严重失真。工业 OPC 要在准确度、鲁棒性和 mask 复杂度之间平衡。

### Q32：PV Band 梯度为什么更难计算？

**参考回答：**

最大/最小操作在工艺角切换位置不可微，而且损失依赖多个 focus/dose 条件。实际可采用当前极值工艺角的次梯度、平滑 max/min（如 log-sum-exp）或有限差分，并验证梯度与数值差分的一致性。

---

## 5. 数值计算与线性代数

### Q33：为什么 SOCS 使用 `σ²` 作为权重？

**参考回答：**

SVD 给出算子振幅方向的奇异值 `σ`。成像最终累加的是复场模平方，因此对应相干模式的强度权重为 `λ = σ²`。

### Q34：如何判断一个矩阵病态？

**参考回答：**

查看最大奇异值与最小有效奇异值之比，即条件数；也可以观察奇异值谱是否快速衰减。条件数很大时，输入中的微小扰动会引起解的巨大变化。

### Q35：为什么浮点求和顺序会影响并行结果？

**参考回答：**

浮点加法不满足严格结合律，`(a+b)+c` 与 `a+(b+c)` 可能不同。项目先并行计算各核的复电场，再按固定核顺序串行累加强度，使 1 线程和多线程采用相同求和顺序，提高可复现性。

### Q36：如何验证解析梯度或反向传播是否正确？

**参考回答：**

使用 gradient check：随机选择少量变量，以中心差分计算数值梯度，与解析梯度比较相对误差。应扫描不同步长，避开 sigmoid 饱和和轮廓不连续区域。

### Q37：为什么要区分绝对误差和相对误差？

**参考回答：**

接近零的量使用相对误差会被无限放大，而大幅值量只看绝对误差又缺乏尺度意义。测试中通常使用：

```text
|actual - expected| <= abs_tol + rel_tol * |expected|
```

### Q38：如何验证 FFT 实现？

**参考回答：**

- 与 NumPy FFT 或直接 DFT 的小矩阵结果对比。
- 验证 `ifft(fft(x)) ≈ x`。
- 验证 Parseval 定理。
- 测试奇数/偶数尺寸下的 shift 往返。
- 测试 delta、常数、单频正弦和随机复数输入。

---

## 6. C++ 与资源管理

### Q39：项目中哪些资源需要 RAII？

**参考回答：**

FFTW 的 buffer 和 plan 是典型手动资源。当前 `Imaging` 和 `LithoPrepare` 在析构函数中调用 `fftw_free` 和 `fftw_destroy_plan`，基本满足 RAII。但还应处理构造中途失败，并使用自定义 deleter 的 `unique_ptr` 或包装类减少裸指针。

### Q40：`Imaging` 当前存在什么 Rule of Five 风险？

**参考回答：**

`Imaging` 拥有 FFTW 裸指针和 plan，并自定义了析构函数，但没有显式删除拷贝构造和拷贝赋值。隐式浅拷贝会让两个对象释放同一资源，产生 double free。应至少：

```cpp
Imaging(const Imaging&) = delete;
Imaging& operator=(const Imaging&) = delete;
```

如果需要移动，还应实现安全的移动构造和移动赋值。

### Q41：`Imaging` 保存 `const ImagingCache&` 有什么生命周期要求？

**参考回答：**

引用不拥有对象，因此 `ImagingCache` 必须比 `Imaging` 活得更久。项目中通常由 `LithoPrepare` 先构造 cache，再构造 `Imaging`。更安全的接口可以用明确的所有权约定、`reference_wrapper`，或者在确实需要独立生命周期时使用 `shared_ptr<const ImagingCache>`。

### Q42：为什么函数参数优先使用 `const&`？

**参考回答：**

Eigen 矩阵和缓存可能很大，按值传递会产生复制。只读输入用 `const T&` 可以避免复制并表达不修改语义；需要转移所有权时再按值接收并 `std::move`。

### Q43：什么时候 `std::move` 不一定提高性能？

**参考回答：**

对 `const` 对象调用 `std::move` 通常仍会复制；对可触发返回值优化的局部返回值手动 move 还可能阻碍 NRVO。移动后对象也只能处于“有效但未指定”状态，不能假定保留原内容。

### Q44：为什么 EDA 面试仍会问数据结构和算法？

**参考回答：**

大型 EDA 工具需要处理海量图元、网表和空间查询。常见结构包括哈希表、堆、图、并查集、KD-tree、R-tree、扫描线和层次化数据库。即使岗位是计算光刻，也需要轮廓排序、连通域、空间分块和热点查询。

### Q45：这个项目最需要补充哪些计算几何内容？

**参考回答：**

目前主要处理栅格 mask。建议补充多边形布尔运算、点在线/多边形判断、线段相交、轮廓方向、offset、空间索引、GDSII/OASIS 层次展开及 tile halo。这样才能从研究矩阵走向真实版图。

---

## 7. OpenMP 与高性能计算

### Q46：项目的主要性能瓶颈在哪里？

**参考回答：**

MEEF 每个控制点需要四次扰动成像，每次成像又执行多个 SOCS 核的逆 FFT，所以复杂度近似为：

```text
O(iter × num_cp × 4 × K × N² log N)
```

此外还有参数曲线渲染、EPE 评价和大量结果 I/O，但 FFT 通常是主要计算热点。

### Q47：为什么 MEEF 外层并行时要关闭 Imaging 内层并行？

**参考回答：**

如果外层 `4 × num_cp` 个任务开 `T` 个线程，内层每次成像又开 `T` 个线程，可能产生近似 `T²` 个工作线程，导致过度订阅、上下文切换和缓存竞争。项目用 `omp_in_parallel()` 判断是否已在并行区，外层并行时让内层串行。

### Q48：FFTW 在线程中有哪些注意事项？

**参考回答：**

FFTW planner 通常不能无保护地并发创建 plan。项目在进入 MEEF 并行区前串行创建每个 worker 的 `Imaging`，执行阶段每个线程独占 plan 和 buffer。共享只读 plan 是否安全也必须符合 FFTW 的接口约定，输入输出 buffer 不能产生数据竞争。

### Q49：为什么 MEEF 任务使用 `schedule(dynamic, 1)`？

**参考回答：**

不同控制点的曲线渲染、轮廓和成像成本可能不完全一致。动态调度以单任务为粒度可以改善负载均衡；代价是调度开销更大。是否优于 static 应通过基准测试决定。

### Q50：如何正确报告 OpenMP 加速比？

**参考回答：**

- 使用 Release 构建并固定输入。
- 预热后重复多次，报告中位数和波动。
- 分别测试 1/2/4/8 线程。
- 计时中排除可视化和非必要文件 I/O。
- 同时报告 speedup、parallel efficiency、CPU 型号和问题规模。
- 检查并行结果与单线程结果误差。

### Q51：什么是 false sharing？

**参考回答：**

多个线程虽然写不同变量，但变量位于同一 cache line，缓存一致性会让该 cache line 在核心间反复失效。可以通过线程私有缓冲、padding、分块写入和减少共享计数器缓解。

### Q52：为什么不能只看总运行时间？

**参考回答：**

总时间无法定位瓶颈。应分别测量配置和初始化、SOCS 分解、单次成像、MEEF 矩阵构建、SVD 求解、mask 渲染、EPE 计算和磁盘 I/O，并使用 profiler 验证热点，而不是只靠猜测。

---

## 8. 测试、构建与工程化

### Q53：这个项目最应该增加哪些单元测试？

**参考回答：**

1. FFT/shift 往返和 NumPy 对照。
2. Source 权重归一化和 pupil 支撑域。
3. Abbe/SOCS 小案例 golden test。
4. EPE 符号、法向和已知轮廓距离。
5. MEEF 中心差分与手工小矩阵。
6. SVD 求解残差和病态矩阵。
7. 多线程与单线程一致性。
8. YAML 缺字段、非法路径和非法参数。

### Q54：为什么需要 golden test？

**参考回答：**

复杂数值管线很难仅靠局部单元测试证明正确。可以用 Python/NumPy、较慢的 Abbe 实现或已验证版本生成小规模参考结果，C++ 每次修改后在误差容限内比较，防止优化造成静默数值回归。

### Q55：应该使用哪些 Sanitizer？

**参考回答：**

- ASan：越界、use-after-free、double free。
- UBSan：未定义行为。
- TSan：数据竞争，但通常不能与 ASan 同时启用。
- LSan：内存泄漏。

特别应覆盖 FFTW 原始资源、OpenMP worker 和 Eigen 矩阵索引。

### Q56：CMake 面试需要掌握什么？

**参考回答：**

理解 target-based CMake、`PUBLIC/PRIVATE/INTERFACE`、include 传播、静态/动态链接、Debug/Release、编译选项、依赖查找、CTest、安装导出和跨平台条件。避免全局 `include_directories` 和硬编码本机绝对路径。

### Q57：为什么 EDA 工具通常重视 Linux？

**参考回答：**

芯片设计与制造工具大量运行在 Linux 服务器、工作站和计算集群上，需要 Shell、进程管理、权限、动态库、批处理、远程调试和性能工具。至少应熟练使用 `gdb`、`perf`、`cmake`、`ninja/make`、`grep/rg`、`awk`、`sed`、Shell 和 Git。

### Q58：如何设计持续集成？

**参考回答：**

在 Linux CI 中执行 Release/Debug 构建、单元测试、格式检查、ASan/UBSan，以及小规模数值回归。大规模 benchmark 不必每次提交都运行，可以定时执行并保存性能趋势。

---

## 8A. CMake 专项学习问答

这一节不是让你背 CMake 命令，而是让你能解释本项目如何从 `.cpp` 变成可执行程序，并能独立排查构建错误。

### CMake Q1：CMake、编译器、Make/Ninja 和链接器分别做什么？

**参考回答：**

CMake 是构建系统生成器，它读取 `CMakeLists.txt`，生成 Makefile 或 `build.ninja`；Make/Ninja 根据依赖关系调用编译器；编译器把每个 `.cpp` 编译成目标文件；链接器再把目标文件和 Eigen、FFTW、OpenCV、yaml-cpp、OpenMP 等依赖组合成库或可执行文件。

```text
CMakeLists.txt
    ↓ cmake 配置
Makefile / build.ninja
    ↓ make / ninja
.cpp → 编译器 → .o
    ↓ 链接器
liblitho_core.a + demo_MEEF_Optimizer_init
```

CMake 本身通常不直接完成 C++ 编译，它负责描述并生成构建规则。

### CMake Q2：配置、构建和运行为什么是三个不同阶段？

**参考回答：**

配置阶段查找编译器和依赖，并生成构建文件；构建阶段只编译有变化的源码并链接目标；运行阶段才真正执行程序。

```bash
# 1. 配置
cmake -S . -B build-release -G Ninja -DCMAKE_BUILD_TYPE=Release

# 2. 构建指定目标
cmake --build build-release \
  --target demo_MEEF_Optimizer_init --parallel

# 3. 从包含 config.yaml 和运行资源的构建目录运行
cd build-release
./demo_MEEF_Optimizer_init
```

修改 `.cpp` 后通常只需重新构建；修改 `CMakeLists.txt` 或依赖路径后应重新配置。

### CMake Q3：`-S`、`-B`、`-G` 和 `-D` 分别是什么？

**参考回答：**

- `-S .`：源码目录，包含顶层 `CMakeLists.txt`。
- `-B build-release`：构建目录，用来保存缓存、目标文件和可执行文件。
- `-G Ninja`：选择 Ninja 生成器。
- `-D变量=值`：给 CMake 缓存变量赋值，例如 `CMAKE_BUILD_TYPE=Release`。

将构建产物放在独立的 `build-release` 中叫 out-of-source build。这样源码目录更干净，也能同时保留 Debug、Release、Sanitizer 等不同配置。

### CMake Q4：本项目顶层 `CMakeLists.txt` 的主要结构是什么？

**参考回答：**

```cmake
cmake_minimum_required(VERSION 3.16)
project(LithoSim VERSION 1.0.0 LANGUAGES CXX)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(litho_core STATIC ...)
target_include_directories(litho_core PUBLIC ...)
target_link_libraries(litho_core PUBLIC ...)

add_executable(demo_MEEF_Optimizer_init ...)
target_link_libraries(demo_MEEF_Optimizer_init PRIVATE litho_core)
```

可以分为：声明项目、设置语言标准、查找依赖、创建核心库、给核心库配置头文件和依赖、创建 demo、让 demo 链接核心库、复制运行配置和资源。

### CMake Q5：为什么把公共代码做成 `litho_core` 静态库？

**参考回答：**

`litho_core` 集中编译成 `liblitho_core.a`，多个 demo 只需链接它，不必分别重复列出全部源码。这样模块边界更清晰，也能复用增量编译结果。

```cmake
add_library(litho_core STATIC
    source/litho_model/imaging.cpp
    source/optimizer/MEEF_Optimizer.cpp
    ...
)
```

静态库在链接时被合入可执行文件；动态库则在运行时加载，Linux 上通常是 `.so`。项目目前使用静态核心库，但 FFTW、OpenCV 等外部依赖仍可能是动态库。

### CMake Q6：`PUBLIC`、`PRIVATE`、`INTERFACE` 有什么区别？

**参考回答：**

它们描述“当前 target 是否需要”和“链接当前 target 的下游是否也需要”。

| 关键字 | 当前 target 使用 | 下游 target 继承 |
|---|---:|---:|
| `PRIVATE` | 是 | 否 |
| `PUBLIC` | 是 | 是 |
| `INTERFACE` | 否 | 是 |

例如：

```cmake
target_link_libraries(litho_core
    PUBLIC Eigen3::Eigen OpenMP::OpenMP_CXX
    PRIVATE some_internal_library
)
```

如果 `litho_core` 的公共头文件暴露了 Eigen 类型，那么使用这些头文件的 demo 也需要 Eigen include 路径，因此 Eigen 适合 `PUBLIC`。仅在某个 `.cpp` 内部使用的库可以设为 `PRIVATE`。

### CMake Q7：为什么推荐 target-based CMake？

**参考回答：**

现代 CMake 应围绕 target 设置属性：

```cmake
target_include_directories(litho_core PUBLIC ...)
target_link_libraries(litho_core PUBLIC ...)
target_compile_features(litho_core PUBLIC cxx_std_20)
```

相比全局 `include_directories()`、`link_libraries()` 和修改 `CMAKE_CXX_FLAGS`，target-based 写法能明确依赖属于哪个目标，减少 demo 之间互相污染，也更容易导出和复用。

### CMake Q8：`find_package`、`find_path` 和 `find_library` 有什么区别？

**参考回答：**

- `find_package(Eigen3 REQUIRED)`：查找一个完整依赖包，理想情况下得到 `Eigen3::Eigen` 这样的 imported target。
- `find_path(... fftw3.h)`：只查找头文件目录。
- `find_library(... fftw3)`：只查找库文件。

本项目对 Eigen、OpenCV、yaml-cpp 使用 `find_package`，对 FFTW 分别查找头文件和库。更成熟的工程可以提供或引入 FFTW 的 CMake package，使调用方直接链接 `FFTW3::fftw3`，避免手工维护 include 和 library 变量。

### CMake Q9：为什么链接 imported target 比手写库路径更好？

**参考回答：**

```cmake
target_link_libraries(litho_core PUBLIC Eigen3::Eigen)
```

`Eigen3::Eigen` 不只是一个文件名，它还能携带 include 目录、编译定义和传递依赖。相比硬编码：

```cmake
/opt/homebrew/lib/libxxx.dylib
```

imported target 更容易跨 macOS、Linux 和不同安装路径使用。硬编码 `/opt/homebrew` 只能作为本机查找提示，不应成为跨平台工程唯一可用的路径。

### CMake Q10：Debug、Release 和 RelWithDebInfo 有什么区别？

**参考回答：**

- Debug：优化低，包含调试信息，适合断点和变量检查。
- Release：优化高，适合性能测量，但调试体验较差。
- RelWithDebInfo：保留调试信息同时开启优化，适合定位 Release 性能问题。

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release
cmake -S . -B build-profile -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

比较 OpenMP 或 SOCS 性能时必须使用相同的 Release 类配置，不能拿 Debug 时间作结论。

### CMake Q11：为什么 CMake 缓存会导致“明明换了路径却仍然报旧路径”？

**参考回答：**

第一次配置后，CMake 会在构建目录生成 `CMakeCache.txt`，保存源码目录、编译器和依赖路径。如果复制了旧构建目录或移动了项目，缓存中的 `CMAKE_HOME_DIRECTORY` 仍可能指向旧位置。

先检查：

```bash
rg 'CMAKE_HOME_DIRECTORY|CMAKE_CXX_COMPILER' \
  build-release/CMakeCache.txt
```

最稳妥的处理方式是新建一个构建目录重新配置：

```bash
cmake -S . -B build-release-new -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
```

确认新目录构建成功后再处理旧构建目录，不要把 `CMakeCache.txt` 提交到 Git。

### CMake Q12：`compile_commands.json` 有什么作用？

**参考回答：**

```cmake
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
```

它让 CMake输出每个 `.cpp` 的真实编译命令，包括编译器、宏、语言标准和头文件目录。VS Code、clangd 和静态分析工具可以读取它，从而与实际构建保持一致。

如果项目能编译但 IDE 把 `Eigen/Dense` 标红，应检查：

```bash
rg 'eigen3|fopenmp' build-release/compile_commands.json
```

这属于编辑器配置问题，不一定是编译错误。

### CMake Q13：为什么推荐 `cmake --build`，而不是直接写 `make`？

**参考回答：**

```bash
cmake --build build-release --target demo_MEEF_Optimizer_init --parallel
```

这是生成器无关的命令：底层使用 Make 时会调用 Make，使用 Ninja 时会调用 Ninja。直接执行 `make` 只适用于 Makefile 生成器，而且通常要求当前目录就是构建目录。

### CMake Q14：`configure_file` 和 `file(COPY ...)` 在项目里做什么？

**参考回答：**

项目需要在运行时读取 `config.yaml` 和目标图形，因此配置阶段会把它们复制到构建目录：

```cmake
configure_file(
    ${CMAKE_SOURCE_DIR}/config.yaml
    ${CMAKE_BINARY_DIR}/config.yaml
    COPYONLY
)

file(COPY
    ${CMAKE_SOURCE_DIR}/assets/target_pattern
    DESTINATION ${CMAKE_BINARY_DIR}
)
```

这样从 `build-release` 运行 demo 时，相对路径能够找到资源。长期来看，更稳健的做法是由命令行或配置明确传入资源路径，并增加 install 规则。

### CMake Q15：如何把测试接入 CMake/CTest？

**参考回答：**

```cmake
include(CTest)

if(BUILD_TESTING)
    add_executable(test_fft tests/test_fft.cpp)
    target_link_libraries(test_fft PRIVATE litho_core)
    add_test(NAME fft_roundtrip COMMAND test_fft)
endif()
```

然后执行：

```bash
cmake --build build-release --parallel
ctest --test-dir build-release --output-on-failure
```

当前项目执行 `ctest` 会显示没有测试，说明下一步应把 FFT、SOCS、EPE 和 MEEF 数值回归接入 CTest，而不是只依赖 demo 能运行。

### CMake Q16：遇到构建错误应按什么顺序排查？

**参考回答：**

1. 配置错误：检查 `find_package`、依赖安装和 CMake 缓存。
2. 编译错误：看第一条 `error:`，检查头文件、类型和宏。
3. 链接错误：看到 `undefined reference` 时检查目标是否链接了实现所在的库。
4. 运行时动态库错误：Linux 使用 `ldd` 检查 `.so` 是否能找到。
5. IDE 标红但能编译：检查 `compile_commands.json` 和编辑器配置。

不要从最后一条连锁错误开始修，通常第一条错误最接近根因。

### CMake 专项实操

完成下面练习才算真正掌握：

1. 分别创建 `build-debug` 和 `build-release`，解释生成文件为何不能混用。
2. 只构建 `demo_MEEF_Optimizer_init`，观察哪些 `litho_core` 源文件被增量编译。
3. 修改一个 `.cpp` 后重新构建，解释为什么其他文件没有重新编译。
4. 在 `compile_commands.json` 中找到 `imaging.cpp` 的完整编译命令。
5. 添加一个最小 `test_fft`，通过 `ctest` 执行。
6. 在 Linux 上移除本机 `/opt/homebrew` 假设并完成配置。

---

## 8B. Linux 专项学习问答

对 EDA C++ R&D 来说，Linux 不只是会用几条命令，而是要能在远程服务器上构建、运行、监控、调试和分析数值程序。

### Linux Q1：Linux 文件系统中常见目录有什么作用？

**参考回答：**

- `/home/user`：用户代码和个人文件。
- `/usr/bin`：常用程序。
- `/usr/include`：系统头文件。
- `/usr/lib`、`/usr/lib64`：系统库文件。
- `/usr/local`：手工安装或本地软件。
- `/opt`：独立的第三方软件。
- `/tmp`：临时文件，不能当长期存储。
- `/proc`：内核暴露的进程和系统信息。

EDA 公司还经常把工具、PDK、license 和共享数据放在 NFS 挂载目录中，具体路径由公司环境决定。

### Linux Q2：绝对路径、相对路径和当前工作目录有什么关系？

**参考回答：**

绝对路径从 `/` 开始；相对路径从当前工作目录开始。程序读取 `config.yaml` 时，默认相对的是启动程序时的工作目录，不一定是可执行文件所在目录。

```bash
pwd                 # 当前工作目录
realpath config.yaml
ls -la
```

这正是 demo 从不同目录启动时可能找不到配置或图片的原因。工程程序应明确输入路径，或者在启动时打印最终解析后的绝对路径。

### Linux Q3：Linux 文件权限 `rwx` 如何理解？

**参考回答：**

权限分为 owner、group、others 三组，每组包含读、写、执行：

```text
-rwxr-x---
 ||| ||| |||
 user group other
```

常用命令：

```bash
ls -l demo_MEEF_Optimizer_init
chmod u+x script.sh
chmod 750 script.sh
```

目录的执行权限表示能否进入和访问目录中的条目。不要为了省事使用 `chmod -R 777`，它会造成不必要的安全风险。

### Linux Q4：进程和线程有什么区别？

**参考回答：**

进程拥有独立虚拟地址空间；同一进程中的线程共享代码、堆和全局数据，但有各自的栈和寄存器。项目运行一个 demo 时产生一个进程，OpenMP 在这个进程中创建多个线程。

```text
demo 进程
├── 主线程
├── OpenMP 线程 1
├── OpenMP 线程 2
└── OpenMP 线程 3
```

线程共享数据带来低通信开销，也带来 data race、false sharing 和线程安全问题。

### Linux Q5：如何查看程序进程和线程？

**参考回答：**

```bash
pgrep -af demo_MEEF_Optimizer_init
ps -ef | rg demo_MEEF
top -H -p <PID>
ps -L -p <PID> -o pid,tid,psr,pcpu,stat,comm
```

`top -H` 可以查看每个线程的 CPU 使用率。若设置了 8 个 OpenMP 线程但只有一个线程占用 CPU，应检查是否进入并行区域、任务数是否足够以及是否关闭了内层并行。

### Linux Q6：如何控制 OpenMP 线程数并确认设置生效？

**参考回答：**

```bash
OMP_NUM_THREADS=8 OMP_DISPLAY_ENV=TRUE \
  ./demo_MEEF_Optimizer_init
```

C++ 中可以调用：

```cpp
omp_get_max_threads();  // 可能使用的最大线程数
omp_get_num_threads();  // 当前并行区域实际线程数
omp_get_thread_num();   // 当前线程编号
```

`schedule(static)` 决定循环迭代如何分配，不决定线程数量。还应关注机器物理核心数、超线程和嵌套并行，线程越多不一定越快。

### Linux Q7：前台、后台、`nohup` 和终端会话有什么区别？

**参考回答：**

```bash
./demo                         # 前台运行
./demo &                       # 当前 shell 后台运行
jobs                           # 查看当前 shell 作业
fg %1                          # 切回前台
nohup ./demo > run.log 2>&1 &  # 退出终端后继续运行
```

长时间 EDA 任务更推荐使用 `tmux`、`screen` 或集群调度系统。单纯加 `&` 后关闭 SSH，会话中的任务可能收到 `SIGHUP` 而退出。

### Linux Q8：标准输入、标准输出和标准错误是什么？

**参考回答：**

每个进程默认有三个文件描述符：

```text
0：stdin   标准输入
1：stdout  标准输出
2：stderr  标准错误
```

保存所有终端输出：

```bash
./demo 2>&1 | tee run.log
```

只写文件：

```bash
./demo > run.log 2>&1
```

`tee` 可以一边在终端显示，一边写日志。程序也应返回正确退出码，Shell 中用 `echo $?` 查看上一条命令是否成功。

### Linux Q9：管道 `|` 的工作原理是什么？

**参考回答：**

管道把左侧命令的标准输出连接到右侧命令的标准输入：

```bash
ps -ef | rg demo_MEEF
rg 'error|warning' run.log | sort | uniq -c
```

它不是把两个命令“按顺序随便连接”，而是在进程之间传输字节流。复杂脚本应使用 `set -o pipefail`，否则管道左侧失败可能被最后一个成功命令掩盖。

### Linux Q10：如何安全终止进程？

**参考回答：**

```bash
kill -TERM <PID>  # 请求程序正常退出，默认选择
kill -INT <PID>   # 类似 Ctrl+C
kill -KILL <PID>  # 强制终止，无法清理资源
```

应先使用 `TERM`，给程序保存日志和清理临时文件的机会。只有程序无响应时才使用 `KILL`。`kill -9` 不能被捕获，也不会执行正常清理逻辑。

### Linux Q11：环境变量 `PATH` 和动态库搜索路径有什么区别？

**参考回答：**

`PATH` 决定 Shell 去哪里找可执行程序：

```bash
echo "$PATH"
command -v cmake
```

动态库搜索路径决定程序运行时去哪里找 `.so`。Linux 可以用：

```bash
ldd ./demo_MEEF_Optimizer_init
readelf -d ./demo_MEEF_Optimizer_init | rg 'RPATH|RUNPATH'
```

临时设置 `LD_LIBRARY_PATH` 可以解决实验环境问题，但正式部署更适合正确安装依赖或设置 RPATH，避免依赖用户 Shell 配置。

### Linux Q12：编译错误、链接错误和运行时动态库错误如何区分？

**参考回答：**

- 编译错误：`not declared`、`no matching function`、找不到头文件，发生在 `.cpp → .o`。
- 链接错误：`undefined reference`、`multiple definition`，发生在目标文件和库组合阶段。
- 运行错误：`error while loading shared libraries`，可执行文件已经生成，但启动时找不到 `.so`。

对应排查工具：

```bash
cmake --build build-release --verbose
nm -C liblitho_core.a | rg Imaging
ldd build-release/demo_MEEF_Optimizer_init
```

面试时要能根据错误发生阶段选择工具，而不是看到所有错误都重新安装 CMake。

### Linux Q13：如何使用 GDB 调试崩溃？

**参考回答：**

先使用带调试符号的构建：

```bash
cmake -S . -B build-debug -DCMAKE_BUILD_TYPE=Debug
cmake --build build-debug --parallel
gdb --args build-debug/demo_MEEF_Optimizer_init
```

GDB 中常用：

```text
run                 启动
bt                  当前线程调用栈
frame 2             切换栈帧
print variable      查看变量
info threads        查看线程
thread apply all bt 输出所有线程调用栈
```

多线程死锁时，`thread apply all bt` 特别重要。生产崩溃还应学习 core dump 和 `coredumpctl gdb`。

### Linux Q14：如何分析程序性能？

**参考回答：**

先测量，再优化：

```bash
/usr/bin/time -v ./demo
perf stat ./demo
perf record -g ./demo
perf report
```

- `time -v`：总时间、CPU 占用、最大常驻内存。
- `perf stat`：cycles、instructions、cache miss 等总体计数。
- `perf record/report`：找出热点函数和调用栈。

对本项目应分别测 SOCS 分解、单次成像、MEEF 矩阵构建、曲线渲染和磁盘 I/O，不能只看总时间。

### Linux Q15：如何检查内存、磁盘和系统负载？

**参考回答：**

```bash
free -h              # 内存使用
vmstat 1             # CPU、内存和调度概况
df -h                # 文件系统剩余空间
du -sh result build-* # 目录占用
uptime                # load average
```

EDA 任务产生大量中间矩阵时，磁盘写满会导致保存失败；内存不足可能触发 OOM killer。Linux 上还应检查系统日志，区分程序自身异常和系统强制终止。

### Linux Q16：Sanitizer 和 Valgrind 分别适合什么场景？

**参考回答：**

Sanitizer 在编译时插桩，速度通常比 Valgrind 快，适合 CI：ASan 检查越界和 use-after-free，UBSan 检查未定义行为，TSan 检查数据竞争。Valgrind 不要求重新编译或只需调试信息，但运行更慢，在某些平台和复杂库上兼容性有限。

OpenMP 程序使用 TSan 时可能看到运行时库相关报告，需要先构造最小复现并区分真实共享数据竞争和工具/运行时噪声。

### Linux Q17：SSH、SCP 和 rsync 有什么区别？

**参考回答：**

```bash
ssh user@server                         # 远程登录
scp config.yaml user@server:/work/run/ # 简单复制
rsync -av --progress ./ user@server:/work/Litho_cpp/
```

`rsync` 支持增量同步，重复传输大型工程或结果目录时更高效。同步前应通过 `.gitignore` 或排除规则避免上传 build、结果缓存和无关虚拟环境。

### Linux Q18：Shell 脚本至少需要掌握什么？

**参考回答：**

至少掌握变量、引用、参数、条件判断、循环、函数、退出码和重定向。推荐脚本开头：

```bash
#!/usr/bin/env bash
set -euo pipefail

build_dir="build-release"
cmake -S . -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$build_dir" \
  --target demo_MEEF_Optimizer_init --parallel

(
  cd "$build_dir"
  ./demo_MEEF_Optimizer_init
) 2>&1 | tee meef_run.log
```

- `-e`：未处理的命令失败时退出。
- `-u`：使用未定义变量时报错。
- `pipefail`：管道中任意关键命令失败都能被发现。
- 给变量加双引号，避免路径中空格和通配符展开。

### Linux Q19：EDA 计算集群上的作业调度是什么？

**参考回答：**

共享服务器不能让每个人随意占满 CPU 和内存，因此常用 Slurm、LSF 等调度系统。用户提交作业时声明 CPU、内存、运行时间和队列，调度器选择节点执行。

面试至少应理解：交互作业和批处理作业、资源申请、作业状态、日志文件、超时和取消作业。具体命令取决于公司的调度器，不应把本地 `nohup` 当成集群资源管理方案。

### Linux Q20：`rg`、`find`、`sed` 和 `awk` 各适合什么？

**参考回答：**

- `rg`：在源码或日志内容中快速搜索。
- `find`：按文件名、类型、时间或大小寻找文件。
- `sed`：逐行替换和简单文本转换。
- `awk`：按列处理结构化文本并统计。

```bash
rg -n 'omp parallel|fftw_execute' source include
find result -type f -name '*.txt'
awk -F, 'NR > 1 {sum += $3} END {print sum/(NR-1)}' errors.csv
```

面试不要求背所有参数，但要能组合命令快速定位源码、错误日志和实验结果。

### Linux 专项实操

1. 在 Linux 上从空构建目录完成配置、编译和运行，并保存完整日志。
2. 设置 `OMP_NUM_THREADS=1/2/4/8`，记录时间和加速比。
3. 运行时使用 `top -H` 确认各 OpenMP 线程是否占用 CPU。
4. 使用 `ldd` 解释 demo 依赖的动态库来自哪里。
5. 故意传入错误配置路径，检查退出码和 stderr 日志。
6. 用 GDB 在 `Imaging::compute` 设置断点并查看调用栈。
7. 用 `perf` 找出一次 MEEF 迭代的前三个热点。
8. 写一个 `build_and_run.sh`，完成构建、运行和日志保存。

---

## 9. 面试中的项目追问

### Q59：项目中你解决过最难的问题是什么？

**参考回答模板：**

不要只回答“实现了 MEEF”。按 STAR 表达：

1. **背景**：每个控制点四次成像，运行非常慢且出现并行不稳定。
2. **任务**：在保持数值一致的前提下加速矩阵构建。
3. **行动**：将控制点扰动展开为独立任务；进入并行区前构造线程私有 FFTW worker；关闭嵌套并行；固定强度累加顺序；增加计时与误差验证。
4. **结果**：给出真实的线程加速比、误差上界和测试规模。没有测量过的数字不要编造。

### Q60：如果结果和 Python 不一致，你怎么排查？

**参考回答：**

从最小输入开始逐层对比：坐标轴和单位、mask 归一化、FFT normalization、shift、光瞳频移、光源权重、SOCS 权重、aerial image、sigmoid、EP 顺序、EPE 符号、控制点坐标 `[y,x]`。每层保存中间矩阵，先找第一个产生差异的阶段。

### Q61：如果让你继续改进项目，你会先做什么？

**参考回答：**

优先级是：

1. 建立 GoogleTest + Python golden test。
2. 修复 FFTW 资源类的 Rule of Five 风险。
3. 用显式枚举区分 Abbe/SOCS。
4. 完成 Linux CI 和 Sanitizer。
5. 输出 SOCS 精度—速度曲线和 OpenMP scaling。
6. 加入 GDSII/OASIS 与 tile 处理。
7. 把 PV Band/MRC 约束加入优化目标。

### Q62：你的项目能证明你适合 EDA R&D 的哪三点？

**参考回答：**

第一，我能把傅里叶光学、线性代数和优化算法转成可运行的 C++ 数值代码；第二，我了解 EPE、MEEF、SRAF 和 process variation 等计算光刻问题；第三，我处理过 FFTW/Eigen/OpenMP 的工程和性能问题，并能通过基准、日志和数值对照验证结果。

### Q63：你目前最诚实的短板是什么？

**参考回答：**

我的项目目前仍以栅格和小规模实验为主，对真实 GDS/OASIS 层次化版图、工业工艺标定和大规模分布式计算接触不足。我正在通过补充版图数据库、回归测试和 Linux 性能分析来弥补，而不是把研究原型包装成工业产品。

---

## 10. 建议的学习顺序

### 第一阶段：能把自己的项目讲清楚（1～2 周）

- 熟练回答 Q1～Q29。
- 手画 Abbe/SOCS、MEEF 优化数据流。
- 为每个公式指出对应源码文件。
- 整理一个 10 分钟项目演示。

### 第二阶段：补 C++、数值和 HPC（2～4 周）

- 完成 Q33～Q58。
- 完成 CMake Q1～Q16 和 Linux Q1～Q20，并动手完成两个专项实操。
- 给 FFT、SOCS、EPE、MEEF 增加测试。
- 使用 ASan/UBSan 检查。
- 输出 OpenMP scaling 和 SOCS 精度—速度报告。

### 第三阶段：补工业 EDA 差距（4～8 周）

- 学习 GDSII/OASIS、polygon、R-tree 和 tile halo。
- 使用 KLayout 读取一个 GDS，并接入当前 raster/contour 管线。
- 增加 focus/dose process window 和 mask rule 检查。
- 熟悉 Linux、Python、Shell，了解 Tcl 基础。

---

## 11. 自测标准

当你可以不看文档完成以下任务，就基本具备应届 EDA R&D 面试的项目表达基础：

- 90 秒讲清项目目标、架构、算法和结果。
- 现场推导 Abbe/SOCS 和 MEEF 中心差分。
- 解释 SVD、病态矩阵和正则化。
- 解释 OpenMP/FFTW 的线程安全设计。
- 能从零完成 CMake 配置、构建、依赖排查，并解释 target 依赖传播。
- 能在 Linux 远程环境运行、保存日志、检查线程、调试崩溃和定位性能热点。
- 指出项目至少三个真实缺陷及修复方案。
- 给出数值正确性和性能验证方案。
- 说明研究原型到工业 EDA 工具还缺什么。
