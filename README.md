# Litho_cpp — 光刻仿真 C++ 库

## 依赖安装

### macOS (Homebrew)

```bash
brew install eigen fftw opencv libomp cmake
```

### Ubuntu / Debian

```bash
sudo apt install libeigen3-dev libfftw3-dev libopencv-dev cmake build-essential
```

SRAF 独立宽度优化使用 `libcmaes`。如果系统没有安装，CMake 会在第一次
配置时从官方仓库下载项目固定的版本，因此首次配置需要网络连接。

## 编译

### Release 模式（性能最优，~20x 提速）

```bash
cd /Users/wyh/Desktop/学校/Litho_cpp
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

### Debug 模式（含调试符号）

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)
```

### 只编译单个 Demo

```bash
cd build

# 编译特定目标
make demo_socs_imaging    # SOCS vs Abbe 成像对比
make demo_ep              # EP 评估点选择
make demo_meef            # MEEF 优化
make demo_bs              # 参数化曲线渲染
```

## 运行 Demo

```bash
cd build

# ── SOCS vs Abbe 成像对比 ──────────────────────────────────
cp ../lsm_mask.txt ./        # 拷贝掩模文件
./demo_socs_imaging          # 运行对比 + 自动生成 PNG 可视化图

# ── EP 评估点选择 ─────────────────────────────────────────
./demo_ep

# ── MEEF 优化 ─────────────────────────────────────────────
./demo_meef

# ── 曲线渲染（MSAA 抗锯齿）─────────────────────────────────
./demo_bs
```

## `demo_socs_imaging` 详解

同时计算 SOCS（10 模）和 Abbe（精确参考）的空间像 + wafer，
输出误差分析、加速比统计，并自动调用 `litho_viewer` 生成 4 张对比图。

### 输出文件

| 文件 | 说明 |
|------|------|
| `aerial_socs.txt` / `aerial_abbe.txt` | SOCS / Abbe 空间像 |
| `wafer_socs.txt` / `wafer_abbe.txt` | SOCS / Abbe wafer 图案 |
| `aerial_diff.txt` / `wafer_diff.txt` | 绝对误差矩阵 |
| `aerial_compare.png` | 空间像并排 + 差值图 |
| `wafer_compare.png` | Wafer 并排 + 差值图 |
| `aerial_diff.png` | 空间像误差热图 |
| `socs_vs_abbe_overview.png` | 四合一总览 |

### 终端输出示例

```
Grid: 257x257, pixel_size = 6 nm
Source pts: 248
SOCS: 248 source pts → 24 coherent modes

SOCS aerial time: 45 ms
Abbe aerial time: 850 ms
Speedup (SOCS/Abbe): 18.9x

Aerial error:  max = 0.065, RMS = 0.016 (5.5%)
Wafer error:   max = 0.77, RMS = 0.055, mis-match(>0.1) = 1606 / 66049 (2.4%)
```

## 项目结构

```
Litho_cpp/
├── include/
│   ├── litho_model/           # 光刻模型头文件
│   │   ├── grid.h             # 计算网格
│   │   ├── mask.h             # 掩模表示
│   │   ├── pupil.h            # 光瞳
│   │   ├── source.h           # 光源
│   │   ├── fft.h              # FFT 封装
│   │   ├── litho_prepare.h    # 成像预处理
│   │   └── imaging.h          # Abbe / SOCS 成像
│   └── utils/
│       ├── ep_select.h        # EP 选择
│       ├── msaa.h             # 抗锯齿渲染
│       ├── parametric.h       # 参数化曲线
│       ├── epe.h              # EPE 计算
│       ├── meef_pipeline.h    # MEEF 管线
│       └── meef_optimizer.h   # MEEF 优化器
├── source/                    # 对应 .cpp 实现
├── examples/                  # Demo 文件
├── CMakeLists.txt             # CMake 构建配置
├── litho_viewer.py            # Python 可视化脚本
├── litho_viewer_build.md      # 可视化工具打包与使用说明
└── README.md                  # 本文件
```
