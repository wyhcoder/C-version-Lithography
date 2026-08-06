# Litho_cpp 编译与运行指南

在一台新电脑上从零开始构建并运行本项目的完整流程。支持 **macOS** 和 **Windows**。

---

## 目录

- [0. 术语速览（这些工具都是干嘛的）](#0-术语速览这些工具都是干嘛的)
- [1. 项目依赖](#1-项目依赖)
- [2. macOS 安装步骤](#2-macos-安装步骤)
- [3. Windows 安装步骤](#3-windows-安装步骤)
- [4. 编译项目](#4-编译项目)
- [5. 编译命令详解](#5-编译命令详解)
- [6. 运行 demo](#6-运行-demo)
- [7. 常见问题](#7-常见问题)

---

## 0. 术语速览（这些工具都是干嘛的）

如果你是第一次接触 C++ 项目，下面这些名词可能陌生。按角色分类介绍：

### 🧰 包管理器（安装依赖库的工具）

| 名字 | 平台 | 作用 |
|---|---|---|
| **Homebrew** (`brew`) | macOS / Linux | macOS 上的"应用商店"，命令行装软件的标准工具 |
| **vcpkg** | Windows / 跨平台 | 微软开发的 C++ 库管理器，专门解决 Windows 上装 C++ 库的痛苦 |
| **apt** | Ubuntu/Debian | Linux 系统包管理器（本项目用不到） |
| **pip** | 跨平台 | Python 库管理器（`pip install numpy`） |

**为什么需要包管理器？**

C++ 不像 Python 那样自带包管理器。装一个 OpenCV 手动流程：下载源码 → 装依赖 → cmake 配置 → 编译 30 分钟 → 安装。用 Homebrew 就一条：

```bash
brew install opencv
```

自动搞定下载、编译、安装、依赖链。

---

### **Homebrew** 详解（macOS 用户必备）

**是什么**：macOS 上最流行的开源包管理器，官网 https://brew.sh。

**能做什么**：
```bash
brew install cmake         # 装 CMake
brew install eigen         # 装 Eigen3 库
brew list                  # 列出装过的东西
brew upgrade opencv        # 升级 OpenCV
brew uninstall fftw        # 卸载
brew --prefix libomp       # 查询安装路径
```

**安装位置**：
- Apple Silicon Mac（M1/M2/M3）→ `/opt/homebrew/`
- Intel Mac → `/usr/local/`

这就是为什么本项目 `CMakeLists.txt` 里写 `HINTS /opt/homebrew/lib/cmake/opencv4`——告诉 CMake "去这个路径找 OpenCV"。Intel Mac 需要改成 `/usr/local`。

**"formula" 和 "cask"**：
- **formula**：命令行工具/库（`brew install cmake`）
- **cask**：图形界面 App（`brew install --cask visual-studio-code`）

---

### **vcpkg** 详解（Windows 用户必备）

**是什么**：微软 2016 年开源的 C++ 库管理器，官网 https://vcpkg.io。

**为什么 Windows 需要它**：Windows 上 C++ 库没有统一的安装路径（不像 macOS 的 `/usr/local`），装 OpenCV 传统流程要手动下 lib、include、DLL，还得配 Visual Studio 项目属性——非常痛苦。vcpkg 一条命令搞定：

```powershell
.\vcpkg install opencv4:x64-windows
```

自动下源码、编译、安装到 `vcpkg/installed/x64-windows/`，还生成 CMake 的 `find_package` 配置。

**关键概念 — "triplet"（三元组）**：

```
opencv4:x64-windows
       └──────────┘
       目标平台
```

- `x64-windows`：64 位 Windows，动态链接（`.dll`）
- `x64-windows-static`：静态链接（`.lib` 全打进 exe）
- `x86-windows`：32 位
- `arm64-windows`：ARM Windows

大多数情况用 `x64-windows`。

**关键概念 — "toolchain file"**：

装完库要让 CMake 知道去哪找。vcpkg 提供一个"工具链文件"，配置阶段传给 CMake：

```powershell
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
```

这一句让 CMake 自动搜 `vcpkg/installed/` 里的所有库，`find_package(Eigen3)` 就能找到。

---

### ⚙️ 编译工具链

| 名字 | 作用 |
|---|---|
| **编译器**（clang / gcc / MSVC） | 把 `.cpp` 源码翻译成机器码 `.o` |
| **链接器**（ld / lld / link.exe） | 把多个 `.o` + 库文件拼成最终 exe |
| **CMake** | **构建系统生成器**，读 `CMakeLists.txt` 生成 Makefile/VS 项目 |
| **Make / MSBuild / Ninja** | 真正调用编译器的工具，读 Makefile 决定编译顺序 |
| **Visual Studio** | Windows 上的 IDE + 编译器套装（含 MSVC + MSBuild） |

**流程图**：

```
CMakeLists.txt  ──cmake──►  Makefile  ──make──►  clang/gcc  ──►  .o 文件  ──ld──►  demo.exe
                             (Windows: .sln)     (Windows: MSVC)
```

---

### 🔥 OpenMP

**是什么**：一套让 C/C++/Fortran 代码**多线程并行**的标准（不是库，是编译器扩展）。

**长什么样**：

```cpp
#pragma omp parallel for schedule(static)
for (int i = 0; i < 1000000; ++i) {
    heavy_compute(i);   // 自动分给多核跑
}
```

一行 `#pragma`（编译指令）就让循环并行化，本项目 `_compute_socs_kernels` 就用了。

**为什么单独说**：
- **gcc、MSVC 内置支持** OpenMP，直接就能用
- **Apple 的 clang 没有内置 OpenMP**（政治原因，苹果自己搞 GCD）→ 需要额外装 `libomp`（`brew install libomp`）

**CMake 里怎么用**：

```cmake
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
```

找到就自动加 `-fopenmp` 编译选项，没找到就跳过（并行代码退化成单线程也能跑）。

---

### 📐 依赖库

| 名字 | 类型 | 作用 |
|---|---|---|
| **Eigen3** | header-only C++ 库 | 矩阵/向量/线性代数（类似 Python 的 numpy） |
| **FFTW3** | C 库 | "Fastest Fourier Transform in the West"，业界最快的 FFT 实现 |
| **OpenCV** | C++ 库 | 图像处理（读写图片、连通域、轮廓、二值化等） |
| **yaml-cpp** | C++ 库 | 解析 YAML 文件（用于读 `config.yaml`） |

**header-only 库是什么**：像 Eigen 这样"只有 `.h` 头文件、没有 `.a`/`.lib` 编译产物"的库。用起来只需 `#include <Eigen/Dense>` 就行，不用链接。缺点是每个 .cpp 都要重新编译 Eigen 模板，编译慢。

---

### 🎨 生成器（Generator）

`cmake -G` 后面跟的东西，决定生成哪种构建文件：

| 生成器 | 平台 | 产物 |
|---|---|---|
| `Unix Makefiles`（默认，Mac/Linux） | Mac/Linux | Makefile，用 `make` 编 |
| `Visual Studio 17 2022`（默认，Win） | Windows | `.sln` + `.vcxproj`，用 VS 或 MSBuild |
| `Ninja` | 跨平台 | `build.ninja`，比 Make 快 2-5 倍 |
| `Xcode` | macOS | `.xcodeproj`，可以用 Xcode 打开 |

**用 Ninja 提速编译**：

```bash
brew install ninja       # macOS
# vcpkg install ninja   # Windows 也能装
cmake .. -G Ninja
cmake --build . -j
```

Ninja 启动更快、依赖分析更聪明，大项目差别明显。

---

### 🐍 Python 相关

本项目的**可视化脚本**用 Python（`scripts/show_multi.py` 等），需要装几个库：

| 库 | 作用 |
|---|---|
| `numpy` | 数值计算，加载 txt 数据 |
| `matplotlib` | 画图/弹窗展示 |
| `scipy` | 连通域分析（`scipy.ndimage.label`）等 |

装法：

```bash
pip3 install numpy matplotlib scipy
```

**核心 C++ 代码不依赖 Python**，只是 demo 结尾调 Python 弹图方便查看结果。

---

### 一图看懂全流程

```
┌─ 你的机器 ──────────────────────────────────────────────┐
│                                                       │
│  [Homebrew / vcpkg]  ──装依赖──►  Eigen / FFTW / OpenCV │
│                                       │               │
│                                       ▼               │
│  CMakeLists.txt ──cmake──► Makefile / .sln            │
│                                       │               │
│                                       ▼               │
│                            clang/MSVC + OpenMP        │
│                                       │               │
│                                       ▼               │
│                                  demo_pvband          │
│                                       │               │
│                                       ▼               │
│                              result/*.txt             │
│                                       │               │
│                                       ▼               │
│                              Python matplotlib ──弹图  │
└───────────────────────────────────────────────────────┘
```

---

---

## 1. 项目依赖

| 依赖 | 用途 | 版本要求 |
|---|---|---|
| **C++ 编译器** | 主体代码 | 支持 C++20（clang ≥ 12 / gcc ≥ 10 / MSVC 2019+） |
| **CMake** | 构建系统 | ≥ 3.16 |
| **Eigen3** | 矩阵/线性代数 | ≥ 3.4 |
| **FFTW3** | 傅里叶变换 | ≥ 3.3 |
| **OpenCV** | 图像处理（连通域、轮廓、图像 I/O） | ≥ 4.5 |
| **yaml-cpp** | 读取 `config.yaml` | ≥ 0.7 |
| **OpenMP** | 并行加速（可选，但强烈推荐） | 编译器自带 |
| **Python 3** | 可视化脚本 | ≥ 3.8 |
| Python 库 | `numpy` / `matplotlib` / `scipy` | 最新版即可 |

---

## 2. macOS 安装步骤

### 2.1 安装 Homebrew（若尚未安装）

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Apple Silicon（M 系列芯片）Homebrew 装在 `/opt/homebrew`，Intel Mac 装在 `/usr/local`。项目 `CMakeLists.txt` 已配好 `HINTS /opt/homebrew/lib/...`。**Intel Mac 用户** 需要在 `CMakeLists.txt` 中把 `/opt/homebrew` 改成 `/usr/local`，或者删掉 `HINTS`（让 CMake 自己找）。

### 2.2 安装依赖

```bash
brew install cmake eigen fftw opencv yaml-cpp libomp python@3.11
```

### 2.3 安装 Python 库

```bash
pip3 install numpy matplotlib scipy
```

### 2.4 编译器 OpenMP 支持

macOS 自带的 clang **默认不带 OpenMP**，`brew install libomp` 装完后需要在编译时告诉编译器 include/lib 路径。已经在 CMake 里通过 `find_package(OpenMP)` 处理，一般不用手动配。

如果 CMake 找不到 OpenMP，可以显式指定：

```bash
export OpenMP_ROOT=$(brew --prefix libomp)
```

再重新 cmake 配置。

---

## 3. Windows 安装步骤

推荐使用 **vcpkg + Visual Studio 2022** 组合。

### 3.1 安装 Visual Studio 2022

从 https://visualstudio.microsoft.com/ 下载 Community 版本，安装时勾选 **"使用 C++ 的桌面开发"** 工作负载。

### 3.2 安装 CMake

从 https://cmake.org/download/ 下载 Windows installer，安装时勾选 **"Add CMake to system PATH"**。

### 3.3 安装 vcpkg

```powershell
# 打开 PowerShell
cd C:\
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat
.\vcpkg integrate install
```

### 3.4 通过 vcpkg 安装依赖

```powershell
cd C:\vcpkg
.\vcpkg install eigen3:x64-windows fftw3:x64-windows opencv4:x64-windows yaml-cpp:x64-windows
```

安装时间较长（约 20-60 分钟，OpenCV 编译需要时间）。

### 3.5 安装 Python 3

从 https://www.python.org/ 下载 Python 3.10+，安装时勾选 **"Add Python to PATH"**。

```powershell
pip install numpy matplotlib scipy
```

### 3.6 Windows 上的注意事项

- 项目当前 `CMakeLists.txt` 里有 macOS 特定的 `HINTS /opt/homebrew/...`，Windows 上会 fallback 到 vcpkg 自动搜索
- Windows 上使用 vcpkg 的 toolchain 文件即可让 CMake 自动找到所有库
- `demo_pvband.cpp` 里有 macOS 硬编码路径（如 `/Users/wyh/Desktop/学校/Litho_cpp/result/...`），Windows 上需要改成对应的绝对路径或相对路径

---

## 4. 编译项目

### 4.1 克隆代码

```bash
git clone <你的仓库地址> Litho_cpp
cd Litho_cpp
```

### 4.2 macOS / Linux 编译

**Release 版本（用于日常仿真，推荐）：**

```bash
mkdir -p build-release && cd build-release
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(sysctl -n hw.ncpu)
```

**Debug 版本（仅用于 gdb/lldb 调试）：**

```bash
mkdir -p build-debug && cd build-debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . -j4
```

> ⚠️ **重要**：Debug 版本运行速度比 Release 慢 10-100 倍（Eigen 在 Debug 下没有向量化和内联优化）。日常运行仿真**必须**使用 Release 版本。

### 4.3 Windows 编译（vcpkg 集成）

```powershell
cd Litho_cpp
mkdir build-release
cd build-release

# 用 vcpkg 的 toolchain
cmake .. -DCMAKE_BUILD_TYPE=Release ^
         -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake

# 编译（用 Visual Studio 生成器）
cmake --build . --config Release -j
```

Release 二进制会在 `build-release/Release/` 下。

### 4.4 编译单个 target

```bash
cmake --build . --target demo_pvband -j4
```

可用的 target：
- `demo_CTM` — CTM（Continuous Transmission Mask）优化
- `demo_LSM` — Level Set Method 掩模优化
- `demo_epe` — EPE loss + 梯度测试
- `demo_pvband` — PV band 计算
- `demo_fft_shift` — FFT shift 测试
- `demo_bs`、`demo_ep`、`demo_meef` — 早期示例（部分可能已 API 过时）

---

## 5. 编译命令详解

上面用到的每一条命令、每一个选项都是什么意思？下面逐个拆解。

### 5.1 CMake 是什么

CMake **不直接编译代码**，它是一个 **"构建系统生成器"**：读取 `CMakeLists.txt`（描述"要编译哪些源文件、依赖什么库"），生成对应平台的构建文件：

| 平台 | CMake 生成的东西 | 实际编译工具 |
|---|---|---|
| macOS/Linux | `Makefile` | `make` |
| Windows + VS | `*.sln` / `*.vcxproj` | `MSBuild` |
| 跨平台 | `build.ninja` | `ninja`（更快） |

所以整个流程是**两步**：
1. **配置阶段**（`cmake ..`）：生成构建文件
2. **构建阶段**（`cmake --build .`）：调用底层工具真正编译

---

### 5.2 配置阶段：`cmake .. -DCMAKE_BUILD_TYPE=Release`

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

拆开逐个说：

| 部分 | 含义 |
|---|---|
| `cmake` | CMake 可执行程序本身 |
| `..` | **源码根目录**（相对路径）。你在 `build-release/` 里，`..` 就是项目根，那里放着 `CMakeLists.txt` |
| `-D...` | 定义一个 CMake 变量（`-D` = Define） |
| `CMAKE_BUILD_TYPE=Release` | 变量名=值。告诉 CMake 用 Release 优化等级 |

**`CMAKE_BUILD_TYPE` 可选值**：

| 值 | 编译选项 | 用途 |
|---|---|---|
| `Release` | `-O3 -DNDEBUG` | 最高优化 + 关闭 assert，运行速度快 |
| `Debug` | `-O0 -g` | 无优化 + 调试符号，可以 gdb/lldb 单步 |
| `RelWithDebInfo` | `-O2 -g -DNDEBUG` | 有优化但保留调试符号，用于 profile |
| `MinSizeRel` | `-Os -DNDEBUG` | 优化体积（嵌入式常用） |

**为什么 Release 快 10-100 倍？**
- `-O3`：编译器做**内联、循环展开、SIMD 向量化、常量折叠**等激进优化
- `-DNDEBUG`：定义 `NDEBUG` 宏 → Eigen 的 `eigen_assert()` 全部消失，每次矩阵访问不再做边界检查
- Debug 下这些全没有，热点循环慢几十倍

**配置阶段做什么**：
1. 检查编译器是否可用（`find_package` C++ 编译器）
2. 找依赖库（`find_package(Eigen3)`、`find_package(FFTW3)` 等）
3. 检查 OpenMP 支持
4. 生成 `Makefile`（或 `.sln`）到当前目录
5. 把 `config.yaml` 和 `assets/target_pattern/` 复制到 build 目录（这是 CMakeLists.txt 里配的，运行时可用相对路径 `target_pattern/xxx.bmp` 找图）

**常用附加选项**：

```bash
# 指定编译器（不用系统默认的）
cmake .. -DCMAKE_CXX_COMPILER=/usr/bin/clang++

# 指定安装前缀
cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local

# 用 Ninja 生成器（比 Make 快，需要 brew install ninja）
cmake .. -G Ninja

# Windows + vcpkg 集成
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
```

---

### 5.3 构建阶段：`cmake --build . -j4`

```bash
cmake --build . -j4
```

拆解：

| 部分 | 含义 |
|---|---|
| `cmake --build` | 让 CMake 调用底层构建工具（Make/MSBuild/Ninja）编译 |
| `.` | 构建目录（就是当前目录 `build-release/`） |
| `-j4` | 并行 4 个编译任务 |

**`-j` 到底是几**：`j` 是 GNU make 的传统参数（**jobs**）。数值越大越并行，但受限于 CPU 核数：

```bash
# macOS 查核数
sysctl -n hw.ncpu           # 例如输出 10

# Linux
nproc                       # 例如输出 8

# 用全部核数
cmake --build . -j$(sysctl -n hw.ncpu)    # macOS
cmake --build . -j$(nproc)                # Linux
```

`-j` 不给数字（`-j`）表示 "无限制并行"——不推荐，可能内存爆掉。

**`--target xxx` 只编译指定 target**：

```bash
cmake --build . --target demo_pvband -j4
```

不加 `--target` 就编译所有 target（`add_executable` 加过的每个都会编）。项目里有十几个 demo，全编要更久。

**`--config Release`（Windows 才需要）**：

Visual Studio 生成器是"多配置"的——同一个 `.sln` 同时支持 Debug/Release。**Windows 下配置阶段不用 `-DCMAKE_BUILD_TYPE`，改在构建阶段传：**

```powershell
cmake --build . --config Release -j
```

macOS/Linux 的 Makefile 是"单配置"，`-DCMAKE_BUILD_TYPE` 在配置阶段就定死了。

---

### 5.4 一条完整命令行的分解

以下这一整行：

```bash
cd build-release && cmake --build . --target demo_pvband -j4 && ./demo_pvband ../config.yaml
```

拆成三步用 `&&` 连接（**前一步成功才执行下一步**）：

1. `cd build-release`：进入 Release 构建目录
2. `cmake --build . --target demo_pvband -j4`：只编译 `demo_pvband` 这一个 target，并行 4 线程
3. `./demo_pvband ../config.yaml`：运行编译好的二进制，第一个参数是 config 路径

`../config.yaml` 是相对路径——你在 `build-release/`，`..` 是项目根，那里有 `config.yaml`。

---

### 5.5 CMakeLists.txt 关键片段解读

打开 `CMakeLists.txt` 主要看这几块：

```cmake
cmake_minimum_required(VERSION 3.16)         # 需要 CMake ≥ 3.16
project(LithoSim VERSION 1.0.0 LANGUAGES CXX) # 项目名 LithoSim，只用 C++

set(CMAKE_CXX_STANDARD 20)                    # 用 C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)           # 编译器不支持 C++20 就报错
set(CMAKE_CXX_EXTENSIONS OFF)                 # 关闭 GNU 扩展，用标准 C++

find_package(Eigen3 REQUIRED)                 # 找 Eigen，找不到就中止
find_package(OpenCV REQUIRED HINTS ...)       # HINTS 是"优先看这些路径"
find_package(yaml-cpp REQUIRED HINTS ...)

# 编个静态库 litho_core，包含所有 .cpp
add_library(litho_core STATIC
    source/litho_model/grid.cpp
    ...
)

# 让 litho_core 能找到 include/ 里的头文件
target_include_directories(litho_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/...
    ${EIGEN3_INCLUDE_DIR}       # find_package 找到 Eigen 后自动设的变量
    ...
)

# 链接依赖库
target_link_libraries(litho_core PUBLIC
    Eigen3::Eigen               # 用 imported target（推荐方式）
    ${FFTW3_LIBRARY}
    ${OpenCV_LIBS}
    yaml-cpp::yaml-cpp
)

# 每个 demo 都是独立的可执行文件，链接 litho_core
add_executable(demo_pvband demo/demo_pvband.cpp)
target_link_libraries(demo_pvband litho_core)
```

**核心概念**：
- `add_library(litho_core STATIC ...)`：把所有源文件打包成一个 **静态库** `.a`（Windows 是 `.lib`）
- `add_executable(demo_xxx ...)`：定义一个可执行 target
- `target_link_libraries(demo_xxx litho_core)`：demo 链接静态库，就自动获得所有源码功能

**为什么不每个 demo 单独编译所有源文件？**
- 一份源码只需编译一次，所有 demo 复用
- 修改一个 .cpp 只重编那一个，加上链接步骤，比全量重编快 10 倍

**`STATIC` vs `SHARED`**：
- `STATIC`：`.a`/`.lib`，链接时代码被拷进 exe，运行不依赖外部文件
- `SHARED`：`.dylib`/`.dll`/`.so`，运行时动态加载，多个程序可共享同一份库

本项目用 STATIC，简单省事。

---

### 5.6 CMake 缓存与重新配置

**CMake 会记住上次的配置到 `CMakeCache.txt`**。如果改了 `CMakeLists.txt` 或想清空重来：

```bash
# 方式1：清空 build 目录重新配置
rm -rf build-release/*
cd build-release
cmake .. -DCMAKE_BUILD_TYPE=Release

# 方式2：只清删除 CMake 缓存
rm build-release/CMakeCache.txt
```

**平时改 .cpp/.h 不需要重新 cmake**——直接 `cmake --build .` 即可，Make 会检测文件变化只编译改动的部分（**增量编译**）。

---

## 6. 运行 demo

### 5.1 前置：目标图形

`config.yaml` 里的 `PATTERN_NAME` 决定用哪张 target 图。源图放在 `assets/target_pattern/` 下（如 `assets/target_pattern/工字型.bmp`），编译时 CMake 会把它复制到 `build-*/target_pattern/`，代码通过相对路径 `target_pattern/xxx.bmp` 读取。

### 5.2 运行

```bash
cd build-release
./demo_CTM ../config.yaml       # 先跑 CTM 得到 binary_mask.txt
./demo_LSM ../config.yaml       # 再基于 CTM 结果做 LSM 优化
./demo_pvband ../config.yaml    # 计算 PV band
./demo_epe ../config.yaml       # 测试 EPE loss + 梯度
```

> Windows 上是 `.\demo_CTM.exe ..\config.yaml`

### 5.3 结果输出

所有中间结果 txt 会保存到 `result/` 目录，可视化 PNG 也在这里。CTM 结果在 `result/CTM_result/`。

### 5.4 可视化

多数 demo 结尾会调 Python 脚本 `scripts/show_multi.py` 弹窗展示。用法：

```bash
python3 scripts/show_multi.py path/to/file1.txt gray path/to/file2.txt hot
```

参数是 **`文件路径 colormap`** 对，支持任意个（会自动拼成一个大图）。

---

## 7. 常见问题

### Q1: `CMake Error: Could not find Eigen3 / FFTW3 / OpenCV`

**macOS**：确认 `brew list` 里有对应包。Intel Mac 需要改 CMakeLists.txt 里的 `HINTS`。

**Windows**：`cmake ..` 时必须加 `-DCMAKE_TOOLCHAIN_FILE=<vcpkg_root>/scripts/buildsystems/vcpkg.cmake`。

### Q2: 编译时报 `error: no matching constructor for initialization of 'litho::Grid'`（demo_socs.cpp）

`examples/demo_socs.cpp` 用的是老 API，可暂时不编译。修改 `CMakeLists.txt` 注释掉 `add_executable(demo_socs_imaging ...)` 那两行即可。

### Q3: 运行时报 `load_txt: cannot open ...`

`SaveTxt::load_txt` 对**不含 `/` 的相对路径**会自动拼 `result/` 前缀，含 `/` 的路径按相对当前工作目录使用。建议用**绝对路径**避免坑：

```cpp
SaveTxt::load_txt("/Users/xxx/Litho_cpp/result/CTM_result/binary_mask.txt", mask);
```

Windows 上改成 `C:/Users/.../Litho_cpp/...`。**部分 demo 里有硬编码的 macOS 绝对路径**，Windows 上需要改。

### Q4: 运行很慢（例如 pvband 花了 17 秒）

**几乎 100% 是跑了 Debug 版本**。确认工作目录：

```bash
pwd   # 应该是 .../build-release，不是 build-debug
```

Debug 版二进制约 5 MB，Release 版约 800 KB，可用 `ls -la demo_xxx` 对比。

### Q5: macOS 找不到 OpenMP

```bash
brew install libomp
export OpenMP_ROOT=$(brew --prefix libomp)
```

然后 `rm -rf build-release/*` 重新 cmake 配置。

### Q6: `show_multi.py` 报 `FileNotFoundError`

`show_multi.py` 对不含 `/` 的路径默认从 `result/` 找；含 `/` 的按相对工作目录（一般是 `build-release/`）找。用绝对路径最稳。

### Q7: Python 弹窗不显示（macOS）

macOS 上 matplotlib 默认后端可能不弹窗。设置环境变量：

```bash
export MPLBACKEND=MacOSX
```

或在脚本里加 `matplotlib.use('MacOSX')`。

---

## 8. 目录结构速览

```
Litho_cpp/
├── CMakeLists.txt                # 构建定义
├── config.yaml                   # 仿真参数（光学/光源/胶模型）
├── README.md                     # 项目说明
│
├── include/                      # 头文件
│   ├── litho_model/              # Grid / Pupil / Source / Imaging 等
│   ├── utils/                    # loss / gradient / pv_band_computer 等
│   └── optimizer/                # CTM / LSM 优化器
├── source/                       # C++ 实现（对应 include 三个子目录）
│
├── demo/                         # 主要 demo（推荐入口）
│   ├── demo_CTM.cpp
│   ├── demo_LSM.cpp
│   ├── demo_pvband.cpp
│   ├── demo_epe.cpp
│   └── demo_fftshift.cpp
├── examples/                     # 早期示例（部分 API 过时）
│
├── assets/                       # 输入资源
│   ├── target_pattern/           # 目标图形 bmp
│   └── to_opt_mask/              # 待优化的初始 mask
│
├── scripts/                      # Python 工具脚本（可视化 + 辅助）
│   ├── show_multi.py             # 多图并排展示（C++ demo 调用）
│   ├── show.py                   # 单图展示
│   ├── plot_matrix.py            # 通用矩阵绘图
│   ├── plot.py / plot_ep.py      # 早期绘图脚本
│   ├── litho_viewer.py           # 交互式 GUI
│   └── test.py                   # 测试脚本
│
├── docs/                         # 文档
│   ├── BUILD.md                  # 本文件
│   └── litho_viewer_build.md
│
├── result/                       # 输出目录（.txt / .png）
├── build-release/                # Release 构建产物（推荐用于日常仿真）
└── build-debug/                  # Debug 构建产物（仅调试用）
```

---

## 9. 快速验证（一键跑通）

macOS：

```bash
# 从零开始
git clone <repo> Litho_cpp && cd Litho_cpp
brew install cmake eigen fftw opencv yaml-cpp libomp
pip3 install numpy matplotlib scipy
mkdir -p build-release && cd build-release
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --target demo_CTM -j$(sysctl -n hw.ncpu)
./demo_CTM ../config.yaml
```

跑完在 `result/CTM_result/` 会看到 `binary_mask.txt` 等文件，说明整套依赖装好、代码编译通过、运行成功。
