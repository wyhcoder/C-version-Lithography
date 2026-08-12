# 使用 CMake 编译并运行一个 demo：完整指南

本文以 `demo_MEEF_Optimizer_init` 为例，说明如何从项目源码完成环境检查、CMake 配置、单目标编译、YAML 设置、程序运行、日志查看和常见错误排查。

适用项目目录：

```text
Litho_cpp/
├── CMakeLists.txt
├── config.yaml
├── demo/demo_MEEF_Optimizer_init.cpp
├── include/
├── source/
├── assets/
└── docs/
```

---

## 1. 先理解完整构建流程

```text
CMakeLists.txt
      │
      │ cmake 配置
      ▼
Makefile 或 build.ninja
      │
      │ make / ninja / cmake --build
      ▼
各个 .cpp 编译为 .o
      │
      ├── 链接为 liblitho_core.a
      │
      └── demo 入口 + litho_core + 第三方库
                         │
                         ▼
              demo_MEEF_Optimizer_init
```

三个阶段不能混淆：

1. **配置**：CMake 查找编译器和依赖，生成构建规则。
2. **构建**：Make/Ninja 调用编译器，只重新编译发生变化的源码。
3. **运行**：启动已经生成的可执行程序，读取 YAML 和图片等资源。

---

## 2. 这个 demo 依赖哪些内容

### 2.1 C++ 构建依赖

- 支持 C++20 的 Clang/GCC
- CMake 3.16 或更高版本
- Eigen3
- FFTW3
- OpenCV
- yaml-cpp
- OpenMP；macOS 使用 Homebrew `libomp`

### 2.2 运行时文件

`demo_MEEF_Optimizer_init` 默认需要：

```text
build-release/config.yaml
build-release/target_pattern/<pattern_name>.bmp
assets/lsm_mask/<LSM mask 文件>
```

其中：

- `config.yaml` 在 CMake 配置阶段复制到构建目录。
- `assets/target_pattern` 在 CMake 配置阶段复制为构建目录中的 `target_pattern`。
- LSM mask 路径来自 YAML，并按照项目根目录解析。
- 输出目录也来自 YAML，并按照项目根目录解析。

---

## 3. 第一次构建前检查环境

从项目根目录打开终端：

```bash
cd /Users/wangyuhang/Desktop/school/Litho_cpp
pwd
```

预期：

```text
/Users/wangyuhang/Desktop/school/Litho_cpp
```

检查关键工具：

```bash
command -v cmake
command -v clang++
command -v make
command -v ninja
cmake --version
clang++ --version
uname -m
```

Apple Silicon Mac 的架构通常显示：

```text
arm64
```

检查 Homebrew 依赖：

```bash
brew list cmake
brew list eigen
brew list fftw
brew list opencv
brew list yaml-cpp
brew list libomp
```

缺少依赖时可以安装：

```bash
brew install cmake eigen fftw opencv yaml-cpp libomp
```

如果准备使用 Ninja：

```bash
brew install ninja
```

---

## 4. 选择构建目录和生成器

构建目录用来保存：

- `CMakeCache.txt`
- Makefile 或 `build.ninja`
- `.o` 目标文件
- `liblitho_core.a`
- demo 可执行程序
- 复制后的 `config.yaml` 和 `target_pattern`

不要把这些文件直接生成到源码目录中。

### 4.1 继续使用当前 `build-release`

当前项目的 `build-release` 已经使用：

```text
Unix Makefiles
```

因此可以重新配置：

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release
```

这里没有重新指定 `-G`，CMake 会继续使用缓存中的 Unix Makefiles。

### 4.2 新建 Ninja 构建目录

如果希望使用 Ninja，应创建一个新目录，不能在现有 Makefile 构建目录中直接切换生成器：

```bash
cmake -S . -B build-ninja \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release
```

后续将文档中的 `build-release` 替换为 `build-ninja` 即可。

### 4.3 参数含义

```text
-S .                         源码目录是当前项目根目录
-B build-release             构建产物写入 build-release
-G Ninja                     使用 Ninja 生成器
-DCMAKE_BUILD_TYPE=Release   开启 Release 优化
```

---

## 5. 执行 CMake 配置

在项目根目录运行：

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release
```

CMake 会执行以下工作：

1. 检查 C++ 编译器。
2. 设置 C++20。
3. 查找 Eigen、OpenCV、yaml-cpp 和 FFTW。
4. 在 macOS 上查找 Homebrew `libomp`。
5. 生成 `litho_core` 和所有 demo 的构建规则。
6. 复制 `config.yaml`。
7. 复制 `assets/target_pattern`。
8. 生成 `compile_commands.json` 供 IDE 使用。

重点检查配置输出中是否出现：

```text
OpenMP enabled
Using Homebrew libomp: ...
Configuring done
Generating done
```

确认构建目录对应当前项目：

```bash
rg 'CMAKE_HOME_DIRECTORY|CMAKE_BUILD_TYPE|CMAKE_GENERATOR' \
  build-release/CMakeCache.txt
```

`CMAKE_HOME_DIRECTORY` 应当指向当前 `Litho_cpp` 项目根目录。

---

## 6. 只编译一个 demo

推荐使用生成器无关的命令：

```bash
cmake --build build-release \
  --target demo_MEEF_Optimizer_init \
  --parallel 4
```

各参数含义：

```text
--build build-release                 使用这个构建目录
--target demo_MEEF_Optimizer_init     只构建指定 demo 及其依赖
--parallel 4                          最多并行执行 4 个编译任务
```

注意，`--parallel 4` 控制的是“编译过程中的并行任务数”，不是程序运行时的 OpenMP 线程数。

构建过程通常是：

```text
编译 litho_core 中发生变化的 .cpp
        ↓
链接 liblitho_core.a
        ↓
编译 demo/demo_MEEF_Optimizer_init.cpp
        ↓
链接 demo_MEEF_Optimizer_init
```

成功标志：

```text
[100%] Built target demo_MEEF_Optimizer_init
```

### 6.1 你当前使用的 `make` 命令

如果当前目录已经是 `build-release`：

```bash
cd build-release
make demo_MEEF_Optimizer_init -j4
```

它和下面的目的相同：

```bash
cmake --build build-release \
  --target demo_MEEF_Optimizer_init \
  --parallel 4
```

区别是 `make` 只适用于 Unix Makefiles，而 `cmake --build` 同时适用于 Make 和 Ninja。

---

## 7. 检查构建产物

```bash
ls -lh build-release/demo_MEEF_Optimizer_init
ls -lh build-release/liblitho_core.a
ls -lh build-release/config.yaml
ls -ld build-release/target_pattern
```

检查 YAML 中指定的 target 是否存在：

```bash
rg 'pattern_name|run_mode|output_dir|lsm_mask_path' \
  build-release/config.yaml
```

例如：

```yaml
meef:
  output_dir: "result/MEEF_result/test"
  lsm_mask_path: "assets/lsm_mask/ls_image工字型.txt"
  run_mode: "--init-only"
  pattern_name: "工字型"
```

那么应该存在：

```text
build-release/target_pattern/工字型.bmp
assets/lsm_mask/ls_image工字型.txt
```

---

## 8. 选择 demo 运行模式

`demo_MEEF_Optimizer_init` 的运行模式由 YAML 中的字段控制：

```yaml
meef:
  run_mode: "--init-only"
```

支持三种模式：

| 模式 | 作用 | 建议用途 |
|---|---|---|
| `--init-only` | 只初始化仿真器和优化器 | 第一次验证环境 |
| `--build-meef` | 构建并保存 `Mx/My` | 验证中心差分和多线程 |
| `--optimize` | 执行完整 MEEF 优化 | 正式实验 |

第一次运行建议使用：

```yaml
meef:
  run_mode: "--init-only"
```

完整优化还会读取：

```yaml
meef:
  iter: 50
  optimize_wepe_only: false
```

为了快速验证流程，可以先把 `iter` 设置得较小，再执行正式长时间实验。

### 修改哪一份 YAML

有两种方式。

方式一：修改项目根目录的 `config.yaml`，然后重新执行 CMake 配置，让它复制到构建目录：

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release
```

方式二：从构建目录运行时，显式传入项目根目录的配置：

```bash
cd build-release
./demo_MEEF_Optimizer_init ../config.yaml
```

demo 只接受零个或一个命令行参数：

```text
./demo_MEEF_Optimizer_init [config.yaml]
```

其他运行参数应写在 YAML 的 `meef` 节中。

---

## 9. 正确运行 demo

该 demo 约定从构建目录启动，因为它通过当前工作目录寻找：

```text
./config.yaml
./target_pattern/<pattern_name>.bmp
```

运行：

```bash
cd build-release
./demo_MEEF_Optimizer_init
```

或者传入指定配置：

```bash
./demo_MEEF_Optimizer_init ../config.yaml
```

不要默认从项目根目录这样运行：

```bash
./build-release/demo_MEEF_Optimizer_init
```

因为此时当前工作目录是项目根目录，demo 会把项目根目录错误地当作构建目录，进而影响默认配置、target 和项目根目录的推导。

---

## 10. 运行时 OpenMP 线程数

编译并行和运行并行是两件事：

```text
cmake --build ... --parallel 4   控制编译任务数量
OMP_NUM_THREADS=4 ./demo         控制程序运行时 OpenMP 线程数量
```

例如：

```bash
cd build-release
OMP_NUM_THREADS=4 ./demo_MEEF_Optimizer_init
```

显示 OpenMP 环境：

```bash
OMP_NUM_THREADS=4 \
OMP_DISPLAY_ENV=TRUE \
./demo_MEEF_Optimizer_init
```

性能测试应分别尝试：

```bash
OMP_NUM_THREADS=1 ./demo_MEEF_Optimizer_init
OMP_NUM_THREADS=2 ./demo_MEEF_Optimizer_init
OMP_NUM_THREADS=4 ./demo_MEEF_Optimizer_init
OMP_NUM_THREADS=8 ./demo_MEEF_Optimizer_init
```

每次测试应使用相同输入、相同 Release 构建和相同运行模式。

---

## 11. 终端输出和日志保存

这个 demo 会自动把 `std::cout` 和 `std::cerr` 同时写入终端与日志文件。日志名称来自：

```yaml
meef:
  output_dir: "result/MEEF_result/test"
  console_log_name: "console_output.log"
```

对应文件：

```text
result/MEEF_result/test/console_output.log
```

检查日志：

```bash
cd ..
tail -n 50 result/MEEF_result/test/console_output.log
```

如果还想用 Shell 保存一份外层日志：

```bash
cd build-release
./demo_MEEF_Optimizer_init 2>&1 | tee manual_run.log
```

查看退出状态：

```bash
echo $?
```

- `0`：运行成功。
- `1`：配置、路径或计算过程中抛出异常。
- `2`：优化器构造成功，但初始化输出文件不完整。

---

## 12. 预期输出文件

初始化成功后，输出目录至少包含：

```text
main_cps.txt
sraf_cps.txt
eps.txt
sraf_mask.txt
main_mask.txt
config_used.yaml
console_output.log
```

`--build-meef` 还会生成：

```text
meef_mx.txt
meef_my.txt
```

`--optimize` 还会生成：

```text
errors.csv
lsm_mask.txt
lsm_aerial.txt
lsm_wafer.txt
target_mask.txt
iterations/
best_epe/
best_wepe/
```

---

## 13. 可视化 Python 环境

完整优化结束后，demo 会调用 `scripts/show_multi.py` 显示：

```text
LSM mask / LSM wafer / optimized mask / optimized wafer
```

脚本需要 NumPy 和 Matplotlib。推荐在项目根目录创建虚拟环境：

```bash
cd /Users/wangyuhang/Desktop/school/Litho_cpp
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install numpy matplotlib
```

demo 会优先使用：

```text
<项目根目录>/.venv/bin/python
```

如果缺少依赖，可能看到：

```text
ModuleNotFoundError: No module named 'numpy'
```

这属于可视化 Python 环境问题，不代表 C++ demo 编译失败。

---

## 14. 修改代码后的增量编译

例如修改：

```text
source/litho_model/imaging.cpp
```

重新构建同一个目标：

```bash
cd /Users/wangyuhang/Desktop/school/Litho_cpp
cmake --build build-release \
  --target demo_MEEF_Optimizer_init \
  --parallel 4
```

构建系统会自动完成：

```text
重新编译 imaging.cpp
    ↓
重新链接 litho_core
    ↓
重新链接 demo_MEEF_Optimizer_init
```

没有变化的 `.cpp` 不会重新编译。

修改 `CMakeLists.txt`、新增 `.cpp`、新增 target 或修改资源后，应先重新配置：

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release

cmake --build build-release \
  --target demo_MEEF_Optimizer_init \
  --parallel 4
```

---

## 15. Debug 构建和 Release 构建

Debug 与 Release 应使用不同目录：

```bash
cmake -S . -B build-debug \
  -DCMAKE_BUILD_TYPE=Debug

cmake --build build-debug \
  --target demo_MEEF_Optimizer_init \
  --parallel 4
```

运行 Debug：

```bash
cd build-debug
./demo_MEEF_Optimizer_init
```

用途区别：

| 构建类型 | 主要用途 |
|---|---|
| Debug | 断点、查看变量、排查崩溃 |
| Release | 性能测试和正式数值实验 |
| RelWithDebInfo | 带优化的性能调试 |

不能用 Debug 时间评价 OpenMP 或 SOCS 性能。

---

## 16. 常见错误排查

### 16.1 `CMAKE_HOME_DIRECTORY` 指向旧路径

检查：

```bash
rg CMAKE_HOME_DIRECTORY build-release/CMakeCache.txt
```

如果不是当前项目路径，最稳妥的做法是使用一个新的构建目录：

```bash
cmake -S . -B build-release-new \
  -DCMAKE_BUILD_TYPE=Release
```

### 16.2 在同一个构建目录切换 Make 和 Ninja

典型原因：`build-release` 原来由 Unix Makefiles 生成，后来又执行：

```bash
cmake -S . -B build-release -G Ninja
```

CMake 会报告生成器不匹配。解决办法是继续使用原生成器，或者为 Ninja 新建 `build-ninja`。

### 16.3 找不到 Eigen/OpenCV/yaml-cpp/FFTW

先检查 Homebrew：

```bash
brew --prefix eigen
brew --prefix fftw
brew --prefix opencv
brew --prefix yaml-cpp
```

再重新执行 CMake 配置，并从第一条 `Could NOT find...` 或 `not found` 信息开始排查。

### 16.4 OpenMP 没有启用

检查：

```bash
brew --prefix libomp
rg 'fopenmp|libomp' build-release/compile_commands.json
```

重新配置时应看到：

```text
OpenMP enabled
```

没有 OpenMP 时程序通常仍可构建，但并行区域会退化成单线程。

### 16.5 `config.yaml` 不存在

确保从构建目录运行，并检查：

```bash
pwd
ls -l config.yaml
```

也可以显式传入：

```bash
./demo_MEEF_Optimizer_init ../config.yaml
```

### 16.6 YAML 指定的 target 不存在

如果：

```yaml
pattern_name: "工字型"
```

则必须存在：

```text
build-release/target_pattern/工字型.bmp
```

新增 target 图片后，应重新执行 CMake 配置，让 `file(COPY ...)` 再次同步资源。

### 16.7 LSM mask 不存在

YAML 中：

```yaml
lsm_mask_path: "assets/lsm_mask/ls_image工字型.txt"
```

应对应项目根目录中的：

```text
assets/lsm_mask/ls_image工字型.txt
```

### 16.8 编译成功但 VS Code 标红

检查真实编译数据库：

```bash
rg 'eigen3|fopenmp' build-release/compile_commands.json
```

如果命令行构建成功，则优先检查 VS Code CMake Tools/clangd 是否读取了正确的 `compile_commands.json`，而不是修改能够正常编译的源码。

### 16.9 `BDCSVD is deprecated`

这是 Eigen 接口弃用警告，不是构建失败。如果最后显示：

```text
[100%] Built target demo_MEEF_Optimizer_init
```

说明可执行程序已经生成。警告应后续修复，但不应误判为 error。

---

## 17. 如何添加一个新的 demo

假设新增：

```text
demo/demo_new.cpp
```

在 `CMakeLists.txt` 中添加：

```cmake
add_executable(demo_new demo/demo_new.cpp)
target_link_libraries(demo_new litho_core)
```

然后重新配置：

```bash
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release
```

只构建新 demo：

```bash
cmake --build build-release \
  --target demo_new \
  --parallel 4
```

运行：

```bash
cd build-release
./demo_new
```

`target_link_libraries(demo_new litho_core)` 会让新 demo 自动获得：

- 项目头文件目录
- Eigen
- FFTW
- OpenCV
- yaml-cpp
- OpenMP 编译和链接选项

---

## 18. 最短可执行流程

已经安装依赖后，从项目根目录开始：

```bash
# 1. 配置
cmake -S . -B build-release \
  -DCMAKE_BUILD_TYPE=Release

# 2. 只编译 MEEF demo
cmake --build build-release \
  --target demo_MEEF_Optimizer_init \
  --parallel 4

# 3. 进入构建目录
cd build-release

# 4. 运行；运行模式和参数来自 config.yaml
OMP_NUM_THREADS=4 ./demo_MEEF_Optimizer_init
```

判断成功的三个关键证据：

```text
[100%] Built target demo_MEEF_Optimizer_init
MEEF_Optimizer 测试完成。
result/MEEF_result/... 中生成结果和 console_output.log
```

