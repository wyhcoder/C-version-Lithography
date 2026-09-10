# ============================================================================
# 1. 项目基本信息
# ============================================================================
# 要求使用 CMake 3.16 或更高版本。版本过低时在配置阶段直接停止，避免使用
# 不兼容的 CMake 行为生成错误构建文件。
cmake_minimum_required(VERSION 3.16)

# 定义项目名称、项目版本和使用的编程语言。
# LANGUAGES CXX 表示这里只需要初始化 C++ 编译器，不额外检测 C/Fortran。
project(LithoSim VERSION 1.0.0 LANGUAGES CXX)

# 导出 build-*/compile_commands.json。
# 该文件保存每个 .cpp 的真实编译命令，供 VS Code/clangd/静态分析工具读取，
# 从而让 IDE 使用与实际编译完全一致的 include 路径、宏和编译选项。
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# 2. OpenMP 并行环境
# ============================================================================
# Apple Clang 不自带 OpenMP，且 FindOpenMP 不会自动搜索 Homebrew keg。
# `find_package(OpenMP ... HINTS path)` 对 FindOpenMP 模块不起作用，因此在
# macOS 上先调用 `brew --prefix libomp` 获取安装根目录，再通过缓存变量
# OpenMP_ROOT 告诉 FindOpenMP 去哪里查找 omp.h 和 libomp。
#
# `APPLE` 保证这段 Homebrew 特殊处理只在 macOS 执行；Linux/Windows 会跳过。
# 如果用户在命令行显式传入了 -DOpenMP_ROOT=...，则尊重用户设置。
if(APPLE AND NOT OpenMP_ROOT)
    execute_process(
        # 配置阶段执行外部命令，查询 Homebrew 中 libomp 的安装位置。
        COMMAND brew --prefix libomp
        # 将命令的标准输出保存到 HOMEBREW_LIBOMP_ROOT。
        OUTPUT_VARIABLE HOMEBREW_LIBOMP_ROOT
        # 删除输出末尾的换行符，否则路径检查可能失败。
        OUTPUT_STRIP_TRAILING_WHITESPACE
        # 保存命令退出状态：0 表示执行成功。
        RESULT_VARIABLE HOMEBREW_LIBOMP_RESULT
        # brew 不存在时不立即打印噪声，后面由 FindOpenMP 给出统一提示。
        ERROR_QUIET
    )

    # brew 查询成功且安装目录中确实存在 omp.h 时，才写入 OpenMP_ROOT。
    if(HOMEBREW_LIBOMP_RESULT EQUAL 0
       AND EXISTS "${HOMEBREW_LIBOMP_ROOT}/include/omp.h")
        # CACHE PATH 让该路径写入 CMakeCache.txt，后续重新配置可以复用。
        set(OpenMP_ROOT "${HOMEBREW_LIBOMP_ROOT}" CACHE PATH
            "Root directory of Homebrew libomp")
        message(STATUS "Using Homebrew libomp: ${OpenMP_ROOT}")
    endif()
endif()

# 查找 C++ OpenMP 编译选项和运行库。OpenMP 在本项目中是可选依赖：
# 找到时启用多线程；找不到时 #pragma omp 会被忽略，程序仍可单线程构建。
find_package(OpenMP COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP enabled")
else()
    message(WARNING "OpenMP not found: parallel benchmarks will not be available")
endif()

# ============================================================================
# 3. C++ 语言标准
# ============================================================================
# 所有 C++ target 使用 C++20；如果编译器不支持 C++20，配置/编译会失败。
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 关闭 gnu++20 等编译器扩展，要求使用标准的 -std=c++20，提高跨平台性。
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================================
# 4. 第三方依赖
# ============================================================================
# REQUIRED 表示依赖是必需的：找不到时 CMake 在配置阶段直接报错。
# HINTS 中的 /opt/homebrew 路径用于帮助 Apple Silicon macOS 查找 Homebrew
# 安装的软件；CMake 仍会同时检查系统默认路径。
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED HINTS /opt/homebrew/lib/cmake/opencv4)
find_package(yaml-cpp REQUIRED HINTS /opt/homebrew/lib/cmake/yaml-cpp)

# 优先使用系统安装的 libcmaes；找不到时下载固定提交，保证构建可复现。
find_package(libcmaes CONFIG QUIET)
if(NOT TARGET libcmaes::cmaes)
    include(FetchContent)
    set(LIBCMAES_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(LIBCMAES_BUILD_PYTHON OFF CACHE BOOL "" FORCE)
    set(LIBCMAES_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LIBCMAES_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LIBCMAES_USE_OPENMP OFF CACHE BOOL "" FORCE)
    set(LIBCMAES_ENABLE_SURROG OFF CACHE BOOL "" FORCE)
    FetchContent_Declare(
        libcmaes
        GIT_REPOSITORY https://github.com/CMA-ES/libcmaes.git
        GIT_TAG 6a53cd562c85dc6df3b8df317be3ecb415240c2d
    )
    FetchContent_MakeAvailable(libcmaes)
endif()

# 当前环境没有统一使用 FFTW3 的 CMake imported target，因此分别查找：
#   FFTW3_INCLUDE_DIR：包含 fftw3.h 的头文件目录；
#   FFTW3_LIBRARY：实际链接的 fftw3 库文件。
find_path(FFTW3_INCLUDE_DIR fftw3.h HINTS /opt/homebrew/include)
find_library(FFTW3_LIBRARY fftw3 HINTS /opt/homebrew/lib)

# 只有头文件和库文件都找到时，才认为 FFTW3 完整可用。
if(FFTW3_INCLUDE_DIR AND FFTW3_LIBRARY)
    set(FFTW3_FOUND TRUE)
endif()

# FFT 是光刻成像的核心功能，因此 FFTW3 缺失时不能降级，直接终止配置。
if(NOT FFTW3_FOUND)
    message(FATAL_ERROR "FFTW3 not found")
endif()

# ============================================================================
# 5. 核心静态库 litho_core
# ============================================================================
# 将光刻模型、数值工具和优化器源码统一编译为 liblitho_core.a。
# 每个 demo 只需要链接 litho_core，不必重复列出全部实现文件；修改某个源码后，
# 构建系统也只会增量编译受影响的目标文件。
add_library(litho_core STATIC
    # 光刻模型与成像
    source/litho_model/grid.cpp
    source/litho_model/mask.cpp
    source/litho_model/zernike.cpp
    source/litho_model/pupil.cpp
    source/litho_model/source.cpp
    source/litho_model/fft.cpp
    source/litho_model/litho_prepare.cpp
    source/litho_model/imaging.cpp

    # 通用算法与数据处理工具
    source/utils/ep_select.cpp
    source/utils/msaa.cpp
    source/utils/parametric.cpp
    source/utils/meef_pipeline.cpp
    source/utils/gradient.cpp
    source/utils/simulation_parameters.cpp
    source/utils/sraf_geometry.cpp
    source/utils/sraf_curve.cpp

    # 仿真器及优化算法
    source/litho_model/lithography_simulator.cpp
    source/optimizer/CTM_Optimizer.cpp
    source/utils/loss.cpp
    source/utils/save_txt.cpp
    source/utils/imbinarize.cpp
    source/utils/level_set_utils.cpp
    source/optimizer/LSM_Optimizer.cpp
    source/utils/pv_band_computer.cpp
    source/optimizer/MEEF_Optimizer.cpp
    source/optimizer/SRAF_Optimizer.cpp
)

# 为 litho_core 添加头文件搜索路径。
# PUBLIC 表示：
#   1. 编译 litho_core 自己时使用这些目录；
#   2. 链接 litho_core 的 demo 也自动继承这些目录。
# CMAKE_CURRENT_SOURCE_DIR 是当前 CMakeLists.txt 所在的项目源码根目录。
target_include_directories(litho_core PUBLIC
    # 项目自己的公共头文件
    ${CMAKE_CURRENT_SOURCE_DIR}/include/litho_model
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/include/optimizer
    ${CMAKE_CURRENT_SOURCE_DIR}/include

    # 第三方依赖头文件
    ${EIGEN3_INCLUDE_DIR}
    ${FFTW3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

# 链接核心库所依赖的第三方库。
# Eigen3::Eigen 和 yaml-cpp::yaml-cpp 是 imported target，会携带各自的 include
# 和编译属性；FFTW3_LIBRARY/OpenCV_LIBS 是查找阶段得到的库变量。
# PUBLIC 使链接 litho_core 的 demo 自动继承必要的传递依赖。
target_link_libraries(litho_core PUBLIC
    Eigen3::Eigen
    ${FFTW3_LIBRARY}
    ${OpenCV_LIBS}
    yaml-cpp::yaml-cpp
    libcmaes::cmaes
)

# OpenMP::OpenMP_CXX 同时携带编译选项和链接库，例如 -fopenmp/libomp。
# 只有 FindOpenMP 成功时才链接，因此没有 OpenMP 的环境仍能单线程构建。
if(OpenMP_CXX_FOUND)
    target_link_libraries(litho_core PUBLIC OpenMP::OpenMP_CXX)
endif()

# ============================================================================
# 6. Demo 可执行程序
# ============================================================================
# 每个 demo 都由一个入口 .cpp 创建可执行 target，再链接公共核心库：
#   add_executable(target_name entry.cpp)
#   target_link_libraries(target_name litho_core)
#
# 构建单个 demo 的通用命令：
#   cmake --build build-release --target <target_name> --parallel

# B-spline/参数化曲线渲染示例。
add_executable(demo_bs examples/demo_bs_render.cpp)
target_link_libraries(demo_bs litho_core)

# Evaluation Point 选择示例。
add_executable(demo_ep examples/demo_ep_select.cpp)
target_link_libraries(demo_ep litho_core)

# 基础 MEEF 示例。
add_executable(demo_meef examples/demo_meef.cpp)
target_link_libraries(demo_meef litho_core)

# SOCS 成像示例。
add_executable(demo_socs_imaging examples/demo_socs.cpp)
target_link_libraries(demo_socs_imaging litho_core)

# CTM 优化及其并行性能测试。
add_executable(demo_CTM demo/demo_CTM.cpp)
target_link_libraries(demo_CTM litho_core)
add_executable(demo_CTM_parallel_benchmark demo/demo_CTM_parallel_benchmark.cpp)
target_link_libraries(demo_CTM_parallel_benchmark litho_core)

# Level Set Method 优化示例。
add_executable(demo_LSM demo/demo_LSM.cpp)
target_link_libraries(demo_LSM litho_core)

# SRAF 几何前处理诊断：真实 LSM mask -> SRAF 骨架/路径/控制点统计。
add_executable(demo_sraf_geometry demo/demo_sraf_geometry.cpp)
target_link_libraries(demo_sraf_geometry litho_core)

# SRAF 优化器初始化：读取 LSM mask，提取中心线并保存 (x,y) 控制点。
add_executable(demo_SRAF_Optimizer_init demo/demo_SRAF_Optimizer_init.cpp)
target_link_libraries(demo_SRAF_Optimizer_init litho_core)

# 从 YAML 初始化、构建 MEEF 矩阵或执行完整 MEEF 优化的主 demo。
add_executable(demo_MEEF_Optimizer_init demo/demo_MEEF_Optimizer_init.cpp)
target_link_libraries(demo_MEEF_Optimizer_init litho_core)

# EPE、PV Band、FFT shift 和 MSAA 控制点的独立验证程序。
add_executable(demo_epe demo/demo_epe.cpp)
target_link_libraries(demo_epe litho_core)
add_executable(demo_pvband demo/demo_pvband.cpp)
target_link_libraries(demo_pvband litho_core)
add_executable(demo_fft_shift demo/demo_fftshift.cpp)
target_link_libraries(demo_fft_shift litho_core)
add_executable(demo_msaa_cp examples/demo_msaa_cp.cpp)
target_link_libraries(demo_msaa_cp litho_core)

# SRAF 几何前处理和曲线半宽重建的独立回归测试。
include(CTest)
if(BUILD_TESTING)
    add_executable(test_sraf_geometry tests/test_sraf_geometry.cpp)
    target_link_libraries(test_sraf_geometry litho_core)
    add_test(NAME sraf_geometry COMMAND test_sraf_geometry)

    add_executable(test_sraf_curve tests/test_sraf_curve.cpp)
    target_link_libraries(test_sraf_curve litho_core)
    add_test(NAME sraf_curve COMMAND test_sraf_curve)

    add_executable(test_parametric tests/test_parametric.cpp)
    target_link_libraries(test_parametric litho_core)
    add_test(NAME parametric_b_spline COMMAND test_parametric)

    add_executable(test_dirac_rasterization tests/test_dirac_rasterization.cpp)
    target_link_libraries(test_dirac_rasterization litho_core)
    add_test(NAME dirac_rasterization COMMAND test_dirac_rasterization)

    add_executable(test_level_set_godunov tests/test_level_set_godunov.cpp)
    target_link_libraries(test_level_set_godunov litho_core)
    add_test(NAME level_set_godunov COMMAND test_level_set_godunov)
endif()

# ============================================================================
# 7. 运行时配置和资源
# ============================================================================
# 配置 CMake 时，把源码根目录的 config.yaml 原样复制到构建目录。
# COPYONLY 表示只复制，不展开 ${VAR} 或 @VAR@ 占位符。
# 因此从 build-release/ 运行 demo 时，可以默认读取 ./config.yaml。
configure_file(
    ${CMAKE_SOURCE_DIR}/config.yaml
    ${CMAKE_BINARY_DIR}/config.yaml
    COPYONLY
)

# 在 CMake 配置阶段，把 assets/target_pattern/ 复制为
# build-*/target_pattern/。demo 根据 YAML 中的 meef.pattern_name 组合出：
#   <构建目录>/target_pattern/<pattern_name>.bmp
#
# 注意：file(COPY ...) 发生在“配置阶段”，新增图片后应重新执行 cmake 配置，
# 不能只运行 make/ninja，否则构建目录中可能没有新资源。
file(COPY ${CMAKE_SOURCE_DIR}/assets/target_pattern DESTINATION ${CMAKE_BINARY_DIR})
