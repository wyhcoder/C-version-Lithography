cmake_minimum_required(VERSION 3.16)
project(LithoSim VERSION 1.0.0 LANGUAGES CXX)

# 供 VS Code C/C++ IntelliSense 读取与实际构建完全一致的头文件和编译选项。
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Apple Clang 不自带 OpenMP，且 FindOpenMP 不会自动搜索 Homebrew keg。
# find_package(OpenMP ... HINTS path) 对 FindOpenMP 模块不起作用，
# 因此先通过 Homebrew 获取 libomp 根目录，再传给 FindOpenMP。
if(APPLE AND NOT OpenMP_ROOT)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE HOMEBREW_LIBOMP_ROOT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE HOMEBREW_LIBOMP_RESULT
        ERROR_QUIET
    )
    if(HOMEBREW_LIBOMP_RESULT EQUAL 0
       AND EXISTS "${HOMEBREW_LIBOMP_ROOT}/include/omp.h")
        set(OpenMP_ROOT "${HOMEBREW_LIBOMP_ROOT}" CACHE PATH
            "Root directory of Homebrew libomp")
        message(STATUS "Using Homebrew libomp: ${OpenMP_ROOT}")
    endif()
endif()

find_package(OpenMP COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP enabled")
else()
    message(WARNING "OpenMP not found: parallel benchmarks will not be available")
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED HINTS /opt/homebrew/lib/cmake/opencv4)
find_package(yaml-cpp REQUIRED HINTS /opt/homebrew/lib/cmake/yaml-cpp)

find_path(FFTW3_INCLUDE_DIR fftw3.h HINTS /opt/homebrew/include)
find_library(FFTW3_LIBRARY fftw3 HINTS /opt/homebrew/lib)
if(FFTW3_INCLUDE_DIR AND FFTW3_LIBRARY)
    set(FFTW3_FOUND TRUE)
endif()
if(NOT FFTW3_FOUND)
    message(FATAL_ERROR "FFTW3 not found")
endif()

add_library(litho_core STATIC
    source/litho_model/grid.cpp
    source/litho_model/mask.cpp
    source/litho_model/zernike.cpp
    source/litho_model/pupil.cpp
    source/litho_model/source.cpp
    source/litho_model/fft.cpp
    source/litho_model/litho_prepare.cpp
    source/litho_model/imaging.cpp
    source/utils/ep_select.cpp
    source/utils/msaa.cpp
    source/utils/parametric.cpp
    source/utils/meef_pipeline.cpp
    source/utils/gradient.cpp
    source/utils/simulation_parameters.cpp
    source/litho_model/lithography_simulator.cpp
    source/optimizer/CTM_Optimizer.cpp
    source/utils/loss.cpp
    source/utils/save_txt.cpp
    source/utils/imbinarize.cpp
    source/utils/level_set_utils.cpp
    source/optimizer/LSM_Optimizer.cpp
    source/utils/pv_band_computer.cpp
    source/optimizer/MEEF_Optimizer.cpp
)

target_include_directories(litho_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include/litho_model
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/include/optimizer
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
    ${FFTW3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(litho_core PUBLIC Eigen3::Eigen ${FFTW3_LIBRARY} ${OpenCV_LIBS} yaml-cpp::yaml-cpp)
if(OpenMP_CXX_FOUND)
    target_link_libraries(litho_core PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(demo_bs examples/demo_bs_render.cpp)
target_link_libraries(demo_bs litho_core)
add_executable(demo_ep examples/demo_ep_select.cpp)
target_link_libraries(demo_ep litho_core)
add_executable(demo_meef examples/demo_meef.cpp)
target_link_libraries(demo_meef litho_core)
add_executable(demo_socs_imaging examples/demo_socs.cpp)
target_link_libraries(demo_socs_imaging litho_core)
add_executable(demo_CTM demo/demo_CTM.cpp)
target_link_libraries(demo_CTM litho_core)
add_executable(demo_CTM_parallel_benchmark demo/demo_CTM_parallel_benchmark.cpp)
target_link_libraries(demo_CTM_parallel_benchmark litho_core)
add_executable(demo_LSM demo/demo_LSM.cpp)
target_link_libraries(demo_LSM litho_core)
add_executable(demo_MEEF_Optimizer_init demo/demo_MEEF_Optimizer_init.cpp)
target_link_libraries(demo_MEEF_Optimizer_init litho_core)
add_executable(demo_epe demo/demo_epe.cpp)
target_link_libraries(demo_epe litho_core)
add_executable(demo_pvband demo/demo_pvband.cpp)
target_link_libraries(demo_pvband litho_core)
add_executable(demo_fft_shift demo/demo_fftshift.cpp)
target_link_libraries(demo_fft_shift litho_core)
add_executable(demo_msaa_cp examples/demo_msaa_cp.cpp)
target_link_libraries(demo_msaa_cp litho_core)

# 把 config.yaml 复制到 build 目录，方便 ./demo_xxx 直接以相对路径读取
configure_file(
    ${CMAKE_SOURCE_DIR}/config.yaml
    ${CMAKE_BINARY_DIR}/config.yaml
    COPYONLY
)

# 把 assets/target_pattern/ 目录同步到 build 目录（yaml 里 image_name 相对该目录）
# 运行时程序在 build-*/ 里，通过相对路径 target_pattern/xxx.bmp 找图
file(COPY ${CMAKE_SOURCE_DIR}/assets/target_pattern DESTINATION ${CMAKE_BINARY_DIR})
