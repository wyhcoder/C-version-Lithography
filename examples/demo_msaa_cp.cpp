// demo_msaa_cp.cpp
// 用 MSAA 光栅化 cp_history/cp_iter_*.txt 里的多边形（工字型等）。
//
// 用法：
//   ./demo_msaa_cp [cp_txt_path] [canvas_size] [msaa_level] [mode]
// 默认：
//   cp_txt_path = /Users/wyh/Desktop/学校/Litho_Simulation_cpu/OPCproject/
//                 MEEF_pipeline/BS/工字型_k=7_sym=none_srafArc=5.0_.../cp_iter_020.txt
//   canvas_size = 257
//   msaa_level  = 16
//   mode        = bs   （"direct" 直接当多边形；"bs" 用 Catmull-Rom B 样条插值后再 MSAA）

#include "msaa.h"
#include "parametric.h"

#include <Eigen/Dense>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace litho;

// ── 解析 cp_history/cp_iter_*.txt ──────────────────────────────────────────
// 文件格式：
//   # CP positions at iteration N
//   # num_contours = K
//   # contour 0, num_points = M0
//   y0 x0
//   y1 x1
//   ...
//   # contour 1, num_points = M1
//   ...
// 每行点按 msaa.h 约定是 (y, x)。
static Polygons load_cp_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("无法打开 cp 文件: " + path);

    Polygons polys;
    std::vector<Eigen::Vector2d> tmp;
    std::string line;

    auto flush = [&]() {
        if (tmp.empty()) return;
        Polygon p(tmp.size(), 2);
        for (size_t i = 0; i < tmp.size(); ++i) {
            p(i, 0) = tmp[i](0);
            p(i, 1) = tmp[i](1);
        }
        polys.push_back(std::move(p));
        tmp.clear();
    };

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            // 遇到任何包含 "contour" 的注释行都视为新多边形开始 → 先 flush 上一个
            // 兼容两种格式：
            //   "# contour 0, num_points = 55"     （cp_iter_*.txt）
            //   "# main contour 0" / "# sraf block 0"  （main_cps.txt / sraf_cps.txt）
            if (line.find("contour") != std::string::npos ||
                line.find("block")   != std::string::npos) {
                flush();
            }
            continue;
        }
        std::istringstream iss(line);
        double y, x;
        if (iss >> y >> x) {
            tmp.emplace_back(y, x);
        }
    }
    flush();
    return polys;
}

// ── 保存 [H x W] 矩阵到 txt（Python np.loadtxt(comments='#') 可直接读）────────
static void save_mat(const Eigen::MatrixXd& M, const std::string& fname) {
    std::ofstream f(fname);
    f << std::fixed << std::setprecision(6);
    f << "# rows=" << M.rows() << " cols=" << M.cols() << "\n";
    for (int r = 0; r < M.rows(); ++r) {
        for (int c = 0; c < M.cols(); ++c) {
            f << M(r, c);
            if (c < M.cols() - 1) f << " ";
        }
        f << "\n";
    }
    std::cout << "  saved: " << fname << "\n";
}

// ── 导出多边形顶点（方便 Python 叠加边界画图）──────────────────────────────
static void save_polygons(const Polygons& polys, const std::string& fname) {
    std::ofstream f(fname);
    f << std::fixed << std::setprecision(6);
    f << "# num_contours=" << polys.size() << "\n";
    for (size_t i = 0; i < polys.size(); ++i) {
        f << "# contour " << i << ", num_points=" << polys[i].rows() << "\n";
        for (int r = 0; r < polys[i].rows(); ++r) {
            f << polys[i](r, 0) << " " << polys[i](r, 1) << "\n";
        }
    }
    std::cout << "  saved: " << fname << "\n";
}

int main(int argc, char** argv) {
    const std::string default_cp =
        "/Users/wyh/Desktop/学校/Litho_Simulation_cpu/OPCproject/MEEF_pipeline/"
        "BS/工字型_k=7_sym=none_srafArc=5.0_BS_pipeline_test_bisector_gradient/"
        "curves/cp_history/cp_iter_020.txt";

    std::string cp_path = (argc > 1) ? argv[1] : default_cp;
    int N           = (argc > 2) ? std::stoi(argv[2]) : 257;
    int msaa_level  = (argc > 3) ? std::stoi(argv[3]) : 16;
    std::string mode = (argc > 4) ? argv[4] : "bs";   // "direct" 或 "bs"

    std::cout << "========== MSAA cp 渲染 demo ==========\n";
    std::cout << "cp file  : " << cp_path << "\n";
    std::cout << "canvas   : " << N << " x " << N << "\n";
    std::cout << "msaa     : " << msaa_level << "x\n";
    std::cout << "mode     : " << mode
              << (mode == "bs" ? "  (Catmull-Rom B 样条插值 → MSAA)" :
                  mode == "direct" ? "  (控制点直接当多边形顶点)" : "  (未知)")
              << "\n\n";

    // 1. 加载多边形
    Polygons polys;
    try {
        polys = load_cp_file(cp_path);
    } catch (const std::exception& e) {
        std::cerr << "[error] " << e.what() << "\n";
        return 1;
    }
    std::cout << "loaded " << polys.size() << " contour(s)\n";
    for (size_t i = 0; i < polys.size(); ++i) {
        std::cout << "  contour " << i << ": " << polys[i].rows() << " points\n";
    }
    if (polys.empty()) {
        std::cerr << "[error] 未解析到任何多边形\n";
        return 1;
    }

    // 2. 光栅化
    Eigen::MatrixXd tmpl = Eigen::MatrixXd::Zero(N, N);
    AntiAliasRenderer renderer(msaa_level);

    // 拿到用于光栅化的"实际多边形"——direct 模式用原始控制点；bs 模式先插值成曲线点
    Polygons polys_for_raster;
    if (mode == "bs") {
        ParametricDemo pd_bs("BS", tmpl, msaa_level);
        polys_for_raster = pd_bs.get_curve_points(polys, 200);
        std::cout << "  BS 插值后曲线点数: ";
        for (size_t i = 0; i < polys_for_raster.size(); ++i)
            std::cout << polys_for_raster[i].rows() << " ";
        std::cout << "\n";
    } else {
        polys_for_raster = polys;
    }

    auto t0 = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd mask_gray = renderer.MSAA(polys_for_raster, tmpl, "gray");
    auto t1 = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd mask_bin  = renderer.MSAA(polys_for_raster, tmpl, "binary");
    auto t2 = std::chrono::high_resolution_clock::now();

    double ms_gray = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_bin  = std::chrono::duration<double, std::milli>(t2 - t1).count();

    std::cout << "\n--- 渲染结果 ---\n";
    std::cout << "  gray   : max=" << mask_gray.maxCoeff()
              << "  nonzero(>0.01)=" << (mask_gray.array() > 0.01).count()
              << "  time=" << ms_gray << " ms\n";
    std::cout << "  binary : sum=" << mask_bin.sum()
              << "  (=像素数 " << (mask_bin.array() > 0.5).count() << ")"
              << "  time=" << ms_bin << " ms\n";

    // 3. 保存
    std::cout << "\n--- 保存 ---\n";
    save_mat(mask_gray, "msaa_cp_gray.txt");
    save_mat(mask_bin,  "msaa_cp_binary.txt");
    // 同时保存两种多边形：原始控制点 + 实际光栅化用的曲线点
    save_polygons(polys,             "msaa_cp_polygon.txt");
    save_polygons(polys_for_raster,  "msaa_cp_raster_polygon.txt");

    std::cout << "\n========== done ==========\n";
    return 0;
}
