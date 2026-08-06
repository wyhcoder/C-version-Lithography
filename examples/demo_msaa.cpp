#include "msaa.h"
#include "parametric.h"

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace litho;

static void save_mat(const Eigen::MatrixXd& M, const std::string& fname) {
    std::ofstream f(fname);
    f << std::fixed << std::setprecision(6);
    f << "# rows=" << M.rows() << " cols=" << M.cols() << "\n";
    for (int r = 0; r < M.rows(); ++r) {
        for (int c = 0; c < M.cols(); ++c) {
            f << M(r,c);
            if (c < M.cols()-1) f << " ";
        }
        f << "\n";
    }
    std::cout << "  saved: " << fname << "\n";
}

int main() {
    std::cout << "========== MSAA / Parametric Demo ==========\n\n";

    const int N = 64;
    Eigen::MatrixXd tmpl = Eigen::MatrixXd::Zero(N, N);

    // ── 1. 直接用 MSAA 光栅化一个三角形多边形 ──────────────────────────
    std::cout << "--- MSAA 光栅化三角形 ---\n";
    AntiAliasRenderer renderer(16);

    Polygon tri(3, 2);
    tri << 10.0, 20.0,
           50.0, 10.0,
           40.0, 55.0;

    Eigen::MatrixXd mask_tri = renderer.MSAA({tri}, tmpl, "gray");
    std::cout << "  tri mask max = " << mask_tri.maxCoeff()
              << ", non-zero = " << (mask_tri.array() > 0.01).count() << "\n";
    save_mat(mask_tri, "msaa_triangle.txt");

    // ── 2. OA 模式（折线直接渲染）────────────────────────────────────────
    std::cout << "\n--- ParametricDemo OA 模式 ---\n";
    ParametricDemo pd_oa("OA", tmpl, 16);

    Polygon rect(4, 2);
    rect << 20.0, 15.0,
            20.0, 48.0,
            44.0, 48.0,
            44.0, 15.0;

    Eigen::MatrixXd mask_oa = pd_oa.render_curve({rect});
    std::cout << "  OA mask max = " << mask_oa.maxCoeff()
              << ", non-zero = " << (mask_oa.array() > 0.01).count() << "\n";
    save_mat(mask_oa, "parametric_oa.txt");

    // ── 3. BZ 模式（Bezier 曲线渲染）────────────────────────────────────
    std::cout << "\n--- ParametricDemo BZ 模式 ---\n";
    ParametricDemo pd_bz("BZ", tmpl, 16);

    // 六边形控制点
    Polygon hex(6, 2);
    hex << 12.0, 32.0,
           22.0, 10.0,
           42.0, 10.0,
           52.0, 32.0,
           42.0, 54.0,
           22.0, 54.0;

    auto bz_pts = pd_bz.get_curve_points({hex}, 120);
    std::cout << "  BZ 曲线点数: " << bz_pts[0].rows() << "\n";
    Eigen::MatrixXd mask_bz = pd_bz.render_curve({hex});
    std::cout << "  BZ mask max = " << mask_bz.maxCoeff()
              << ", non-zero = " << (mask_bz.array() > 0.01).count() << "\n";
    save_mat(mask_bz, "parametric_bz.txt");

    // ── 4. BS 模式（B 样条渲染）──────────────────────────────────────────
    std::cout << "\n--- ParametricDemo BS 模式 ---\n";
    ParametricDemo pd_bs("BS", tmpl, 16);

    Eigen::MatrixXd mask_bs = pd_bs.render_curve({rect});
    std::cout << "  BS mask max = " << mask_bs.maxCoeff()
              << ", non-zero = " << (mask_bs.array() > 0.01).count() << "\n";
    save_mat(mask_bs, "parametric_bs.txt");

    std::cout << "\n========== Demo 完成 ==========\n";
    return 0;
}
