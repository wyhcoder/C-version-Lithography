#include "grid.h"
#include "mask.h"
#include "pupil.h"
#include "source.h"
#include "litho_prepare.h"
#include "imaging.h"

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <complex>
#include <chrono>

using namespace litho;

int main() {
    std::cout << "========== 1. Grid 测试 ==========" << std::endl;

    Grid grid(257, 4.0);   // 8×8 网格，像素 5nm

    std::cout << "grid.size() = " << grid.size()
              << ", pixel_size = " << grid.pixel_size() << " nm"
              << std::endl;

    const auto& coords = grid.grid_coords();

    // 空间坐标：中心应为 0
    std::cout << "\nx(4,4) = " << coords.x(4, 4)   // 中心点
              << " (期望 0)" << std::endl;
    std::cout << "y(0,7) = " << coords.y(0, 7)     // 某个角
              << std::endl;

    // 频率轴
    std::cout << "\nFreq axis (1D): ";
    for (int i = 0; i < 8; ++i)
        std::cout << coords.Fx_1d(i) << " ";
    std::cout << std::endl;

    // 2D 频率网格
    std::cout << "Fx_2d(0,4) = " << coords.Fx_2d(0, 4)
              << ", Fy_2d(4,0) = " << coords.Fy_2d(4, 0)
              << std::endl;


    std::cout << "\n========== 2. Mask 矩阵构造测试 ==========" << std::endl;

    // 构造一个 5x5 的 L 形掩模，居中到 16x16
    Eigen::MatrixXd pattern(5, 5);
    pattern << 1, 1, 1, 1, 1,
               1, 0, 0, 0, 0,
               1, 0, 0, 0, 0,
               1, 0, 0, 0, 0,
               1, 0, 0, 0, 0;

    Mask mask(pattern);   // 直接用矩阵构造，无需图片文件

    std::cout << "mask size = " << mask.rows() << "x" << mask.cols() << std::endl;

    // 打印掩模（稀疏文本）
    std::cout << "\nmask.data():" << std::endl;
    for (int i = 0; i < mask.rows(); ++i) {
        std::cout << "  ";
        for (int j = 0; j < mask.cols(); ++j) {
            std::cout << (mask.data()(i, j) > 0.5 ? "█" : " ");
        }
        std::cout << std::endl;
    }


    std::cout << "\n========== 3. Mask 文件加载测试 ==========" << std::endl;

    auto opt_mask = Mask::from_file("examples/pattern_basic.png", 32);
    if (opt_mask) {
        std::cout << "Loaded from file, size = "
                  << opt_mask->rows() << "x" << opt_mask->cols() << std::endl;
        std::cout << "Has data: " << (opt_mask->data().maxCoeff() > 0 ? "yes" : "no")
                  << std::endl;
    } else {
        std::cout << "No image file found (expected if ./examples/pattern_basic.png 不存在)"
                  << std::endl;
    }


    std::cout << "\n========== 4. Pupil 测试 ==========" << std::endl;

    // 用 Grid 生成频率网格
    Grid pg(257, 4.0);
    const auto& pc = pg.grid_coords();  

    // 设置光刻参数
    Params params;
    params.NA         = 1.35;
    params.wavelength = 193.0;   // ArF 波长 [nm]
    params.n          = 1.44;    // 水浸液折射率
    params.defocus_nm = 0.0;     // 无离焦
    params.zernike_coeffs = {
        {4,  0.05},   // 离焦 Z4（用 Zernike 方式）
        {9, -0.02},   // Trefoil Y
    };

    Pupil pupil(params, pc.Fx_2d, pc.Fy_2d);

    const auto& amp   = pupil.get_amplitude();
    const auto& phase = pupil.get_phase();
    const auto& P     = pupil.get_pupil();

    // 基本尺寸验证
    std::cout << "Pupil matrix size: " << P.rows() << "x" << P.cols() << std::endl;

    // amplitude 应该是 0/1 圆形光瞳：统计圆内像素数
    int inside = amp.array().cast<int>().sum();
    std::cout << "Pixels inside pupil: " << inside
              << " / " << (P.rows() * P.cols())
              << " (ratio = " << (100.0 * inside / (P.rows() * P.cols())) << "%)" << std::endl;

    // 圆内 amplitude 应为 1
    double amp_max = amp.maxCoeff();
    double amp_min = amp.minCoeff();
    std::cout << "Amplitude range: [" << amp_min << ", " << amp_max << "]"
              << (amp_max == 1.0 && amp_min == 0.0 ? "  OK" : "  FAIL") << std::endl;

    // 圆内 |P| 应为 1
    Eigen::MatrixXd P_abs = P.cwiseAbs();
    double p_max_inside = (P_abs.array() * amp.array()).maxCoeff();
    std::cout << "|P| max inside pupil: " << p_max_inside
              << (std::abs(p_max_inside - 1.0) < 1e-9 ? "  OK" : "  FAIL") << std::endl;

    // 打印中心 7x7 的 amplitude（ASCII 可视化）
    int N   = P.rows();
    int mid = N / 2;
    int hw  = std::min(7, (N - 1) / 2);    // 安全半宽
    std::cout << "\nAmplitude (center " << (2*hw+1) << "x" << (2*hw+1) << "):" << std::endl;
    for (int i = mid - hw; i <= mid + hw; ++i) {
        std::cout << "  ";
        for (int j = mid - hw; j <= mid + hw; ++j)
            std::cout << (amp(i, j) > 0.5 ? "█" : "·");
        std::cout << std::endl;
    }

    // 打印中心点的 phase 值
    std::cout << "\nPhase at center (0,0): " << phase(mid, mid) << " rad" << std::endl;
    std::cout << "P at center: " << P(mid, mid).real()
              << " + " << P(mid, mid).imag() << "i" << std::endl;


    std::cout << "\n========== 5. Source 测试 ==========" << std::endl;

    // 用一个 128×128 的频率网格
    Grid sg(257, 4.0);
    const auto& sc = sg.grid_coords();

    SourceParams sp;
    sp.wavelength_nm = 193.0;
    sp.NA            = 1.35;
    sp.n             = 1.44;
    sp.sigma_in      = 0.6;   // 环形内径
    sp.sigma_out     = 0.9;   // 环形外径
    sp.upsample      = 10;
    sp.smoothing     = 0.01;

    Source src(sp);
    const SourceMap& sm = src.compute_source_map(sc.Fx_1d, sc.Fy_1d);

    // ── 基本验证 ──────────────────────────────────────────────────────────
    std::cout << "source_map size   : "
              << sm.source_map.rows() << "x" << sm.source_map.cols() << std::endl;
    std::cout << "fx_norm range     : ["
              << sm.fx_norm.minCoeff() << ", " << sm.fx_norm.maxCoeff() << "]" << std::endl;
    std::cout << "source_map max    : " << sm.source_map.maxCoeff() << std::endl;
    std::cout << "weight_map sum    : " << sm.source_weight_map.sum()
              << "  (期望 ≈ 1.0)" << std::endl;
    std::cout << "weight_map max    : " << sm.source_weight_map.maxCoeff() << std::endl;

    double wsum = sm.source_weight_map.sum();
    std::cout << "归一化检查        : "
              << (std::abs(wsum - 1.0) < 1e-6 ? "OK" : "FAIL")
              << std::endl;

    // ── ASCII 可视化（source_map 二值化）────────────────────────────────
    double thresh = sm.source_map.maxCoeff() * 0.3;
    int rows = sm.source_map.rows(), cols = sm.source_map.cols();
    int mid_r = rows / 2, mid_c = cols / 2;
    int hw2 = std::min(10, (rows - 1) / 2);             // 安全半宽（不会越界）
    int r0 = mid_r - hw2, r1 = mid_r + hw2;
    int c0 = mid_c - hw2, c1 = mid_c + hw2;
    std::cout << "\nSource map (center " << (r1 - r0 + 1) << "x" << (c1 - c0 + 1)
              << ", threshold=" << std::fixed << std::setprecision(2) << thresh << "):" << std::endl;
    for (int i = r0; i <= r1; ++i) {
        std::cout << "  ";
        for (int j = c0; j <= c1; ++j)
            std::cout << (sm.source_map(i, j) > thresh ? "█" : "·");
        std::cout << "\n";
    }

    // ── 导出 source_weight_map 为 txt ────────────────────────────────────
    std::ofstream ofs("source_weight_map.txt");
    if (ofs.is_open()) {
        ofs << std::fixed << std::setprecision(8);
        ofs << "# source_weight_map  rows=" << rows << " cols=" << cols << "\n";
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                ofs << sm.source_weight_map(i, j);
                if (j < cols - 1) ofs << " ";
            }
            ofs << "\n";
        }
        std::cout << "\nsource_weight_map 已写入 source_weight_map.txt"
                  << "  (" << rows << "x" << cols << " 矩阵)" << std::endl;
    } else {
        std::cout << "WARNING: 无法写入 source_weight_map.txt" << std::endl;
    }

    // ── 同时导出 source_map（原始强度）──────────────────────────────────
    std::ofstream ofs2("source_map.txt");
    if (ofs2.is_open()) {
        ofs2 << std::fixed << std::setprecision(8);
        ofs2 << "# source_map  rows=" << rows << " cols=" << cols << "\n";
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                ofs2 << sm.source_map(i, j);
                if (j < cols - 1) ofs2 << " ";
            }
            ofs2 << "\n";
        }
        std::cout << "source_map        已写入 source_map.txt" << std::endl;
    }


    std::cout << "\n========== 6. Abbe 成像测试 ==========" << std::endl;

    // 复用已有的 Grid(257, 4.0), Pupil, Source
    // 重新构造完整的成像流水线
    Grid img_grid(257, 4.0);
    const auto& gc6 = img_grid.grid_coords();

    // ---- 构造 Pupil ----
    Params pp;
    pp.NA         = 1.35;
    pp.wavelength = 193.0;
    pp.n          = 1.44;
    pp.defocus_nm = 0.0;   // 无离焦
    pp.zernike_coeffs = {{4, 0.05} };
    Pupil img_pupil(pp, gc6.Fx_2d, gc6.Fy_2d);

    // ---- 构造 Source ----
    SourceParams sp6;
    sp6.wavelength_nm = 193.0;
    sp6.NA            = 1.35;
    sp6.n             = 1.44;
    sp6.sigma_in      = 0.6;
    sp6.sigma_out     = 0.9;
    sp6.upsample      = 10;
    sp6.smoothing     = 0.01;
    Source img_src(sp6);
    img_src.compute_source_map(gc6.Fx_1d, gc6.Fy_1d);

    // ---- 预计算 ----
    auto t0 = std::chrono::high_resolution_clock::now();
    LithoPrepare prep(img_grid, img_pupil, img_src);
    const auto& cache = prep.cache();
    auto t1 = std::chrono::high_resolution_clock::now();

    std::cout << "Grid      : " << cache.N << "×" << cache.N << std::endl;
    std::cout << "Source pts: " << cache.source_fs_phys.size() << std::endl;
    std::cout << "Prep time : "
              << std::chrono::duration<double, std::milli>(t1 - t0).count()
              << " ms" << std::endl;

    // ---- 构造测试掩模：孤立线 / 密集线 ----
    int NN = cache.N;
    int mid_pt = NN / 2;

    // 掩模 1：垂直孤立线（宽约 100nm）
    double dx_nm = 4.0;                     // pixel size
    Eigen::MatrixXd mask_iso = Eigen::MatrixXd::Zero(NN, NN);
    int w_iso = std::max(1, static_cast<int>(100.0 / dx_nm));   // 100nm → 25 像素
    int j_start = mid_pt - w_iso / 2;
    for (int j = j_start; j < j_start + w_iso && j < NN; ++j)
        if (j >= 0) mask_iso.col(j).setOnes();   // 全部行 = 1，宽 w_iso 列

    // 掩模 2：密集 line/space（pitch = 200nm，duty 1:1）
    double pitch_nm = 200.0;
    int pitch_px = std::max(2, static_cast<int>(pitch_nm / dx_nm));
    int half_pitch = pitch_px / 2;
    Eigen::MatrixXd mask_ls = Eigen::MatrixXd::Zero(NN, NN);
    for (int j = 0; j < NN; ++j) {
        int pos = (j - mid_pt + NN * pitch_px) % pitch_px;  // 相对于中心的周期位置
        if (pos < half_pitch)
            mask_ls.col(j).setOnes();
    }

    // ---- 成像 ----
    Imaging img(prep.cache());

    auto t2 = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd aerial = img.compute_wafer(mask_iso);
    auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "Aerial time: "
              << std::chrono::duration<double, std::milli>(t3 - t2).count()
              << " ms" << std::endl << std::endl;

    // ---- 结果展示 ----
    double a_max = aerial.maxCoeff();

    // 中心一行剖面
    std::cout << "--- 孤立线 aerial 中心行剖面 (y=0) ---" << std::endl;
    std::cout << "  ";
    for (int j = mid_pt - 30; j <= mid_pt + 30; ++j) {
        char c = ' ';
        double v = aerial(mid_pt, j) / a_max;
        if      (v > 0.8) c = '#';
        else if (v > 0.5) c = '=';
        else if (v > 0.2) c = '-';
        std::cout << c;
    }
    std::cout << "\n  max=" << a_max << ", min=" << aerial.minCoeff() << std::endl;
    std::cout.precision(6);  // 恢复默认精度

    // 中心区域 ASCII 可视化
    int hw3 = std::min(20, (NN - 1) / 2);
    std::cout << "\nAerial (center " << (2*hw3+1) << "x" << (2*hw3+1)
              << ", threshold=" << std::fixed << std::setprecision(2)
              << (0.3 * a_max) << "):" << std::endl;
    for (int i = mid_pt - hw3; i <= mid_pt + hw3; ++i) {
        std::cout << "  ";
        for (int j = mid_pt - hw3; j <= mid_pt + hw3; ++j) {
            double v = aerial(i, j) / a_max;
        if      (v > 0.8) std::cout << "█";
        else if (v > 0.5) std::cout << "▓";
        else if (v > 0.3) std::cout << "▒";
        else if (v > 0.1) std::cout << "░";
        else              std::cout << "·";
        }
        std::cout << "\n";
    }

    // ---- 密集 L/S 成像 ----
    std::cout << "\n--- 密集线 aerial (中心区域) ---" << std::endl;
    Eigen::MatrixXd aerial_ls = img.compute_aerial(mask_ls);
    double a_ls_max = aerial_ls.maxCoeff();
    std::cout << "  max=" << a_ls_max << ", min=" << aerial_ls.minCoeff() << std::endl;
    std::cout << "  中心行剖面: ";
    for (int j = mid_pt - 30; j <= mid_pt + 30; ++j) {
        double v = aerial_ls(mid_pt, j) / a_ls_max;
        if      (v > 0.8) std::cout << '#';
        else if (v > 0.5) std::cout << '=';
        else if (v > 0.2) std::cout << '-';
        else              std::cout << ' ';
    }
    std::cout << std::endl;

    // ---- 导出文件 ----
    auto export_mat = [](const Eigen::MatrixXd& M, const std::string& fname) {
        std::ofstream f(fname);
        if (!f) return;
        f << std::fixed << std::setprecision(8);
        f << "# rows=" << M.rows() << " cols=" << M.cols() << "\n";
        for (int i = 0; i < M.rows(); ++i) {
            for (int j = 0; j < M.cols(); ++j) {
                f << M(i, j);
                if (j < M.cols() - 1) f << " ";
            }
            f << "\n";
        }
        std::cout << "  已导出: " << fname << " (" << M.rows() << "x" << M.cols() << ")" << std::endl;
    };

    export_mat(mask_iso,    "mask_isolated.txt");
    export_mat(aerial,      "aerial_isolated.txt");
    export_mat(mask_ls,     "mask_dense.txt");
    export_mat(aerial_ls,   "aerial_dense.txt");

    // ---- Wafer 显影 ----
    std::cout << "\n--- Wafer 显影 (sigmoid threshold=0.3, step=0.05) ---" << std::endl;
    double threshold = 0.25;
    double step_w    = 85;
    Eigen::MatrixXd wafer_iso = img.compute_wafer(aerial, threshold, step_w);
    Eigen::MatrixXd wafer_ls  = img.compute_wafer(aerial_ls, threshold, step_w);

    // 孤立线 wafer 中心行剖面
    std::cout << "  孤立线 wafer 中心行: ";
    for (int j = mid_pt - 30; j <= mid_pt + 30; ++j) {
        double v = wafer_iso(mid_pt, j);
        if      (v > 0.8) std::cout << "█";
        else if (v > 0.5) std::cout << "▓";
        else if (v > 0.3) std::cout << "▒";
        else if (v > 0.1) std::cout << "░";
        else              std::cout << "·";
    }
    std::cout << "  (1=显影, 0=未显影)\n";

    // 密集 L/S wafer 剖面
    std::cout << "  密集线 wafer 中心行: ";
    for (int j = mid_pt - 30; j <= mid_pt + 30; ++j) {
        double v = wafer_ls(mid_pt, j);
        if      (v > 0.8) std::cout << "█";
        else if (v > 0.5) std::cout << "▓";
        else if (v > 0.3) std::cout << "▒";
        else if (v > 0.1) std::cout << "░";
        else              std::cout << "·";
    }
    std::cout << "\n";

    // 导出 wafer
    export_mat(wafer_iso,  "wafer_isolated.txt");
    export_mat(wafer_ls,   "wafer_dense.txt");

    // 导出中心行剖面（方便画 1D 曲线）
    auto save_profile = [](const Eigen::MatrixXd& M, int row, const std::string& fname) {
        std::ofstream f(fname);
        if (!f) return;
        f << std::fixed << std::setprecision(8);
        f << "# profile at row=" << row << "\n";
        f << "# col value\n";
        for (int j = 0; j < M.cols(); ++j)
            f << j << " " << M(row, j) << "\n";
    };
    int center_row = mid_pt;
    save_profile(mask_iso,   center_row, "profile_mask_iso.txt");
    save_profile(aerial,     center_row, "profile_aerial_iso.txt");
    save_profile(wafer_iso,  center_row, "profile_wafer_iso.txt");
    save_profile(mask_ls,    center_row, "profile_mask_ls.txt");
    save_profile(aerial_ls,  center_row, "profile_aerial_ls.txt");
    save_profile(wafer_ls,   center_row, "profile_wafer_ls.txt");
    std::cout << "  剖面数据已导出到 profile_*.txt" << std::endl;


    std::cout << "\n========== 全部通过 ==========" << std::endl;
    return 0;
}
