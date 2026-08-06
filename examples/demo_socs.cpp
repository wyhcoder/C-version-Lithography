#include "grid.h"

#include "pupil.h"
#include "source.h"
#include "litho_prepare.h"
#include "imaging.h"

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <iomanip>

#include <chrono>
#include <cstdlib>
#include <string>
#include <sstream>

using namespace litho;

static Eigen::MatrixXd load_txt(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open " + path);
    std::vector<std::vector<double>> rows;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::vector<double> vals;
        double v;
        while (ss >> v) vals.push_back(v);
        if (!vals.empty()) rows.push_back(vals);
    }
    int H = (int)rows.size(), W = (int)rows[0].size();
    Eigen::MatrixXd m(H, W);
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j) m(i,j) = rows[i][j];
    return m;
}

static void save_txt(const Eigen::MatrixXd& M, const std::string& fname) {
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
}

int main(){
    std::cout << "========== SOCS vs Abbe 成像对比 ==========\n" << std::endl;

    // ---- 构建光刻系统 ----
    litho::Grid img_grid(257, 4.0); 
    std::cout << "Grid: " << img_grid.size()
              << "x" << img_grid.size()
              << ", pixel_size = " << img_grid.pixel_size() << " nm" << std::endl;
    const auto& gc6 = img_grid.grid_coords();

    // Pupil
    Params pp_params;
    pp_params.NA = 1.35;
    pp_params.wavelength = 193.0;
    pp_params.n = 1.414;
    pp_params.defocus_nm = 0.0;
    pp_params.zernike_coeffs = {{4, 0}};
    litho::Pupil img_pupil(pp_params, gc6.Fx_2d, gc6.Fy_2d);

    // Source (环形照明)
    litho::SourceParams sp6;
    sp6.wavelength_nm = 193.0;
    sp6.NA            = 1.35;
    sp6.n             = 1.44;
    sp6.sigma_in      = 0.6;
    sp6.sigma_out     = 0.9;
    sp6.upsample      = 10;
    sp6.smoothing     = 0.01;
    litho::Source img_source(sp6);
    img_source.compute_source_map(gc6.Fx_1d, gc6.Fy_1d);

    // ---- 预计算 ----
    auto t0 = std::chrono::high_resolution_clock::now();
    LithoPrepare prep(img_grid, img_pupil, img_source, true);
    const auto& cache = prep.cache();
    auto t1 = std::chrono::high_resolution_clock::now();
    LithoPrepare prep_abbe(img_grid, img_pupil, img_source, false);
    const auto& cache_abbe = prep_abbe.cache();
    auto t2 = std::chrono::high_resolution_clock::now();

    int Ns = static_cast<int>(cache.source_fs_phys.size());
    std::cout << "Grid      : " << cache.N << "×" << cache.N << std::endl;
    std::cout << "Source pts: " << Ns << std::endl;
    std::cout << "Prep time for socs : "
              << std::chrono::duration<double, std::milli>(t1 - t0).count()
              << " ms" << std::endl;
    std::cout << "Prep time for abbe : "
              << std::chrono::duration<double, std::milli>(t2 - t1).count()
              << " ms" << std::endl;
    // ---- 构造两个 Imaging 对象 ----
    // socs_modes=10 → SOCS 10 个相干模
    // socs_modes=0  → 纯 Abbe
    litho::Imaging imaging_socs(cache);
    litho::Imaging imaging_abbe(cache_abbe);

    

    // ---- 加载掩模 ----
    std::cout << "\n========== 加载掩模 ==========" << std::endl;
    // Eigen::MatrixXd mask = load_txt("lsm_mask.txt");
    // std::cout << "Mask: " << mask.rows() << "x" << mask.cols() << std::endl;
    // std::cout << "  range: [" << mask.minCoeff() << ", " << mask.maxCoeff() << "]" << std::endl;
    Eigen::MatrixXd mask = load_txt("ls_image对角通孔pe_epe_pvband.txt");
    std::cout << "Mask: " << mask.rows() << "x" << mask.cols() << std::endl;
    std::cout << "  range: [" << mask.minCoeff() << ", " << mask.maxCoeff() << "]" << std::endl;

    // ---- SOCS 成像 ----
    std::cout << "\n========== SOCS 成像 ==========" << std::endl;
    auto t_socs0 = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd aerial_socs = imaging_socs.compute_aerial(mask);
    auto t_socs1 = std::chrono::high_resolution_clock::now();
    double time_socs = std::chrono::duration<double, std::milli>(t_socs1 - t_socs0).count();
    std::cout << "SOCS aerial time: " << time_socs << " ms" << std::endl;
    std::cout << "  range: [" << aerial_socs.minCoeff() << ", " << aerial_socs.maxCoeff() << "]" << std::endl;

    Eigen::MatrixXd wafer_socs = imaging_socs.compute_wafer(aerial_socs);
    save_txt(aerial_socs, "aerial_socs.txt");
    save_txt(wafer_socs, "wafer_socs.txt");
    std::cout << "  saved: aerial_socs.txt, wafer_socs.txt" << std::endl;

    // ---- Abbe 成像 ----
    std::cout << "\n========== Abbe 成像 (精确参考) ==========" << std::endl;
    auto t_abbe0 = std::chrono::high_resolution_clock::now();
    Eigen::MatrixXd aerial_abbe = imaging_abbe.compute_aerial(mask);
    auto t_abbe1 = std::chrono::high_resolution_clock::now();
    double time_abbe = std::chrono::duration<double, std::milli>(t_abbe1 - t_abbe0).count();
    std::cout << "Abbe aerial time: " << time_abbe << " ms" << std::endl;
    std::cout << "  range: [" << aerial_abbe.minCoeff() << ", " << aerial_abbe.maxCoeff() << "]" << std::endl;

    Eigen::MatrixXd wafer_abbe = imaging_abbe.compute_wafer(aerial_abbe);
    save_txt(aerial_abbe, "aerial_abbe.txt");
    save_txt(wafer_abbe, "wafer_abbe.txt");
    std::cout << "  saved: aerial_abbe.txt, wafer_abbe.txt" << std::endl;

    // ---- 对比分析 ----
    std::cout << "\n========== 对比分析 ==========" << std::endl;

    // 加速比
    std::cout << "Speedup (SOCS/Abbe): "
              << std::fixed << std::setprecision(1)
              << (time_abbe / time_socs) << "x"
              << std::endl;

    // 空间像误差
    Eigen::MatrixXd aerial_diff = (aerial_socs - aerial_abbe).cwiseAbs();
    double aerial_max_err = aerial_diff.maxCoeff();
    double aerial_rms_err = std::sqrt(aerial_diff.array().square().mean());
    double aerial_rel_err = 100.0 * aerial_rms_err / (aerial_abbe.maxCoeff() - aerial_abbe.minCoeff() + 1e-12);

    std::cout << "Aerial error:  max = " << std::scientific << std::setprecision(4)
              << aerial_max_err
              << ", RMS = " << aerial_rms_err
              << std::fixed << " (" << std::setprecision(3) << aerial_rel_err << "%)" << std::endl;

    // Wafer 误差（二值接近，看像素差异比例）
    Eigen::MatrixXd wafer_diff = (wafer_socs - wafer_abbe).cwiseAbs();
    double wafer_max_err   = wafer_diff.maxCoeff();
    double wafer_rms_err   = std::sqrt(wafer_diff.array().square().mean());
    int    wafer_mis_total  = (int)mask.size();
    int    wafer_mis_01     = (int)((wafer_diff.array() > 0.1).count());  // >0.1 差异

    std::cout << "Wafer error:   max = " << wafer_max_err
              << ", RMS = " << wafer_rms_err
              << ", mis-match(>0.1) = " << wafer_mis_01
              << " / " << wafer_mis_total
              << " (" << std::setprecision(2)
              << (100.0 * wafer_mis_01 / wafer_mis_total) << "%)" << std::endl;

    // ---- 导出对比文件 ----
    save_txt(aerial_diff, "aerial_diff.txt");
    save_txt(wafer_diff,  "wafer_diff.txt");
    std::cout << "  saved: aerial_diff.txt, wafer_diff.txt" << std::endl;

    // ---- 可视化 ----
    std::cout << "\n========== 生成可视化 ==========" << std::endl;

    // 1) SOCS vs Abbe aerial 并排 + 差值
    {
        int ret = std::system("../dist/litho_viewer --mode compare"
                    " --files aerial_socs.txt aerial_abbe.txt"
                    " --labels \"SOCS Aerial\" \"Abbe Aerial\""
                    " --title \"SOCS vs Abbe - Aerial Image\""
                    " --save aerial_compare.png");
        std::cout << (ret == 0 ? "  saved: aerial_compare.png" : "  FAILED") << std::endl;
    }

    // 2) SOCS vs Abbe wafer 并排 + 差值
    {
        int ret = std::system("../dist/litho_viewer --mode compare"
                    " --files wafer_socs.txt wafer_abbe.txt"
                    " --labels \"SOCS Wafer\" \"Abbe Wafer\""
                    " --title \"SOCS vs Abbe - Wafer Pattern\""
                    " --save wafer_compare.png");
        std::cout << (ret == 0 ? "  saved: wafer_compare.png" : "  FAILED") << std::endl;
    }

    // 3) 差值热图
    {
        int ret = std::system("../dist/litho_viewer --mode matrix"
                    " --files aerial_diff.txt"
                    " --cmaps bwr"
                    " --labels \"Aerial Error |SOCS-Abbe|\""
                    " --title \"SOCS vs Abbe - Aerial Difference\""
                    " --save aerial_diff.png");
        std::cout << (ret == 0 ? "  saved: aerial_diff.png" : "  FAILED") << std::endl;
    }

    // 4) 整体一览：SOCS aerial, Abbe aerial, wafer_socs, wafer_abbe
    {
        int ret = std::system("../dist/litho_viewer --mode matrix"
                    " --files aerial_socs.txt aerial_abbe.txt wafer_socs.txt wafer_abbe.txt"
                    " --labels \"SOCS Aerial\" \"Abbe Aerial\" \"SOCS Wafer\" \"Abbe Wafer\""
                    " --cmaps inferno inferno gray gray"
                    " --title \"SOCS vs Abbe - Full Overview\""
                    " --save socs_vs_abbe_overview.png");
        std::cout << (ret == 0 ? "  saved: socs_vs_abbe_overview.png" : "  FAILED") << std::endl;
    }

    std::cout << "\n========== 完成 ==========" << std::endl;

    return 0;
}
