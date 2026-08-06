#include "grid.h"
#include "pupil.h"
#include "source.h"
#include "litho_prepare.h"
#include "imaging.h"
#include "meef_pipeline.h"
#include "meef_optimizer.h"
#include "msaa.h"
#include "parametric.h"

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>

using namespace litho;

// 加载 txt → MatrixXd
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

int main() {
    std::cout << "========== MEEF 优化 Demo ==========\n" << std::endl;

    // ── 0. 加载数据 ─────────────────────────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();

    Eigen::MatrixXd target = load_txt("target_mask.txt");  // 257x257
    int N = target.rows();
    std::cout << "target mask : " << N << "x" << N << std::endl;

    // ── 1. 构建成像系统 ───────────────────────────────────────────
    Grid grid(N, 6.0);
    const auto& gc = grid.grid_coords();

    Params pp;
    pp.NA = 1.35; pp.wavelength = 193.0; pp.n = 1.44; pp.defocus_nm = 0.0;
    Pupil pupil(pp, gc.Fx_2d, gc.Fy_2d);

    SourceParams sp;
    sp.wavelength_nm = 193.0; sp.NA = 1.35; sp.n = 1.44;
    sp.sigma_in = 0.6; sp.sigma_out = 0.9;
    sp.upsample = 10; sp.smoothing = 0.01;
    Source src(sp);
    src.compute_source_map(gc.Fx_1d, gc.Fy_1d);

    LithoPrepare prep(grid, pupil, src);
    const auto& cache = prep.cache();
    auto t1 = std::chrono::high_resolution_clock::now();
    std::cout << "imager prep: "
              << std::chrono::duration<double, std::milli>(t1-t0).count() << " ms\n";

    std::cout << "source pts : " << cache.source_fs_phys.size() << std::endl;

    Imaging imaging(cache, 6);  // SOCS: 6 coherent modes

    // ── 2. MEEF 管道 ──────────────────────────────────────────────
    MEEFConfig cfg;
    cfg.pattern_name    = "line_end";
    cfg.ls_mask_path    = "lsm_mask.txt";
    cfg.file_name       = "demo";
    cfg.main_cp_interval = 4;
    cfg.main_symmetry   = "none";
    cfg.curve_type      = "BS";
    cfg.delta           = 0.15;
    cfg.dilate_radius   = 2;
    cfg.msaa_level      = 4;
    cfg.interval_line   = 5;
    cfg.interval_corner = 1;
    cfg.mid_weight      = 4.0;
    cfg.other_weight    = 1.0;

    MEEFPipeline pipeline(target, cfg);

    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "pipeline:   "
              << std::chrono::duration<double, std::milli>(t2-t1).count() << " ms\n";

    std::cout << "main cps   : " << pipeline.num_cps << std::endl;
    std::cout << "sraf cps   : " << pipeline.num_sraf_cps << std::endl;
    std::cout << "eps points : " << pipeline.num_eps << std::endl;
    std::cout << "weps count : " << pipeline.num_weps << std::endl;
    std::cout << "M x N      : " << pipeline.num_eps << " x " << pipeline.num_cps
              << "  " << (pipeline.num_eps > pipeline.num_cps ? "OK" : "FAIL (need rows>cols)")
              << std::endl;

    // 保存中间产物
    save_txt(pipeline.lsm_mask,    "output_lsm_mask.txt");
    save_txt(pipeline.sraf_mask,   "output_sraf_mask.txt");
    save_txt(pipeline.initial_mask,"output_initial_mask.txt");

    // ── 3. MEEF 优化器 ───────────────────────────────────────────
    AntiAliasRenderer renderer(4);
    ParametricDemo parametric(cfg.curve_type, target, 16);

    MEEFOptimizer opt(imaging, parametric, renderer,
                      pipeline.target_mask,
                      pipeline.sraf_mask,
                      pipeline.cps,
                      pipeline.eps_result,
                      pipeline.curve_type,
                      pipeline.delta,
                      0.3);   // resist threshold

    auto t3 = std::chrono::high_resolution_clock::now();

    auto rec = opt.run(40);     // 5 次迭代

    auto t4 = std::chrono::high_resolution_clock::now();
    std::cout << "\noptim time: "
              << std::chrono::duration<double, std::milli>(t4-t3).count() << " ms\n";

    // ── 4. 输出结果 ──────────────────────────────────────────────
    std::cout << "\n========== 结果 ==========\n";
    std::cout << "iter  EPE       wEPE      PE\n";
    for (size_t i = 0; i < rec.epe_history.size(); ++i) {
        std::cout << std::setw(4) << i << "  "
                  << std::fixed << std::setprecision(6)
                  << rec.epe_history[i] << "  "
                  << rec.wepe_history[i] << "  "
                  << std::setprecision(5) << rec.pe_history[i] << "\n";
    }

    std::cout << "\nbest wEPE at iter " << rec.best_wepe_iter
              << ", value = " << rec.best_wepe << std::endl;
    std::cout << "best EPE  at iter " << rec.best_epe_iter
              << ", value = " << rec.best_epe << std::endl;

    save_txt(rec.best_wepe_mask, "output_best_wepe_mask.txt");
    save_txt(rec.best_epe_mask,  "output_best_epe_mask.txt");

    std::cout << "\n========== 完成 ==========\n";
    return 0;
}
