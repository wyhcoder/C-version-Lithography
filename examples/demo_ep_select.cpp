#include "ep_select.h"

#include <Eigen/Dense>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace litho;

// 把矩阵打印成 ASCII 图
static void print_ascii(const Eigen::MatrixXd& M, double thresh = 0.5,
                         int max_hw = 20) {
    int rows = static_cast<int>(M.rows());
    int cols = static_cast<int>(M.cols());
    int r0 = 0, r1 = rows - 1, c0 = 0, c1 = cols - 1;
    if (rows > 2 * max_hw + 1) {
        r0 = rows/2 - max_hw;
        r1 = rows/2 + max_hw;
    }
    if (cols > 2 * max_hw + 1) {
        c0 = cols/2 - max_hw;
        c1 = cols/2 + max_hw;
    }
    for (int i = r0; i <= r1; ++i) {
        std::cout << "  ";
        for (int j = c0; j <= c1; ++j)
            std::cout << (M(i,j) > thresh ? "█" : "·");
        std::cout << "\n";
    }
}

// 在矩阵上叠加 EP 点并打印
static void print_with_eps(const Eigen::MatrixXd& mask,
                            const Eigen::MatrixXd& eps,
                            int max_hw = 20) {
    int rows = static_cast<int>(mask.rows());
    int cols = static_cast<int>(mask.cols());
    int r0 = std::max(0, rows/2 - max_hw);
    int r1 = std::min(rows-1, rows/2 + max_hw);
    int c0 = std::max(0, cols/2 - max_hw);
    int c1 = std::min(cols-1, cols/2 + max_hw);

    // 把 EP 点存入 set 方便查找
    std::set<std::pair<int,int>> ep_set;
    for (int k = 0; k < eps.rows(); ++k)
        ep_set.insert({eps(k,0), eps(k,1)});

    for (int i = r0; i <= r1; ++i) {
        std::cout << "  ";
        for (int j = c0; j <= c1; ++j) {
            if (ep_set.count({i,j})) std::cout << "○";
            else std::cout << (mask(i,j) > 0.5 ? "█" : "·");
        }
        std::cout << "\n";
    }
}

// 把 EP 点和权重写到 txt
static void save_eps(const EpsResult& r, const std::string& fname) {
    std::ofstream f(fname);
    if (!f) { std::cerr << "无法写入 " << fname << "\n"; return; }
    f << "# y x w_epe w_meef\n";
    int M = static_cast<int>(r.eps.rows());
    for (int i = 0; i < M; ++i)
        f << r.eps(i,0) << " " << r.eps(i,1) << " "
          << r.weight_epe(i) << " " << r.weight_meef(i) << "\n";
    std::cout << "  已导出: " << fname << "  (" << M << " 个EP点)\n";
}

int main() {
    // ── 1. 构造测试 mask：孤立矩形（模拟单个线端图案）──────────────────
    std::cout << "========== EP Select Demo ==========\n\n";

    const int N = 64;
    Eigen::MatrixXd mask = Eigen::MatrixXd::Zero(N, N);

    // 在中心放一个 20×10 的矩形（模拟 line-end）
    int cy = N/2, cx = N/2;
    int half_h = 10, half_w = 5;
    for (int i = cy - half_h; i <= cy + half_h; ++i)
        for (int j = cx - half_w; j <= cx + half_w; ++j)
            mask(i, j) = 1.0;

    std::cout << "--- 测试 mask（64×64，中心矩形）---\n";
    print_ascii(mask, 0.5, 15);

    // ── 2. 构造 EpSelect ────────────────────────────────────────────────
    double mid_weight   = 10.0;
    double other_weight = 1.0;
    EpSelect ep(mask, mid_weight, other_weight, "line_end_test");

    // ── 3. select_eps_others ────────────────────────────────────────────
    std::cout << "\n--- select_eps_others(interval_line=4, interval_corner=1) ---\n";
    auto res_others = ep.select_eps_others(4, 1);
    int M_oth = static_cast<int>(res_others.eps.rows());
    std::cout << "  EP 总点数      : " << M_oth << "\n";
    std::cout << "  w_epe = 1 的点 : "
              << (res_others.weight_epe.array() > 0.5).count() << "\n";
    std::cout << "  w_meef max     : " << res_others.weight_meef.maxCoeff() << "\n";
    std::cout << "  w_meef min     : " << res_others.weight_meef.minCoeff() << "\n";

    std::cout << "\n  mask + EP 叠加图（○=EP点）:\n";
    print_with_eps(mask, res_others.eps, 15);
    save_eps(res_others, "eps_others.txt");

    // ── 4. select_eps_via（把 mask 当 via 孔）──────────────────────────
    // 构造一个小方形 via 图案
    Eigen::MatrixXd via_mask = Eigen::MatrixXd::Zero(N, N);
    for (int i = cy - 5; i <= cy + 5; ++i)
        for (int j = cx - 5; j <= cx + 5; ++j)
            via_mask(i, j) = 1.0;

    EpSelect ep_via(via_mask, mid_weight, other_weight);
    std::cout << "\n--- select_eps_via(r=2)  via 孔图案 ---\n";
    auto res_via = ep_via.select_eps_via(1);
    int M_via = static_cast<int>(res_via.eps.rows());
    std::cout << "  EP 总点数      : " << M_via << "\n";
    std::cout << "  w_epe = 1 的点 : "
              << (res_via.weight_epe.array() > 0.5).count() << "\n";

    std::cout << "\n  via mask + EP 叠加图（○=EP点）:\n";
    print_with_eps(via_mask, res_via.eps, 10);
    save_eps(res_via, "eps_via.txt");

    // ── 5. extract_mask_control_points ─────────────────────────────────
    std::cout << "\n--- extract_mask_control_points(k=3) ---\n";
    auto cps = ep.extract_mask_control_points(3);
    std::cout << "  轮廓数量: " << cps.size() << "\n";
    for (int ci = 0; ci < (int)cps.size(); ++ci)
        std::cout << "  轮廓 " << ci << ": " << cps[ci].size() << " 个控制点\n";

    // ── 6. remove_adjacent_close_points ────────────────────────────────
    auto cps_cleaned = EpSelect::remove_adjacent_close_points(cps, 3.0);
    std::cout << "\n--- remove_adjacent_close_points(threshold=3) ---\n";
    for (int ci = 0; ci < (int)cps_cleaned.size(); ++ci)
        std::cout << "  轮廓 " << ci << ": " << cps[ci].size()
                  << " → " << cps_cleaned[ci].size() << " 个控制点\n";

    std::cout << "\n========== Demo 完成 ==========\n";
    return 0;
}
