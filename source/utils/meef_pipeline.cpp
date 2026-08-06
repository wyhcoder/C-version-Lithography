#include "meef_pipeline.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <queue>
#include <set>
#include <stdexcept>
#include <opencv2/imgproc.hpp>

namespace litho {

// ── helper: load txt → MatrixXd ────────────────────────────────────────────
Eigen::MatrixXd MEEFPipeline::_load_txt(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("无法打开 LSM 文件: " + path);

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
    if (rows.empty()) throw std::runtime_error("LSM 文件为空: " + path);

    int H = static_cast<int>(rows.size());
    int W = static_cast<int>(rows[0].size());
    Eigen::MatrixXd m(H, W);
    for (int i = 0; i < H; ++i)
        for (int j = 0; j < W; ++j)
            m(i, j) = rows[i][j];
    return m;
}

// ── center-crop / pad ──────────────────────────────────────────────────────
Eigen::MatrixXd MEEFPipeline::_center_crop_pad(
    const Eigen::MatrixXd& src, int th, int tw)
{
    int sh = static_cast<int>(src.rows()), sw = static_cast<int>(src.cols());
    if (sh == th && sw == tw) return src;

    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(th, tw);

    // 高度方向
    int r_start = 0, r_end = th, s_row = 0, s_h = sh;
    if (sh > th) {
        s_row = (sh - th) / 2;
        s_h   = th;
    } else {
        r_start = (th - sh) / 2;
        r_end   = r_start + sh;
    }
    // 宽度方向
    int c_start = 0, c_end = tw, s_col = 0, s_w = sw;
    if (sw > tw) {
        s_col = (sw - tw) / 2;
        s_w   = tw;
    } else {
        c_start = (tw - sw) / 2;
        c_end   = c_start + sw;
    }

    out.block(r_start, c_start, s_h, s_w) = src.block(s_row, s_col, s_h, s_w);
    return out;
}

// ── binary dilation (disk kernel) ──────────────────────────────────────────
Eigen::MatrixXd MEEFPipeline::_binary_dilation(
    const Eigen::MatrixXd& mask, int radius)
{
    int H = static_cast<int>(mask.rows());
    int W = static_cast<int>(mask.cols());

    // 生成圆形结构元素
    std::vector<std::pair<int,int>> offsets;
    int r2 = radius * radius;
    for (int dy = -radius; dy <= radius; ++dy)
        for (int dx = -radius; dx <= radius; ++dx)
            if (dy*dy + dx*dx <= r2)
                offsets.push_back({dy, dx});

    Eigen::MatrixXd result = mask;  // copy
    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            if (mask(r, c) > 0.5) {
                double val = mask(r, c);
                for (auto [dy, dx] : offsets) {
                    int nr = r + dy, nc = c + dx;
                    if (nr >= 0 && nr < H && nc >= 0 && nc < W)
                        result(nr, nc) = std::max(result(nr, nc), val);
                }
            }
        }
    }
    return (result.array() > 0.5).cast<double>();
}

// ── 连通块标记（BFS）+ 面积过滤 ────────────────────────────────────────────
std::vector<Eigen::MatrixXi> MEEFPipeline::_label_components(
    const Eigen::MatrixXd& bin, int min_area)
{
    int H = static_cast<int>(bin.rows());
    int W = static_cast<int>(bin.cols());
    Eigen::MatrixXi labels = Eigen::MatrixXi::Zero(H, W);
    std::vector<int> areas;
    int L = 0;

    for (int r = 0; r < H; ++r) {
        for (int c = 0; c < W; ++c) {
            if (bin(r, c) <= 0.5 || labels(r, c) != 0) continue;
            ++L;
            std::queue<std::pair<int,int>> q;
            q.push({r, c});
            labels(r, c) = L;
            int area = 0;
            while (!q.empty()) {
                auto [y, x] = q.front(); q.pop();
                ++area;
                for (int dy : {-1, 0, 1})
                    for (int dx : {-1, 0, 1}) {
                        if (dy == 0 && dx == 0) continue;
                        int ny = y + dy, nx = x + dx;
                        if (ny<0 || ny>=H || nx<0 || nx>=W) continue;
                        if (bin(ny, nx) > 0.5 && labels(ny, nx) == 0) {
                            labels(ny, nx) = L;
                            q.push({ny, nx});
                        }
                    }
            }
            areas.push_back(area);
        }
    }

    std::vector<Eigen::MatrixXi> comps;
    for (int l = 1; l <= L; ++l) {
        if (areas[l-1] < min_area) continue;
        Eigen::MatrixXi mask_l = (labels.array() == l).cast<int>();
        comps.push_back(mask_l);
    }
    return comps;
}

// ── SRAF 弧长重采样 ────────────────────────────────────────────────────────
Eigen::MatrixXd MEEFPipeline::_resample_arclen(
    const Eigen::MatrixXd& contour, double step, int min_n)
{
    int n = static_cast<int>(contour.rows());
    if (n < 3) return contour;

    // 累积弧长（闭合，不含重复首点）
    Eigen::VectorXd cum(n + 1);
    cum(0) = 0.0;
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        double dy = contour(j, 0) - contour(i, 0);
        double dx = contour(j, 1) - contour(i, 1);
        cum(i + 1) = cum(i) + std::sqrt(dy*dy + dx*dx);
    }
    double total = cum(n);
    if (total < 1e-6) return contour.row(0);

    int n_by = static_cast<int>(std::floor(total / std::max(step, 1e-3)));
    int num  = std::max(min_n, n_by);

    Eigen::MatrixXd res(num, 2);
    for (int k = 0; k < num; ++k) {
        double t = total * k / num;
        // 在 cum 上二分查找区间
        int idx = 0;
        while (idx + 1 <= n && cum(idx + 1) <= t) ++idx;
        double tt = (cum(idx + 1) - cum(idx) > 1e-12)
            ? (t - cum(idx)) / (cum(idx + 1) - cum(idx)) : 0.0;
        int i0 = idx % n, i1 = (idx + 1) % n;
        res(k, 0) = contour(i0, 0) + tt * (contour(i1, 0) - contour(i0, 0));
        res(k, 1) = contour(i0, 1) + tt * (contour(i1, 1) - contour(i0, 1));
    }
    return res;
}

// ── 对称变换 ────────────────────────────────────────────────────────────────
Eigen::MatrixXd MEEFPipeline::_apply_symmetry(
    const Eigen::MatrixXd& pts, int h, int w, const std::string& symmetry)
{
    int n = static_cast<int>(pts.rows());
    Eigen::MatrixXd out(n, 2);
    double cy = h / 2.0, cx = w / 2.0;

    if (symmetry == "center") {
        for (int i = 0; i < n; ++i) {
            out(i, 0) = 2*cy - pts(i, 0);
            out(i, 1) = 2*cx - pts(i, 1);
        }
    } else if (symmetry == "left-right") {
        for (int i = 0; i < n; ++i) {
            out(i, 0) = pts(i, 0);
            out(i, 1) = 2*cx - pts(i, 1);
        }
    } else if (symmetry == "diagonal") {
        for (int i = 0; i < n; ++i) {
            out(i, 0) = h - 1 - pts(i, 1);
            out(i, 1) = w - 1 - pts(i, 0);
        }
    } else {
        out = pts;
    }
    return out;
}

// ── SRAF 连通块对称配对 ───────────────────────────────────────────────────
void MEEFPipeline::_symmetry_pair_blocks(
    const std::vector<Eigen::MatrixXd>& contours,
    const std::vector<Eigen::Vector2d>& centroids,
    int h, int w, const std::string& symmetry,
    std::vector<int>& refs, std::vector<int>& mirrors,
    std::vector<int>& self_sym)
{
    int n = static_cast<int>(centroids.size());
    refs.clear(); mirrors.clear(); self_sym.clear();
    if (n == 0) return;

    // 镜像质心
    Eigen::MatrixXd cents(n, 2);
    for (int i = 0; i < n; ++i) {
        cents(i, 0) = centroids[i](0);
        cents(i, 1) = centroids[i](1);
    }
    Eigen::MatrixXd mirror_cents = _apply_symmetry(cents, h, w, symmetry);

    // 距离矩阵
    double tol = std::max(2.0, 0.015 * std::max(h, w));
    std::vector<bool> used(n, false);

    // 自身对称
    for (int i = 0; i < n; ++i) {
        double dy = mirror_cents(i, 0) - cents(i, 0);
        double dx = mirror_cents(i, 1) - cents(i, 1);
        if (std::sqrt(dy*dy + dx*dx) < tol) {
            self_sym.push_back(i);
            used[i] = true;
        }
    }

    // 配对
    for (int i = 0; i < n; ++i) {
        if (used[i]) continue;
        int best_j = -1;
        double best_d = 1e12;
        for (int j = 0; j < n; ++j) {
            if (used[j] || j == i) continue;
            double dy = mirror_cents(i, 0) - cents(j, 0);
            double dx = mirror_cents(i, 1) - cents(j, 1);
            double d = std::sqrt(dy*dy + dx*dx);
            if (d < best_d) { best_d = d; best_j = j; }
        }
        if (best_j >= 0 && best_d < tol) {
            refs.push_back(i);
            mirrors.push_back(best_j);
            used[i] = true;
            used[best_j] = true;
        } else {
            self_sym.push_back(i);
            used[i] = true;
        }
    }
}

// ══════════════════════════════════════════════════════════════════════════
//   MEEFPipeline 构造（全流程编排）
// ══════════════════════════════════════════════════════════════════════════
MEEFPipeline::MEEFPipeline(const Eigen::MatrixXd& target, const MEEFConfig& cfg)
    : _cfg(cfg)
{
    target_mask  = target;
    curve_type   = cfg.curve_type;
    pattern_name = cfg.pattern_name;
    delta        = cfg.delta;

    _load_lsm();
    _split_main_and_sraf();
    _extract_main();
    _render_sraf();
    _assemble_initial();
}

// ── 1. 加载 LSM ──────────────────────────────────────────────────────────
void MEEFPipeline::_load_lsm() {
    lsm_mask = _load_txt(_cfg.ls_mask_path);

    int th = static_cast<int>(target_mask.rows());
    int tw = static_cast<int>(target_mask.cols());
    if (lsm_mask.rows() != th || lsm_mask.cols() != tw) {
        lsm_mask = _center_crop_pad(lsm_mask, th, tw);
    }
}

// ── 2. 主图形 / SRAF 拆分 ────────────────────────────────────────────────
void MEEFPipeline::_split_main_and_sraf() {
    int H = static_cast<int>(lsm_mask.rows());
    int W = static_cast<int>(lsm_mask.cols());

    // target 二值化
    Eigen::MatrixXd origin_bin = (target_mask.array() > 0.0).cast<double>();

    // 膨胀掩膜
    Eigen::MatrixXd dilated = _binary_dilation(origin_bin, _cfg.dilate_radius);

    main_mask = Eigen::MatrixXd::Zero(H, W);
    Eigen::MatrixXd sraf_raw = Eigen::MatrixXd::Zero(H, W);
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) {
            if (dilated(r, c) > 0.5) main_mask(r, c) = lsm_mask(r, c);
            else                      sraf_raw(r, c) = lsm_mask(r, c);
        }

    // SRAF CP 提取基于 sraf_raw，存为临时成员
    // (sraf_raw 不对外暴露，用完即弃)
    {
        // 二值化
        Eigen::MatrixXd bin_sraf = (sraf_raw.array() > 1e-3).cast<double>();

        if (bin_sraf.sum() == 0) {
            num_sraf_cps = 0;
            sraf_contour_points = 0;
            sraf_mask = Eigen::MatrixXd::Zero(H, W);
            return;
        }

        auto comps = _label_components(bin_sraf, _cfg.sraf_min_area);
        if (comps.empty()) {
            num_sraf_cps = 0;
            sraf_contour_points = 0;
            sraf_mask = Eigen::MatrixXd::Zero(H, W);
            return;
        }

        // 提取每个连通块的外轮廓
        std::vector<Eigen::MatrixXd> block_contours;
        std::vector<Eigen::Vector2d> centroids;
        int total_contour_pts = 0;

        for (const auto& comp : comps) {
            cv::Mat cv_comp(H, W, CV_8UC1);
            for (int r = 0; r < H; ++r)
                for (int c = 0; c < W; ++c)
                    cv_comp.at<uint8_t>(r, c) = (comp(r, c) > 0) ? 255 : 0;

            std::vector<std::vector<cv::Point>> cv_contours;
            cv::findContours(cv_comp, cv_contours, cv::RETR_EXTERNAL,
                           cv::CHAIN_APPROX_NONE);
            if (cv_contours.empty()) continue;

            // 选最长轮廓
            auto& best = *std::max_element(cv_contours.begin(), cv_contours.end(),
                [](auto& a, auto& b){ return a.size() < b.size(); });

            Eigen::MatrixXd ctr(best.size(), 2);
            for (int i = 0; i < (int)best.size(); ++i) {
                ctr(i, 0) = best[i].y;
                ctr(i, 1) = best[i].x;
            }
            block_contours.push_back(ctr);
            centroids.push_back(Eigen::Vector2d(ctr.col(0).mean(), ctr.col(1).mean()));
            total_contour_pts += ctr.rows();
        }

        sraf_contour_points = total_contour_pts;

        // 对称处理
        std::string sym = (_cfg.main_symmetry == "none") ? "" : _cfg.main_symmetry;
        if (sym.empty()) {
            for (auto& ctr : block_contours)
                sraf_cps.push_back(_resample_arclen(ctr, _cfg.sraf_cp_arclen,
                                                    _cfg.sraf_min_cps));
        } else {
            std::vector<Eigen::MatrixXd> sraf_cps_temp(block_contours.size());
            std::vector<int> refs, mirrors, self_sym;
            _symmetry_pair_blocks(block_contours, centroids,
                                  H, W, sym, refs, mirrors, self_sym);

            for (int idx : self_sym)
                sraf_cps_temp[idx] = _resample_arclen(block_contours[idx],
                    _cfg.sraf_cp_arclen, _cfg.sraf_min_cps);

            for (int k = 0; k < (int)refs.size(); ++k) {
                auto rp = _resample_arclen(block_contours[refs[k]],
                    _cfg.sraf_cp_arclen, _cfg.sraf_min_cps);
                sraf_cps_temp[refs[k]] = rp;
                sraf_cps_temp[mirrors[k]] = _apply_symmetry(rp, H, W, sym);
            }

            for (auto& cp : sraf_cps_temp)
                if (cp.rows() > 0) sraf_cps.push_back(cp);
        }

        num_sraf_cps = 0;
        for (auto& c : sraf_cps) num_sraf_cps += c.rows();
    }
}

// ── 3. 主图形 CP + EP ────────────────────────────────────────────────────
void MEEFPipeline::_extract_main() {
    EpSelect ep(main_mask, _cfg.mid_weight, _cfg.other_weight, _cfg.pattern_name);

    std::string sym = (_cfg.main_symmetry == "none") ? "" : _cfg.main_symmetry;
    cps = ep.extract_mask_control_points(_cfg.main_cp_interval, sym);
    cps = EpSelect::remove_adjacent_close_points(cps, 2.0);

    num_cps = 0;
    for (auto& c : cps) num_cps += static_cast<int>(c.rows());

    // EP 选取
    if (_cfg.pattern_name == "中心对称通孔" || _cfg.pattern_name == "对角通孔")
        eps_result = ep.select_eps_via(2);
    else
        eps_result = ep.select_eps_others(_cfg.interval_line, _cfg.interval_corner);

    num_eps = static_cast<int>(eps_result.eps.rows());
    num_weps = static_cast<int>((eps_result.weight_epe.array() == 1.0).count());
}

// ── 4. SRAF 渲染 ──────────────────────────────────────────────────────────
void MEEFPipeline::_render_sraf() {
    int H = static_cast<int>(target_mask.rows());
    int W = static_cast<int>(target_mask.cols());
    if (sraf_cps.empty()) {
        sraf_mask = Eigen::MatrixXd::Zero(H, W);
        return;
    }

    ParametricDemo pd("BS", target_mask, _cfg.msaa_level);
    sraf_mask = pd.render_curve(sraf_cps);
}

// ── 5. 组装 initial_mask ──────────────────────────────────────────────────
void MEEFPipeline::_assemble_initial() {
    ParametricDemo pd(_cfg.curve_type, target_mask, _cfg.msaa_level);
    Eigen::MatrixXd main_gray = pd.render_curve(cps);
    initial_mask = main_gray + sraf_mask;
}

}  // namespace litho
