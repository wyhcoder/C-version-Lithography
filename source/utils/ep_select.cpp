#include "ep_select.h"
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

namespace litho {

// ── 构造 ─────────────────────────────────────────────────────────────────
EpSelect::EpSelect(const Eigen::MatrixXd& target_mask,
                   double mid_weight, double other_weight,
                   const std::string& pattern_name)
    : _target(target_mask),
      _mid_weight(mid_weight),
      _other_weight(other_weight),
      _pattern_name(pattern_name)
{}

// ── 转换：Eigen → cv::Mat ────────────────────────────────────────────────
cv::Mat EpSelect::to_cv8u(const Eigen::MatrixXd& M) {
    int rows = static_cast<int>(M.rows());
    int cols = static_cast<int>(M.cols());
    cv::Mat out(rows, cols, CV_8UC1);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c)
            out.at<uint8_t>(r, c) = static_cast<uint8_t>(
                std::clamp(M(r, c) * 255.0, 0.0, 255.0));
    return out;
}

// ── 转换：cv::Point → IPoints (y,x) ─────────────────────────────────────
IPoints EpSelect::cv_to_ipoints(const std::vector<cv::Point>& c){
    IPoints out;
    out.reserve(c.size());
    for (const auto& p : c) out.push_back({p.y, p.x});
    return out;
}

// ── 转换：IPoints → Contour (MatrixXd) ──────────────────────────────────
Contour EpSelect::ipoints_to_contour(const IPoints& pts) {
    int n = static_cast<int>(pts.size());
    Contour m(n, 2);
    for (int i = 0; i < n; ++i) {
        m(i, 0) = static_cast<double>(pts[i][0]);  // y
        m(i, 1) = static_cast<double>(pts[i][1]);  // x
    }
    return m;
}

// ── Bresenham ─────────────────────────────────────────────────────────────
IPoints EpSelect::bresenham_line(int y1, int x1, int y2, int x2) {
    IPoints pts;
    pts.push_back({y1, x1});
    int dy = std::abs(y2-y1), dx = std::abs(x2-x1);
    int sy = (y2 > y1) ? 1 : -1;
    int sx = (x2 > x1) ? 1 : -1;
    int err = dx - dy;
    while (y1 != y2 || x1 != x2) {
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x1 += sx; }
        if (e2 <  dx) { err += dx; y1 += sy; }
        pts.push_back({y1, x1});
    }
    return pts;
}

// ── sample_elements ───────────────────────────────────────────────────────
IPoints EpSelect::sample_elements(const IPoints& lst, int k, const std::string& mode){
    int step = std::max(1, (mode == "skip") ? k+1 : k);
    IPoints out;
    for (int i = 0; i < (int)lst.size(); i += step)
        out.push_back(lst[i]);
    return out;
}

// ── segment_orientation ──────────────────────────────────────────────────
std::string EpSelect::segment_orientation(const IPoint& s, const IPoint& e) {
    int dy = e[0]-s[0], dx = e[1]-s[1];
    if (dy == 0 && dx != 0) return "horizontal";
    if (dx == 0 && dy != 0) return "vertical";
    if (dx != 0 && dy != 0) return "diagonal";
    return "point";
}

// ── is_short_corner_connector ─────────────────────────────────────────────
bool EpSelect::is_short_corner_connector(const IPoints& pts, int seg_idx, int seg_len, int left_lim, int right_lim)
{
    if (seg_len < 1) return false;
    int n = static_cast<int>(pts.size());
    if (segment_orientation(pts[seg_idx], pts[(seg_idx+1)%n]) != "diagonal")
        return false;
    std::set<std::string> nb = {
        segment_orientation(pts[(seg_idx-1+n)%n], pts[seg_idx]),
        segment_orientation(pts[(seg_idx+1)%n],   pts[(seg_idx+2)%n])
    };
    if (nb != std::set<std::string>{"horizontal","vertical"}) return false;
    return left_lim > right_lim;
}

// ── fix_cps有对称的版图才用 ───────────────────────────────────────────────────────────────
ControlPoints EpSelect::fix_cps(const Eigen::MatrixXd& ls_mask, int k,
                                  const std::string& symmetry) const
{
    cv::Mat m8 = to_cv8u(ls_mask);
    std::vector<std::vector<cv::Point>> cv_c;
    cv::findContours(m8, cv_c, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    ControlPoints result;
    for (auto& c : cv_c)
        result.push_back(ipoints_to_contour(sample_elements(cv_to_ipoints(c), k, "skip")));

    if ((int)result.size() > 1) {
        int cy = (int)ls_mask.rows()/2, cx = (int)ls_mask.cols()/2;
        int h = (int)ls_mask.rows(),    w  = (int)ls_mask.cols();
        Contour sym(result[0].rows(), 2);
        for (int i = 0; i < (int)result[0].rows(); ++i) {
            double y = result[0](i,0), x = result[0](i,1);
            if      (symmetry == "center")     { sym(i,0)=2*cy-y; sym(i,1)=2*cx-x; }
            else if (symmetry == "left-right") { sym(i,0)=y;      sym(i,1)=2*cx-x; }
            else if (symmetry == "diagonal")   { sym(i,0)=h-1-x;  sym(i,1)=w-1-y;  }
            else throw std::invalid_argument("symmetry: center / left-right / diagonal");
        }
        result[1] = sym;
    }
    return result;
}

// ── fix_OPC_cps ───────────────────────────────────────────────────────────
ControlPoints EpSelect::fix_OPC_cps(const Eigen::MatrixXd& ls_mask, int k) const {
    cv::Mat m8 = to_cv8u(ls_mask);
    std::vector<std::vector<cv::Point>> cv_c;
    cv::findContours(m8, cv_c, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    ControlPoints result;
    for (auto& c : cv_c)
        result.push_back(ipoints_to_contour(sample_elements(cv_to_ipoints(c), k, "skip")));
    return result;
}

// ── extract_mask_control_points ──────────────────────────────────────────
ControlPoints EpSelect::extract_mask_control_points(int k,
                                                     const std::string& symmetry) const
{
    cv::Mat m8 = to_cv8u(_target);
    std::vector<std::vector<cv::Point>> cv_c;
    cv::findContours(m8, cv_c, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    ControlPoints result;
    for (auto& c : cv_c)
        result.push_back(ipoints_to_contour(sample_elements(cv_to_ipoints(c), k, "skip")));

    if (symmetry.empty() || (int)result.size() < 2) return result;

    int h=_target.rows(), w=_target.cols(), cy=h/2, cx=w/2;
    Contour sym(result[0].rows(), 2);
    for (int i = 0; i < (int)result[0].rows(); ++i) {
        double y=result[0](i,0), x=result[0](i,1);
        if      (symmetry=="center")     { sym(i,0)=2*cy-y; sym(i,1)=2*cx-x; }
        else if (symmetry=="left-right") { sym(i,0)=y;      sym(i,1)=2*cx-x; }
        else if (symmetry=="diagonal")   { sym(i,0)=h-1-x;  sym(i,1)=w-1-y;  }
        else throw std::invalid_argument("Unsupported symmetry: " + symmetry);
    }
    result[1] = sym;
    return result;
}

// ── select_eps_others ────────────────────────────────────────────────────
EpsResult EpSelect::select_eps_others(int interval_line, int interval_corner) const {
    cv::Mat m8 = to_cv8u(_target);
    std::vector<std::vector<cv::Point>> cv_c;
    cv::findContours(m8, cv_c, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    IPoints all_eps, core_pts;

    for (auto& cv_cont : cv_c) {
        auto pts = cv_to_ipoints(cv_cont);
        int num  = (int)pts.size();

        for (int i = 0; i < num; ++i) {
            auto seg = bresenham_line(pts[i][0], pts[i][1],
                                      pts[(i+1)%num][0], pts[(i+1)%num][1]);
            int seg_len = (int)seg.size();
            if (seg_len < 1) continue;

            int left_lim  = std::min(std::max(0,interval_corner), seg_len-1);
            int right_lim = std::max(0, seg_len-1-std::min(std::max(0,interval_corner),seg_len-1));

            if (is_short_corner_connector(pts, i, seg_len, left_lim, right_lim))
                continue;

            int mid_idx = seg_len / 2;
            if (left_lim > right_lim) left_lim = right_lim = mid_idx;

            std::set<int> idx_set;
            idx_set.insert(mid_idx);
            for (int idx=mid_idx-interval_line; idx>=left_lim;  idx-=interval_line) idx_set.insert(idx);
            for (int idx=mid_idx+interval_line; idx<=right_lim; idx+=interval_line) idx_set.insert(idx);
            // 只在边界点离最近的采样点 >= interval_line 时才加入，避免两点过近
            if (left_lim < seg_len && left_lim >= 0) {
                int nearest = *idx_set.lower_bound(left_lim);
                if (std::abs(nearest - left_lim) >= interval_line)
                    idx_set.insert(left_lim);
            }
            if (right_lim >= 0 && right_lim < seg_len) {
                auto it = idx_set.upper_bound(right_lim);
                int nearest = (it != idx_set.begin()) ? *std::prev(it) : -999;
                if (std::abs(right_lim - nearest) >= interval_line)
                    idx_set.insert(right_lim);
            }

            std::vector<int> sampled(idx_set.begin(), idx_set.end());
            sampled.erase(std::remove_if(sampled.begin(), sampled.end(),
                [&](int v){ return v<0 || v>=seg_len; }), sampled.end());
            std::sort(sampled.begin(), sampled.end());
            if (sampled.empty()) continue;

            for (int idx : sampled) all_eps.push_back(seg[idx]);

            // core 70% 筛选
            int lmg = mid_idx-sampled.front(), rmg = sampled.back()-mid_idx;
            std::vector<int> final_idx;
            if (sampled.front()!=sampled.back() && lmg<interval_line && rmg<interval_line) {
                final_idx = {sampled.front(), sampled.back()};
            } else if ((int)sampled.size() <= 2) {
                final_idx = sampled;
            } else {
                std::vector<int> interior(sampled.begin()+1, sampled.end()-1);
                if (interior.empty()) {
                    final_idx = {sampled.front(), sampled.back()};
                } else {
                    int tc = std::max(1, std::min((int)interior.size(),
                        (int)std::round(interior.size()*0.7)));
                    if (tc%2==0) tc = (tc<(int)interior.size()) ? tc+1 : tc-1;
                    auto it = std::find(sampled.begin(),sampled.end(),mid_idx);
                    int mp = (it!=sampled.end()) ? (int)(it-sampled.begin())
                             : (int)(std::min_element(sampled.begin(),sampled.end(),
                                 [&](int a,int b){ return std::abs(a-mid_idx)<std::abs(b-mid_idx);}
                               )-sampled.begin());
                    int mi = std::max(0, std::min((int)interior.size()-1, mp-1));
                    int r0 = std::max(0, mi-(tc-1)/2);
                    int r1 = std::min((int)interior.size(), r0+tc);
                    r0 = std::max(0, r1-tc);
                    final_idx.push_back(sampled.front());
                    for (int v : std::vector<int>(interior.begin()+r0, interior.begin()+r1))
                        final_idx.push_back(v);
                    final_idx.push_back(sampled.back());
                    std::vector<int> dedup; std::set<int> seen;
                    for (int v : final_idx) if (seen.insert(v).second) dedup.push_back(v);
                    final_idx = dedup;
                }
            }
            for (int idx : final_idx) core_pts.push_back(seg[idx]);
        }
    }

    int M = (int)all_eps.size();
    if (M == 0) return { Eigen::MatrixXd(0,2), Eigen::RowVectorXd(0), Eigen::RowVectorXd(0) };

    std::set<std::pair<int,int>> core_set;
    for (auto& p : core_pts) core_set.insert({p[0],p[1]});

    Eigen::MatrixXd eps(M, 2);
    Eigen::RowVectorXd w_epe(M), w_meef(M);
    for (int i = 0; i < M; ++i) {
        eps(i,0) = all_eps[i][0];
        eps(i,1) = all_eps[i][1];
        bool ic = core_set.count({all_eps[i][0],all_eps[i][1]}) > 0;
        w_meef(i) = ic ? _mid_weight : _other_weight;
        w_epe(i)  = ic ? 1.0 : 0.0;
    }
    return {eps, w_epe, w_meef};
}

// ── select_eps_via ────────────────────────────────────────────────────────
EpsResult EpSelect::select_eps_via(int r) const {
    cv::Mat m8 = to_cv8u(_target);
    std::vector<std::vector<cv::Point>> cv_c;
    cv::findContours(m8, cv_c, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    IPoints all_eps;
    std::vector<double> w_epe_v, w_meef_v;

    for (auto& cv_cont : cv_c) {
        auto pts = cv_to_ipoints(cv_cont);
        int len  = (int)pts.size();
        for (int i = 0; i < len; ++i) {
            auto seg = bresenham_line(pts[i][0], pts[i][1],
                                      pts[(i+1)%len][0], pts[(i+1)%len][1]);
            if ((int)seg.size() < 4) continue;
            if (i < 4) {
                static const int off[4][2] = {{r,r},{-r,r},{-r,-r},{r,-r}};
                all_eps.push_back({seg[0][0]+off[i][0], seg[0][1]+off[i][1]});
                w_epe_v.push_back(0.0); w_meef_v.push_back(_mid_weight);
            }
            int n = (int)seg.size();
            for (auto [pt, is_m, wm] : std::vector<std::tuple<IPoint,bool,double>>{
                {seg[n/4],     false, _other_weight},
                {seg[n/3],     false, _other_weight},
                {seg[n/2],     true,  _mid_weight  },
                {seg[n-1-n/3], false, _other_weight},
                {seg[n-1-n/4], false, _other_weight}})
            {
                all_eps.push_back(pt);
                w_epe_v.push_back(is_m ? 1.0 : 0.0);
                w_meef_v.push_back(wm);
            }
        }
    }

    int M = (int)all_eps.size();
    Eigen::MatrixXd eps(M, 2);
    Eigen::RowVectorXd w_epe(M), w_meef(M);
    for (int i = 0; i < M; ++i) {
        eps(i,0)=all_eps[i][0]; eps(i,1)=all_eps[i][1];
        w_epe(i)=w_epe_v[i]; w_meef(i)=w_meef_v[i];
    }
    return {eps, w_epe, w_meef};
}

// ── remove_adjacent_close_points ──────────────────────────────────────────
ControlPoints EpSelect::remove_adjacent_close_points(
    const ControlPoints& contours, double threshold)
{
    ControlPoints cleaned;
    for (const auto& contour : contours) {
        int n = (int)contour.rows();
        if (n == 0) { cleaned.push_back(Contour(0,2)); continue; }
        if (n == 1) { cleaned.push_back(contour);      continue; }

        std::vector<int> keep;
        for (int i = 0; i < n; ++i) {
            int prev = (i-1+n)%n;
            double dy = contour(i,0)-contour(prev,0);
            double dx = contour(i,1)-contour(prev,1);
            if (std::sqrt(dy*dy+dx*dx) >= threshold) keep.push_back(i);
        }
        Contour out((int)keep.size(), 2);
        for (int k = 0; k < (int)keep.size(); ++k)
            out.row(k) = contour.row(keep[k]);
        cleaned.push_back(out);
    }
    return cleaned;
}

}  // namespace litho
