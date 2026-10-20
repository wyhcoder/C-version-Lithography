#include "ep_select.h"
#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>

namespace litho {

namespace {

// Marching Squares 的临时坐标。target 外补了一圈 0，并把 padded 坐标乘以 2 后存成整数：
// 原图坐标 (0,0.5) -> padded 坐标 (1,1.5) -> GridPoint (2,3)。这样可精确比较整数/半整数端点，避免用 double 作为 map 键。
using GridPoint = std::array<int, 2>;
// 最终输出使用原图局部零基 (y,x) 坐标，单位为 pixel；坐标可以是 0.5，也可以是弧长插值得到的其他小数。
using SubpixelPoint = std::array<double, 2>;
using GridSegment = std::pair<GridPoint, GridPoint>; // 一个方格内的有向等值线小段：from -> to。
using SubpixelContour = std::vector<SubpixelPoint>;  // 已按首尾相接顺序排列的一条浮点轮廓。

enum class SegmentOrientation { Horizontal, Vertical, Diagonal, Point };

struct OrientedSegment {
    SubpixelContour points;
    SegmentOrientation orientation;
};

double point_distance(const SubpixelPoint& a, const SubpixelPoint& b) {
    // 轮廓弧长使用像素坐标中的欧氏距离，水平/垂直一步为 1 pixel，斜向按 hypot(dy,dx) 计算。
    return std::hypot(a[0] - b[0], a[1] - b[1]);
}

bool is_close(double a, double b, double atol = 1e-8) {
    // 弧长经过加减和插值后存在浮点误差，因此不能直接使用 a == b 判断是否为同一个采样距离。
    return std::abs(a - b) <= atol + 1e-5 * std::abs(b);
}

double round_ten_digits(double value) {
    // 这里只对“弧长参数”去除数值噪声，不会把最终 EP 坐标取整。
    return std::round(value * 1e10) / 1e10;
}

// 在 target 外隐式补一圈 0，并用 Marching Squares 扫描 target=0.5 的等值线。
// target 是二值图：相邻像素中心的值若由 1 变为 0，线性插值的 0.5 交点恰好位于两中心的中点，因此初始轮廓顶点是整数或半整数。
std::vector<GridSegment> marching_squares_segments(const Eigen::MatrixXd& target) {
    const int height = static_cast<int>(target.rows());
    const int width = static_cast<int>(target.cols());
    std::vector<GridSegment> segments;

    // padded 坐标 [1,height]×[1,width] 对应原图 [0,height-1]×[0,width-1]；范围外统一视为背景 0。
    auto foreground = [&](int padded_y, int padded_x) {
        return padded_y >= 1 && padded_y <= height && padded_x >= 1 && padded_x <= width && target(padded_y - 1, padded_x - 1) > 0.5;
    };

    // 保留 from/to 方向，后面会按有向端点把相邻方格产生的小线段拼成完整闭合轮廓。
    auto append = [&](const GridPoint& from, const GridPoint& to) { segments.emplace_back(from, to); };
    // 补零后的图共有 (height+1)×(width+1) 个 2×2 方格；(y,x) 是当前方格左上角的 padded 像素中心坐标。
    for (int y = 0; y < height + 1; ++y) {
        for (int x = 0; x < width + 1; ++x) {
            // 四个角的位权如下，前景为 1、背景为 0，因此 square_case 的范围是 [0,15]：
            //   TL×1 ---- TR×2
            //     |        |
            //   BL×4 ---- BR×8
            const int square_case = static_cast<int>(foreground(y, x)) + 2 * static_cast<int>(foreground(y, x + 1))
                                  + 4 * static_cast<int>(foreground(y + 1, x)) + 8 * static_cast<int>(foreground(y + 1, x + 1));
            // 0000 是全背景，1111 是全前景，方格内都没有 0.5 等值线。
            if (square_case == 0 || square_case == 15) continue;

            // 四个点分别是方格上、下、左、右边的中点。坐标已乘 2，所以半像素位置由奇数精确表示。
            const GridPoint top{2 * y, 2 * x + 1};
            const GridPoint bottom{2 * y + 2, 2 * x + 1};
            const GridPoint left{2 * y + 1, 2 * x};
            const GridPoint right{2 * y + 1, 2 * x + 2};

            // 根据四角的 0/1 组合连接边中点。case 5/10 产生贯穿方格的直线；其余普通 case 产生一个转角小段。
            // case 6/9 是对角前景的歧义情形，这里按 Python skimage.find_contours(..., fully_connected="low") 的规则拆成两段。
            switch (square_case) {
                case 1: append(top, left); break;                         // 0001：仅 TL 为前景。
                case 2: append(right, top); break;                        // 0010：仅 TR 为前景。
                case 3: append(right, left); break;                       // 0011：上侧两个点为前景。
                case 4: append(left, bottom); break;                      // 0100：仅 BL 为前景。
                case 5: append(top, bottom); break;                       // 0101：左侧两个点为前景。
                case 6: append(right, top); append(left, bottom); break;  // 0110：TR、BL 对角前景。
                case 7: append(right, bottom); break;                     // 0111：仅 BR 为背景。
                case 8: append(bottom, right); break;                     // 1000：仅 BR 为前景。
                case 9: append(top, left); append(bottom, right); break;  // 1001：TL、BR 对角前景。
                case 10: append(bottom, top); break;                      // 1010：右侧两个点为前景。
                case 11: append(bottom, left); break;                     // 1011：仅 BL 为背景。
                case 12: append(left, right); break;                      // 1100：下侧两个点为前景。
                case 13: append(top, right); break;                       // 1101：仅 TR 为背景。
                case 14: append(left, top); break;                        // 1110：仅 TL 为背景。
                default: break;
            }
        }
    }
    return segments;
}

// 将每个方格产生的有向小线段按公共端点连接成完整轮廓。
// 连接顺序与 scikit-image 的首尾端点规则一致，因此 C++ 和 Python 得到的轮廓方向、起点顺序保持一致。
std::vector<SubpixelContour> assemble_subpixel_contours(const std::vector<GridSegment>& segments) {
    std::vector<std::deque<GridPoint>> contours; // 每个 deque 是一条正在生长的轮廓，允许从首端或尾端追加点。
    std::vector<bool> active;                    // 两条轮廓合并后，被吸收的旧轮廓标为 false，最终不再输出。
    std::map<GridPoint, int> starts;             // 起点坐标 -> 以该点开头的轮廓编号。
    std::map<GridPoint, int> ends;               // 终点坐标 -> 以该点结尾的轮廓编号。

    // 查询某个端点属于哪条轮廓，并立刻删除旧映射；轮廓延长或合并后会写入新的首尾端点映射。
    auto take = [](std::map<GridPoint, int>& endpoints, const GridPoint& point) -> std::optional<int> {
        const auto it = endpoints.find(point);
        if (it == endpoints.end()) return std::nullopt;
        const int contour_id = it->second;
        endpoints.erase(it);
        return contour_id;
    };

    for (const auto& [from, to] : segments) {
        if (from == to) continue;
        // 如果 to 是已有轮廓的起点，新线段可以接到该轮廓前面，所以记为 tail_id。
        const std::optional<int> tail_id = take(starts, to);
        // 如果 from 是已有轮廓的终点，新线段可以接到该轮廓后面，所以记为 head_id。
        const std::optional<int> head_id = take(ends, from);

        if (tail_id && head_id) {
            if (*tail_id == *head_id) {
                // from/to 同时接到同一条轮廓，说明最后一段把轮廓首尾闭合；把 to 作为闭合终点追加进去。
                contours[*head_id].push_back(to);
            } else if (*tail_id > *head_id) {
                // 新线段把两条不同轮廓连接起来：head 的尾端接到 tail 的首端，合并后保留编号较小的 head。
                auto& head = contours[*head_id];
                const auto& tail = contours[*tail_id];
                head.insert(head.end(), tail.begin(), tail.end());
                active[*tail_id] = false;
                starts[head.front()] = *head_id;
                ends[head.back()] = *head_id;
            } else {
                // 与上一分支连接关系相同，但这里保留编号较小的 tail，并把 head 插到 tail 前面。
                auto& tail = contours[*tail_id];
                const auto& head = contours[*head_id];
                tail.insert(tail.begin(), head.begin(), head.end());
                active[*head_id] = false;
                starts[tail.front()] = *tail_id;
                ends[tail.back()] = *tail_id;
            }
        } else if (!tail_id && !head_id) {
            // 两个端点都没有出现过，这条小线段成为一条新轮廓的起点。
            const int contour_id = static_cast<int>(contours.size());
            contours.push_back({from, to});
            active.push_back(true);
            starts[from] = contour_id;
            ends[to] = contour_id;
        } else if (tail_id) {
            // to 接到已有轮廓首端：把 from 插到最前面，并用 from 更新轮廓起点。
            auto& tail = contours[*tail_id];
            tail.push_front(from);
            starts[from] = *tail_id;
        } else {
            // from 接到已有轮廓尾端：把 to 追加到最后，并用 to 更新轮廓终点。
            auto& head = contours[*head_id];
            head.push_back(to);
            ends[to] = *head_id;
        }
    }

    std::vector<SubpixelContour> result;
    for (int contour_id = 0; contour_id < static_cast<int>(contours.size()); ++contour_id) {
        // 被其他轮廓吸收的临时记录不再输出。
        if (!active[contour_id]) continue;
        SubpixelContour contour;
        contour.reserve(contours[contour_id].size() + 1);
        // GridPoint 保存的是“二倍 padded 坐标”，因此先乘 0.5 还原 padded 坐标，再减 1 去掉外圈 padding。
        for (const GridPoint& point : contours[contour_id]) contour.push_back({0.5 * point[0] - 1.0, 0.5 * point[1] - 1.0});
        // 开放链的首尾若尚未重复，则显式追加起点，使后续拆段能处理跨数组首尾的最后一条边。
        if (!contour.empty() && point_distance(contour.front(), contour.back()) > 1e-9) contour.push_back(contour.front());
        if (contour.size() >= 2) result.push_back(std::move(contour));
    }
    return result;
}

SegmentOrientation subpixel_orientation(const SubpixelPoint& delta) {
    // delta 的顺序同样是 (dy,dx)：dy≈0 为水平边，dx≈0 为垂直边，两者都非零为斜向边。
    const double dy = delta[0];
    const double dx = delta[1];
    if (std::abs(dy) <= 1e-9 && std::abs(dx) > 1e-9) return SegmentOrientation::Horizontal;
    if (std::abs(dx) <= 1e-9 && std::abs(dy) > 1e-9) return SegmentOrientation::Vertical;
    if (std::abs(dx) > 1e-9 && std::abs(dy) > 1e-9) return SegmentOrientation::Diagonal;
    return SegmentOrientation::Point;
}

// 将闭合轮廓按水平、垂直、斜向拆成连续段，使每条几何边单独按中点、角点保护距离和固定间隔选 EP。
// 如果同一方向的连续段刚好被轮廓数组首尾切开，最后再把它们合并回来。
std::vector<OrientedSegment> split_subpixel_segments(const SubpixelContour& contour) {
    std::vector<OrientedSegment> segments;
    SubpixelContour current{contour.front()};     // 当前正在收集的同方向点序列。
    std::optional<SegmentOrientation> current_orientation; // 第一次遇到有效边后才确定方向。

    for (int i = 1; i < static_cast<int>(contour.size()); ++i) {
        const SubpixelPoint delta{contour[i][0] - contour[i - 1][0], contour[i][1] - contour[i - 1][1]};
        if (std::hypot(delta[0], delta[1]) <= 1e-12) continue;
        const SegmentOrientation orientation = subpixel_orientation(delta);
        if (!current_orientation) {
            current_orientation = orientation;
        } else if (orientation != *current_orientation) {
            // 方向发生变化，说明到达拐角：保存上一段，并让拐角点同时成为新段的第一个点。
            if (current.size() >= 2) segments.push_back({current, *current_orientation});
            current = {contour[i - 1]};
            current_orientation = orientation;
        }
        current.push_back(contour[i]);
    }
    if (current.size() >= 2 && current_orientation) segments.push_back({current, *current_orientation});

    // 闭合轮廓的数组起点可能落在一条直边中间，导致同一条边被拆成“最后一段+第一段”；这里恢复成完整的一段。
    if (segments.size() > 1 && segments.front().orientation == segments.back().orientation
        && point_distance(segments.back().points.back(), segments.front().points.front()) <= 1e-9) {
        SubpixelContour merged = segments.back().points;
        merged.insert(merged.end(), std::next(segments.front().points.begin()), segments.front().points.end());
        std::vector<OrientedSegment> reordered;
        reordered.reserve(segments.size() - 1);
        reordered.push_back({std::move(merged), segments.front().orientation});
        for (int i = 1; i + 1 < static_cast<int>(segments.size()); ++i) reordered.push_back(std::move(segments[i]));
        segments = std::move(reordered);
    }
    return segments;
}

std::vector<double> cumulative_lengths(const SubpixelContour& points) {
    // cumulative[i] 表示从本段起点 points[0] 沿轮廓走到 points[i] 的累计弧长，单位为 pixel。
    std::vector<double> cumulative(points.size(), 0.0);
    for (int i = 1; i < static_cast<int>(points.size()); ++i) cumulative[i] = cumulative[i - 1] + point_distance(points[i - 1], points[i]);
    return cumulative;
}

SubpixelPoint point_at_distance(const SubpixelContour& points, const std::vector<double>& cumulative, double distance) {
    // distance 是从当前连续段起点出发的目标弧长；落在端点之外时直接返回对应端点。
    if (distance <= 0.0) return points.front();
    if (distance >= cumulative.back()) return points.back();
    // upper_bound 找到第一个累计弧长大于 distance 的点，因此目标位于 points[index] 与 points[index+1] 之间。
    const auto upper = std::upper_bound(cumulative.begin(), cumulative.end(), distance);
    const int index = std::clamp(static_cast<int>(upper - cumulative.begin()) - 1, 0, static_cast<int>(cumulative.size()) - 2);
    const double segment_length = cumulative[index + 1] - cumulative[index];
    if (segment_length <= 1e-12) return points[index];
    // t 是目标点在当前两轮廓顶点之间的比例；对 y/x 分别线性插值，所以结果不必落在整数或半整数位置。
    const double t = (distance - cumulative[index]) / segment_length;
    return {points[index][0] + t * (points[index + 1][0] - points[index][0]),
            points[index][1] + t * (points[index + 1][1] - points[index][1])};
}

// 输入：segment_length、interval_line、interval_corner 的单位都是 pixel。
// 输出：sampled 保存需要输出/可视化的全部候选弧长；active 保存其中真正参与 EPE 的弧长。
// 采样规则逐项对应 Python _sample_segment_distances。
std::pair<std::vector<double>, std::vector<double>> sample_segment_distances(
    double segment_length, double interval_line, double interval_corner) {
    // interval 是沿直线段相邻主采样点的期望间距；corner 是希望与两端拐角保持的距离。
    const double interval = std::max(interval_line, 1e-9);
    const double corner = std::max(interval_corner, 0.0);
    // 可采样范围为 [left_limit,right_limit]；若线段比两侧 corner 总和还短，两端限制在中点相遇。
    double left_limit = std::min(corner, segment_length);
    double right_limit = std::max(0.0, segment_length - std::min(corner, segment_length));
    const double mid_distance = 0.5 * segment_length;
    if (left_limit > right_limit) left_limit = right_limit = mid_distance;

    // 无论线段长短，先把中点和两侧角点保护位置放入候选集合。
    std::vector<double> distances{mid_distance, left_limit, right_limit};
    // 再从中点向左、向右每隔 interval_line 增加候选点，且不越过角点保护位置。
    for (double distance = mid_distance - interval; distance >= left_limit - 1e-9; distance -= interval) {
        distances.push_back(std::max(left_limit, distance));
    }
    for (double distance = mid_distance + interval; distance <= right_limit + 1e-9; distance += interval) {
        distances.push_back(std::min(right_limit, distance));
    }

    // set 同时完成排序和去重；十位小数舍入只用于消除重复弧长的浮点噪声。
    std::set<double> sampled_set;
    for (double distance : distances) sampled_set.insert(round_ten_digits(distance));
    std::vector<double> sampled(sampled_set.begin(), sampled_set.end());
    if (sampled.empty()) return {sampled, sampled};

    // 记录最左/最右候选点到中点的距离，用于判断端部候选点是否与中点靠得太近。
    const double left_mid_gap = mid_distance - sampled.front();
    const double right_mid_gap = sampled.back() - mid_distance;
    // final_distances 是“准备判定为 active”的初选集合；sampled 本身仍保留全部候选点并最终输出。
    std::vector<double> final_distances;
    if (!is_close(sampled.front(), sampled.back()) && left_mid_gap < interval && right_mid_gap < interval) {
        // 短线段的两个端部候选都离中点不足一个 interval，先只保留两端参与后续 active 判定，中点会在最后强制加入。
        final_distances = {sampled.front(), sampled.back()};
    } else if (sampled.size() <= 2) {
        // 候选不超过两个时无需再围绕中点裁剪。
        final_distances = sampled;
    } else {
        // 首尾是靠近拐角的候选点，interior 是两者之间的中间候选。
        const std::vector<double> interior(sampled.begin() + 1, sampled.end() - 1);
        if (interior.empty()) {
            final_distances = {sampled.front(), sampled.back()};
        } else {
            int target_count = static_cast<int>(interior.size());
            // 尽量选取奇数个内部点，使中点左右数量对称；无法增加时就减少一个。
            if (target_count % 2 == 0) {
                target_count = target_count < static_cast<int>(interior.size()) ? target_count + 1 : target_count - 1;
            }
            std::vector<double> selected;
            if (target_count > 0) {
                // 找到 sampled 中最靠近几何中点的位置，再以它为中心截取连续的 target_count 个内部候选。
                int mid_position = 0;
                for (int i = 1; i < static_cast<int>(sampled.size()); ++i) {
                    if (std::abs(sampled[i] - mid_distance) < std::abs(sampled[mid_position] - mid_distance)) mid_position = i;
                }
                const int mid_interior = std::clamp(mid_position - 1, 0, static_cast<int>(interior.size()) - 1);
                const int radius = (target_count - 1) / 2;
                int start = std::max(0, mid_interior - radius);
                int end = std::min(static_cast<int>(interior.size()), start + target_count);
                start = std::max(0, end - target_count);
                selected.assign(interior.begin() + start, interior.begin() + end);
            }
            // 端部候选仍保留，后面根据“距中点是否至少一个 interval”决定它是不是 active。
            final_distances.push_back(sampled.front());
            final_distances.insert(final_distances.end(), selected.begin(), selected.end());
            final_distances.push_back(sampled.back());
        }
    }

    std::set<double> active_set;
    for (double distance : final_distances) {
        // 中点和内部点直接 active；首尾端点只有在离中点至少一个 interval 时才 active，避免短边上 EPE 点过密。
        const bool keep = is_close(distance, mid_distance) || (is_close(distance, sampled.front()) && left_mid_gap >= interval)
                       || (is_close(distance, sampled.back()) && right_mid_gap >= interval)
                       || (!is_close(distance, sampled.front()) && !is_close(distance, sampled.back()));
        if (keep) active_set.insert(round_ten_digits(distance));
    }
    // 无论上面的短边规则如何裁剪，每个有效方向段都至少保留中点作为一个 active EP。
    bool contains_mid = false;
    for (double distance : active_set) contains_mid = contains_mid || is_close(distance, mid_distance, 1e-7);
    if (!contains_mid) active_set.insert(round_ten_digits(mid_distance));
    return {sampled, std::vector<double>(active_set.begin(), active_set.end())};
}

bool contains_distance(const std::vector<double>& values, double target) {
    // sampled/active 来自浮点弧长计算，用容差判断某个候选距离是否属于 active 集合。
    return std::any_of(values.begin(), values.end(), [&](double value) { return is_close(target, value, 1e-7); });
}

enum class EdgeAxis { Vertical, Horizontal };

struct EdgeBoundarySegment {
    EdgeAxis axis;
    double fixed_coord; // 垂直边为 x，水平边为 y，单位 pixel。
    double start_coord; // 沿边方向的闭区间起点，单位 pixel。
    double end_coord;   // 沿边方向的闭区间终点，单位 pixel。
};

std::vector<double> sample_edge_coordinates(double start, double end, double sample_step_px, double corner_margin_px) {
    const double a = start + corner_margin_px;
    const double b = end - corner_margin_px;
    if (b < a) return {};
    const double length = b - a;
    if (length <= 1e-12) return {0.5 * (a + b)};
    const int count = std::max(static_cast<int>(std::floor(length / sample_step_px + 1e-12)) + 1, 1);
    if (count == 1) return {0.5 * (a + b)};
    std::vector<double> coordinates(count);
    for (int i = 0; i < count; ++i) coordinates[i] = a + (b - a) * static_cast<double>(i) / static_cast<double>(count - 1);
    return coordinates;
}

} // namespace

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

// ── select_eps_subpixel：全芯片 ILT 使用的 Python 对齐亚像素选点 ──────────
EpsResult EpSelect::select_eps_subpixel(double interval_line, double interval_corner) const {
    // 第一步：从二值 target 的 0.5 等值线生成方格内小线段，再按公共端点拼成原图局部坐标中的闭合亚像素轮廓。
    const auto contours = assemble_subpixel_contours(marching_squares_segments(_target));
    std::vector<SubpixelPoint> all_points; // 全部候选 EP，坐标顺序为 (y,x)，单位为 pixel。
    std::vector<bool> active_flags;        // 与 all_points 一一对应，true 表示该点参与 EPE。

    for (const SubpixelContour& contour : contours) {
        // 第二步：按水平、垂直、斜向拆段，使每条几何边独立执行中点和等间隔采样。
        for (const OrientedSegment& segment : split_subpixel_segments(contour)) {
            if (segment.points.size() < 2) continue;
            // cumulative.back() 是当前连续段的总弧长，单位为 pixel。
            const std::vector<double> cumulative = cumulative_lengths(segment.points);
            const double segment_length = cumulative.back();
            if (segment_length <= 1e-9) continue;
            // 轴对齐图形拐角处常出现长度不超过 1 pixel 的斜向连接段；跳过它，避免把拐角连接线当成独立评价边。
            if (segment.orientation == SegmentOrientation::Diagonal && segment_length <= 1.0 + 1e-9) continue;

            // 第三步：先在弧长坐标中确定全部候选位置和 active 位置，再把每个弧长插值回二维亚像素坐标。
            const auto [sampled_distances, active_distances] = sample_segment_distances(segment_length, interval_line, interval_corner);
            for (double distance : sampled_distances) {
                all_points.push_back(point_at_distance(segment.points, cumulative, distance));
                active_flags.push_back(contains_distance(active_distances, distance));
            }
        }
    }

    // 第四步：写入统一输出。eps 的每行是局部零基 (y,x) 浮点像素坐标，后续 EPE 会在该位置双线性采样空中像。
    const int point_count = static_cast<int>(all_points.size());
    Eigen::MatrixXd eps(point_count, 2);
    Eigen::RowVectorXd weight_epe(point_count);
    Eigen::RowVectorXd weight_meef(point_count);
    for (int i = 0; i < point_count; ++i) {
        eps(i, 0) = all_points[i][0];
        eps(i, 1) = all_points[i][1];
        // active 点计入 EPE，并使用 mid_weight；inactive 候选点不计入 EPE，但保留 other_weight 供 MEEF/可视化使用。
        weight_epe(i) = active_flags[i] ? 1.0 : 0.0;
        weight_meef(i) = active_flags[i] ? _mid_weight : _other_weight;
    }
    return {eps, weight_epe, weight_meef};
}

EpsResult EpSelect::select_eps_edge_sampling(double sample_step_px, double corner_margin_px,
                                             bool include_image_boundary, double threshold) const {
    const int height = static_cast<int>(_target.rows()), width = static_cast<int>(_target.cols());
    std::vector<EdgeBoundarySegment> segments;

    // 与 edge_sampling.py 一致：先按列间分界扫描垂直边，并把法向相同的连续像素合并为一段。
    for (int x_boundary = 0; x_boundary <= width; ++x_boundary) {
        if (!include_image_boundary && (x_boundary == 0 || x_boundary == width)) continue;
        int y = 0;
        while (y < height) {
            const bool left_inside = x_boundary > 0 && _target(y, x_boundary - 1) >= threshold;
            const bool right_inside = x_boundary < width && _target(y, x_boundary) >= threshold;
            const int sign = left_inside == right_inside ? 0 : (left_inside ? 1 : -1);
            if (sign == 0) {
                ++y;
                continue;
            }
            const int start_y = y;
            while (y + 1 < height) {
                const bool next_left = x_boundary > 0 && _target(y + 1, x_boundary - 1) >= threshold;
                const bool next_right = x_boundary < width && _target(y + 1, x_boundary) >= threshold;
                const int next_sign = next_left == next_right ? 0 : (next_left ? 1 : -1);
                if (next_sign != sign) break;
                ++y;
            }
            segments.push_back({EdgeAxis::Vertical, static_cast<double>(x_boundary) - 0.5,
                                static_cast<double>(start_y) - 0.5, static_cast<double>(y) + 0.5});
            ++y;
        }
    }

    // 再按行间分界扫描水平边；坐标仍先按 Python 的 (x,y) 几何定义，写入 EpsResult 时转换为 (y,x)。
    for (int y_boundary = 0; y_boundary <= height; ++y_boundary) {
        if (!include_image_boundary && (y_boundary == 0 || y_boundary == height)) continue;
        int x = 0;
        while (x < width) {
            const bool top_inside = y_boundary > 0 && _target(y_boundary - 1, x) >= threshold;
            const bool bottom_inside = y_boundary < height && _target(y_boundary, x) >= threshold;
            const int sign = top_inside == bottom_inside ? 0 : (top_inside ? 1 : -1);
            if (sign == 0) {
                ++x;
                continue;
            }
            const int start_x = x;
            while (x + 1 < width) {
                const bool next_top = y_boundary > 0 && _target(y_boundary - 1, x + 1) >= threshold;
                const bool next_bottom = y_boundary < height && _target(y_boundary, x + 1) >= threshold;
                const int next_sign = next_top == next_bottom ? 0 : (next_top ? 1 : -1);
                if (next_sign != sign) break;
                ++x;
            }
            segments.push_back({EdgeAxis::Horizontal, static_cast<double>(y_boundary) - 0.5,
                                static_cast<double>(start_x) - 0.5, static_cast<double>(x) + 0.5});
            ++x;
        }
    }

    std::vector<SubpixelPoint> points;
    for (const EdgeBoundarySegment& segment : segments) {
        const std::vector<double> coordinates = sample_edge_coordinates(
            segment.start_coord, segment.end_coord, sample_step_px, corner_margin_px);
        for (double coordinate : coordinates) {
            if (segment.axis == EdgeAxis::Vertical) points.push_back({coordinate, segment.fixed_coord});
            else points.push_back({segment.fixed_coord, coordinate});
        }
    }

    Eigen::MatrixXd eps(points.size(), 2);
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        eps(i, 0) = points[i][0];
        eps(i, 1) = points[i][1];
    }
    // Python 返回的 valid_mask 对这些非 padding 点全部为 true，因此 C++ 中每个输出点都参与 EPE 和 MEEF。
    return {eps, Eigen::RowVectorXd::Ones(points.size()), Eigen::RowVectorXd::Ones(points.size())};
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
