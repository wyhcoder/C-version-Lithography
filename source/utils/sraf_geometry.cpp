#include "sraf_geometry.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace litho {
namespace {

/**
 * @brief 将任意单通道 cv::Mat 统一转换成 0/255 的二值图。
 *
 * 只要输入像素值大于 0，输出对应位置就是 255，否则为 0。这个函数用于
 * 消除输入可能是 0/1、0/255、float 或 double 等格式差异，保证后面的
 * OpenCV 形态学、连通域和骨架化函数接收到统一的 CV_8UC1 数据。
 *
 * @param input 任意深度的单通道矩阵；空矩阵允许传入。
 * @return CV_8UC1 二值图；输入为空时返回空 cv::Mat。
 * @throws std::invalid_argument 输入包含多个通道时抛出。
 */
cv::Mat normalized_binary(const cv::Mat& input) {
    if (input.empty()) return {};
    if (input.channels() != 1) {
        throw std::invalid_argument(
            "SrafGeometry: binary mask must have exactly one channel");
    }

    cv::Mat binary;
    cv::compare(input, 0, binary, cv::CMP_GT);
    return binary;
}

/**
 * @brief 生成半径为 radius 的离散圆盘形结构元素。
 *
 * 圆盘内满足 x²+y²<=radius² 的位置设为 1，主要供膨胀和开运算使用。
 * radius=0 时返回 1x1 结构元素，相当于不改变图形。
 *
 * @param radius 圆盘半径，单位为像素。
 * @return CV_8UC1 的圆盘形结构元素。
 * @throws std::invalid_argument radius 小于 0 时抛出。
 */
cv::Mat disk_kernel(int radius) {
    if (radius < 0) {
        throw std::invalid_argument(
            "SrafGeometry: morphology radius must be non-negative");
    }
    if (radius == 0) return cv::Mat::ones(1, 1, CV_8UC1);

    const int size = 2 * radius + 1;
    cv::Mat kernel = cv::Mat::zeros(size, size, CV_8UC1);
    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            if (x * x + y * y <= radius * radius) {
                kernel.at<uchar>(y + radius, x + radius) = 1;
            }
        }
    }
    return kernel;
}

/**
 * @brief 按照“先 y、再 x”的顺序比较两个整数像素坐标。
 *
 * 该顺序用来稳定地选择骨架起点和邻居，确保同一输入每次得到完全相同的
 * 路径方向，不受容器遍历顺序影响。
 */
bool point_less(const cv::Point& lhs, const cv::Point& rhs) {
    return lhs.y != rhs.y ? lhs.y < rhs.y : lhs.x < rhs.x;
}

}  // namespace

/**
 * @brief 从包含主图形和 SRAF 的完整掩模中分离二者。
 *
 * 处理步骤：
 * 1. 按 foreground_threshold 将 full_mask 二值化；
 * 2. 对前景做 8 连通域标记；
 * 3. 统计每个连通域与 target_mask 的真实重叠像素数；
 * 4. 重叠比例达到 overlap_ratio 的连通域判为主图形，其余判为 SRAF；
 * 5. 如果没有任何连通域匹配主图形，则用膨胀后的 target 区域兜底拆分。
 *
 * 返回结果保留 full_mask 中的原始灰度，而不是只返回 0/1 二值值。
 *
 * @param full_mask 包含主图形和 SRAF 的完整灰度掩模。
 * @param target_mask 只包含主图形的参考目标掩模。
 * @param foreground_threshold full_mask 的前景阈值。
 * @param target_threshold target_mask 的前景阈值。
 * @param overlap_ratio 连通域判为主图形所需的最小重叠面积比例。
 * @param fallback_dilate_radius 兜底拆分时 target 的膨胀半径，单位 pixel。
 * @return main_mask 和 sraf_mask，两者尺寸及数据类型与输入一致。
 */
SrafSplitResult SrafGeometry::split_main_and_sraf(
    const Eigen::MatrixXd& full_mask,
    const Eigen::MatrixXd& target_mask,
    double foreground_threshold,
    double target_threshold,
    double overlap_ratio,
    int fallback_dilate_radius) {
    if (full_mask.rows() != target_mask.rows() ||
        full_mask.cols() != target_mask.cols()) {
        throw std::invalid_argument(
            "SrafGeometry::split_main_and_sraf: mask dimensions must match");
    }
    if (!full_mask.allFinite() || !target_mask.allFinite()) {
        throw std::invalid_argument(
            "SrafGeometry::split_main_and_sraf: masks must contain finite values");
    }
    if (!std::isfinite(foreground_threshold) ||
        !std::isfinite(target_threshold)) {
        throw std::invalid_argument(
            "SrafGeometry::split_main_and_sraf: thresholds must be finite");
    }
    if (!std::isfinite(overlap_ratio) || overlap_ratio < 0.0 ||
        overlap_ratio > 1.0) {
        throw std::invalid_argument(
            "SrafGeometry::split_main_and_sraf: overlap_ratio must be in [0, 1]");
    }
    if (fallback_dilate_radius < 0) {
        throw std::invalid_argument(
            "SrafGeometry::split_main_and_sraf: fallback radius must be non-negative");
    }

    const int height = static_cast<int>(full_mask.rows());
    const int width = static_cast<int>(full_mask.cols());
    SrafSplitResult result{
        Eigen::MatrixXd::Zero(height, width),
        Eigen::MatrixXd::Zero(height, width)};
    if (height == 0 || width == 0) return result;

    // Step 1：生成 full mask 前景图和 target 参考前景图。
    const cv::Mat foreground = to_binary_mask(full_mask, foreground_threshold);
    const cv::Mat target_binary = to_binary_mask(target_mask, target_threshold);
    if (cv::countNonZero(foreground) == 0) return result;

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        foreground, labels, stats, centroids, 8, CV_32S);

    // Step 2：逐像素统计每个完整连通域与 target 的重叠面积。
    std::vector<int> overlap_count(
        static_cast<std::size_t>(component_count), 0);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int label = labels.at<int>(y, x);
            if (label > 0 && target_binary.at<uchar>(y, x) != 0) {
                ++overlap_count[static_cast<std::size_t>(label)];
            }
        }
    }

    std::vector<uchar> is_main(static_cast<std::size_t>(component_count), 0); //强制类型转换为无符号整型
    bool matched_main = false;
    // Step 3：根据重叠像素数/连通域面积判断该连通域的归属。
    for (int label = 1; label < component_count; ++label) {
        const int area = stats.at<int>(label, cv::CC_STAT_AREA);
        const int overlap = overlap_count[static_cast<std::size_t>(label)];
        const int required_overlap = std::max(
            1, static_cast<int>(overlap_ratio * static_cast<double>(area)));
        if (overlap > 0 && overlap >= required_overlap) {
            is_main[static_cast<std::size_t>(label)] = 1;
            matched_main = true;
        }
    }

    // Step 4：正常情况直接按连通域标签复制原始灰度值。
    if (matched_main) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                const int label = labels.at<int>(y, x);
                if (label == 0) continue;
                if (is_main[static_cast<std::size_t>(label)] != 0) {
                    result.main_mask(y, x) = full_mask(y, x);
                } else {
                    result.sraf_mask(y, x) = full_mask(y, x);
                }
            }
        }
        return result;
    }

    // Step 5：完全没有匹配到主图形时，用 target 膨胀区域进行兜底拆分。
    cv::Mat main_region;
    if (fallback_dilate_radius == 0) {
        main_region = target_binary.clone();
    } else {
        cv::dilate(target_binary, main_region,
                   disk_kernel(fallback_dilate_radius));
    }

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (foreground.at<uchar>(y, x) == 0) continue;
            if (main_region.at<uchar>(y, x) != 0) {
                result.main_mask(y, x) = full_mask(y, x);
            } else {
                result.sraf_mask(y, x) = full_mask(y, x);
            }
        }
    }
    return result;
}

/**
 * @brief 将 Eigen::MatrixXd 按阈值转换成 OpenCV 二值图。
 *
 * mask(y,x)>threshold 时输出 255，否则输出 0。该转换是 Eigen 数值矩阵
 * 进入 OpenCV 图像算法的统一入口。
 *
 * @param mask 输入的 double 矩阵。
 * @param threshold 前景判断阈值。
 * @return 与 mask 同尺寸的 CV_8UC1 二值图。
 */
cv::Mat SrafGeometry::to_binary_mask(
    const Eigen::MatrixXd& mask,
    double threshold) {
    if (!mask.allFinite() || !std::isfinite(threshold)) {
        throw std::invalid_argument(
            "SrafGeometry::to_binary_mask: mask and threshold must be finite");
    }

    const int height = static_cast<int>(mask.rows());
    const int width = static_cast<int>(mask.cols());
    cv::Mat binary(height, width, CV_8UC1, cv::Scalar(0));
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (mask(y, x) > threshold) binary.at<uchar>(y, x) = 255;
        }
    }
    return binary;
}

/**
 * @brief 清理 SRAF 二值图，为骨架化准备稳定的连通域。
 *
 * 首先把输入统一为 0/255；opening_radius>0 时执行圆盘结构元素开运算，
 * 去除细小毛刺；随后通过 8 连通域面积过滤，删除小于
 * minimum_component_area 的噪声区域。
 *
 * @param binary_sraf SRAF 二值图，可以是任意单通道深度。
 * @param opening_radius 开运算圆盘半径；0 表示跳过开运算。
 * @param minimum_component_area 最小保留面积，单位为像素数。
 * @return 清理后的 CV_8UC1 二值图，前景为 255。
 */
cv::Mat SrafGeometry::clean_binary_sraf(
    const cv::Mat& binary_sraf,
    int opening_radius,
    int minimum_component_area) {
    if (opening_radius < 0) {
        throw std::invalid_argument(
            "SrafGeometry::clean_binary_sraf: opening radius must be non-negative");
    }
    if (minimum_component_area < 1) {
        throw std::invalid_argument(
            "SrafGeometry::clean_binary_sraf: minimum component area must be positive");
    }

    cv::Mat cleaned = normalized_binary(binary_sraf);
    if (cleaned.empty()) return cleaned;

    if (opening_radius > 0) {
        cv::morphologyEx(cleaned, cleaned, cv::MORPH_OPEN,
                         disk_kernel(opening_radius));
    }

    cv::Mat labels; // 标签图
    cv::Mat stats;  // 保存连通区域的统计信息
    cv::Mat centroids; // 每个连通区域的中心坐标
    const int component_count = cv::connectedComponentsWithStats(
        cleaned, labels, stats, centroids, 8, CV_32S);

    cv::Mat filtered = cv::Mat::zeros(cleaned.size(), CV_8UC1);
    std::vector<uchar> keep(static_cast<std::size_t>(component_count), 0);
    for (int label = 1; label < component_count; ++label) {
        if (stats.at<int>(label, cv::CC_STAT_AREA) >= minimum_component_area) {
            keep[static_cast<std::size_t>(label)] = 1;
        }
    }
    for (int y = 0; y < labels.rows; ++y) {
        for (int x = 0; x < labels.cols; ++x) {
            const int label = labels.at<int>(y, x);
            if (label > 0 && keep[static_cast<std::size_t>(label)] != 0) {
                filtered.at<uchar>(y, x) = 255;
            }
        }
    }
    return filtered;
}

/**
 * @brief 将有宽度的 SRAF 二值区域细化成一像素宽中心骨架。
 *
 * 内部调用 OpenCV contrib 的 cv::ximgproc::thinning，并明确选择
 * Zhang-Suen 算法。该算法反复删除不影响拓扑连通性的边界像素，直到图形
 * 收敛为中心骨架。输入和输出都统一为 CV_8UC1、0/255。
 *
 * @param binary_sraf 清理后的 SRAF 二值图。
 * @return 一像素宽骨架；空输入或全背景输入直接返回对应空图。
 */
cv::Mat SrafGeometry::skeletonize(const cv::Mat& binary_sraf) {
    cv::Mat binary = normalized_binary(binary_sraf);
    if (binary.empty() || cv::countNonZero(binary) == 0) return binary;

    cv::Mat skeleton;
    cv::ximgproc::thinning(
        binary,
        skeleton,
        cv::ximgproc::THINNING_ZHANGSUEN);
    return skeleton;
}

/**
 * @brief 将单个 SRAF 的无序骨架像素排列成连续的中心路径。
 *
 * 核心方法是把骨架看成无向图：每个骨架像素是一个节点，相邻像素之间是
 * 一条边。函数随后统计节点度数并区分三种情况：
 * - 单个节点：点型 SRAF，直接作为合法中心几何保留；
 * - 两个度数为 1 的端点：开放路径，从固定端点依次遍历；
 * - 没有端点且所有节点度数为 2：闭合环路。
 *
 * 若出现度数大于 2 的分叉、异常端点数量或未遍历全部像素，返回
 * valid=false，并通过 diagnostic 说明原因，避免参数曲线静默丢失分支。
 *
 * @param single_component_skeleton 只包含一个连通域的一像素骨架图。
 * @return 有序坐标、开闭状态、端点/分叉统计和诊断信息。
 */
SkeletonPathResult SrafGeometry::order_skeleton_path(
    const cv::Mat& single_component_skeleton) {
    SkeletonPathResult result;
    const cv::Mat skeleton = normalized_binary(single_component_skeleton);
    if (skeleton.empty() || cv::countNonZero(skeleton) == 0) {
        result.diagnostic = "骨架为空";
        return result;
    }

    std::vector<cv::Point> pixels;
    pixels.reserve(static_cast<std::size_t>(cv::countNonZero(skeleton)));
    std::unordered_map<int, int> node_at;
    for (int y = 0; y < skeleton.rows; ++y) {
        for (int x = 0; x < skeleton.cols; ++x) {
            if (skeleton.at<uchar>(y, x) == 0) continue;
            const int index = static_cast<int>(pixels.size());
            pixels.emplace_back(x, y);
            node_at.emplace(y * skeleton.cols + x, index); // 将二维坐标变为一维顺序
        }
    }

    if (pixels.size() == 1) {
        result.points.emplace_back(
            static_cast<double>(pixels.front().x),
            static_cast<double>(pixels.front().y));
        result.valid = true;
        result.diagnostic = "合法单点骨架（点型 SRAF）";
        return result;
    }

    // Step 1：为每个骨架像素建立邻接表。
    std::vector<std::vector<int>> adjacency(pixels.size());
    const std::array<cv::Point, 8> offsets = {
        cv::Point{-1, -1}, cv::Point{0, -1}, cv::Point{1, -1},
        cv::Point{-1,  0},                    cv::Point{1,  0},
        cv::Point{-1,  1}, cv::Point{0,  1}, cv::Point{1,  1}};

    const auto has_pixel = [&](int x, int y) {
        if (x < 0 || x >= skeleton.cols || y < 0 || y >= skeleton.rows) {
            return false;
        }
        return skeleton.at<uchar>(y, x) != 0;
    };

    for (std::size_t i = 0; i < pixels.size(); ++i) {
        const cv::Point point = pixels[i];
        for (const auto& offset : offsets) {
            const int nx = point.x + offset.x;
            const int ny = point.y + offset.y;
            if (!has_pixel(nx, ny)) continue;

            // 当对角像素之间已经能通过水平/竖直邻居相连时，不再添加对角
            // 捷径，避免普通 90 度拐角被误判成三节点闭环或分叉。
            if (offset.x != 0 && offset.y != 0 &&
                (has_pixel(point.x + offset.x, point.y) ||
                 has_pixel(point.x, point.y + offset.y))) {
                continue;
            }

            const auto found = node_at.find(ny * skeleton.cols + nx);
            if (found != node_at.end()) adjacency[i].push_back(found->second);
        }
        std::sort(adjacency[i].begin(), adjacency[i].end(),
                  [&](int lhs, int rhs) {
                      return point_less(pixels[lhs], pixels[rhs]);
                  });
        adjacency[i].erase(
            std::unique(adjacency[i].begin(), adjacency[i].end()),
            adjacency[i].end());
    }

    // Step 2：通过节点度数统计端点和分叉点。
    std::vector<int> endpoints;
    for (std::size_t i = 0; i < adjacency.size(); ++i) {
        if (adjacency[i].size() == 1) endpoints.push_back(static_cast<int>(i));
        if (adjacency[i].size() > 2) ++result.branchpoint_count;
    }
    result.endpoint_count = static_cast<int>(endpoints.size());

    if (result.branchpoint_count != 0) {
        const auto edge_key = [](int lhs, int rhs) {
            return std::pair<int, int>{std::min(lhs, rhs), std::max(lhs, rhs)};
        };
        std::set<std::pair<int, int>> visited_edges;
        std::size_t total_edges = 0; // std::size_t 无符号整型数据
        for (const auto& neighbors : adjacency) total_edges += neighbors.size();
        total_edges /= 2;

        for (int start = 0; start < static_cast<int>(pixels.size()); ++start) {
            if (adjacency[start].size() == 2) continue;
            for (const int neighbor : adjacency[start]) {
                if (visited_edges.count(edge_key(start, neighbor)) != 0) continue;
                std::vector<cv::Point2d> edge;
                edge.emplace_back(pixels[start].x, pixels[start].y);
                int previous = start;
                int current = neighbor;
                while (true) {
                    if (!visited_edges.insert(edge_key(previous, current)).second) { // 如果这个健已经存在那么。second返回false 也就是重复了
                        result.diagnostic = "分叉边遍历时重复访问骨架连接";
                        result.edges.clear();
                        return result;
                    }
                    edge.emplace_back(pixels[current].x, pixels[current].y);
                    if (adjacency[current].size() != 2) break;
                    const int next = adjacency[current][0] == previous ? adjacency[current][1] : adjacency[current][0];
                    previous = current;
                    current = next;
                }
                result.edges.push_back(std::move(edge));
            }
        }
        if (visited_edges.size() != total_edges || result.edges.empty()) {
            result.diagnostic = "分叉骨架未能完整拆分为无分叉边";
            result.edges.clear();
            return result;
        }
        result.valid = true;
        result.diagnostic = "合法分叉骨架：" + std::to_string(result.edges.size()) + " 条边";
        return result;
    }

    // Step 3：根据端点数量确定开放/闭合类型，并选择确定性的起点。
    int start = -1;
    if (endpoints.size() == 2) {
        result.closed = false;
        start = point_less(pixels[endpoints[0]], pixels[endpoints[1]])
            ? endpoints[0] : endpoints[1];
    } else if (endpoints.empty()) {
        const bool all_degree_two = std::all_of(
            adjacency.begin(), adjacency.end(),
            [](const auto& neighbors) { return neighbors.size() == 2; });
        if (!all_degree_two) {
            result.diagnostic = "骨架没有两个端点，且不是合法闭环";
            return result;
        }
        result.closed = true;
        start = 0;  // pixels 按 (y,x) 扫描，0 即确定性的左上起点。
    } else {
        std::ostringstream message;
        message << "开放骨架应有两个端点，当前检测到 "
                << endpoints.size() << " 个";
        result.diagnostic = message.str();
        return result;
    }

    std::vector<uchar> visited(pixels.size(), 0);
    std::vector<int> ordered_indices;
    ordered_indices.reserve(pixels.size());
    int previous = -1;
    int current = start;

    // Step 4：沿唯一的未访问邻居前进，把像素图转换成有序路径。
    while (true) {
        if (visited[static_cast<std::size_t>(current)] != 0) {
            result.diagnostic = "路径排序时提前重复访问骨架像素";
            return result;
        }
        visited[static_cast<std::size_t>(current)] = 1;
        ordered_indices.push_back(current);

        int next = -1;
        for (const int neighbor : adjacency[static_cast<std::size_t>(current)]) {
            if (neighbor == previous) continue;
            if (result.closed && neighbor == start &&
                ordered_indices.size() == pixels.size()) {
                next = start;
                break;
            }
            if (visited[static_cast<std::size_t>(neighbor)] == 0) {
                next = neighbor;
                break;
            }
        }

        if (next < 0) break;
        if (result.closed && next == start) break;
        previous = current;
        current = next;
    }

    // Step 5：必须访问该连通域的全部像素，否则认为路径排序失败。
    if (ordered_indices.size() != pixels.size()) {
        std::ostringstream message;
        message << "路径只访问了 " << ordered_indices.size() << "/"
                << pixels.size() << " 个骨架像素";
        result.diagnostic = message.str();
        return result;
    }

    result.points.reserve(ordered_indices.size());
    for (const int index : ordered_indices) {
        result.points.emplace_back(
            static_cast<double>(pixels[index].x),
            static_cast<double>(pixels[index].y));
    }
    result.valid = true;
    result.diagnostic = result.closed ? "合法闭合骨架" : "合法开放骨架";
    return result;
}

/**
 * @brief 沿有序骨架路径按固定像素下标间隔直接选取控制点。
 *
 * 依次选择 ordered_path[0]、ordered_path[interval]、
 * ordered_path[2*interval]……作为控制点，不进行线性插值，因此返回的每个
 * 坐标都来自原始整数骨架。开放路径如果最后一个端点没有刚好被间隔选中，
 * 会额外加入该端点；闭合路径不重复首点。
 *
 * @param ordered_path 已按连接关系排序的中心路径，坐标约定为 (x,y)。
 * @param interval 相邻控制点之间间隔的骨架点数量，必须大于 0。
 * @param closed 是否为闭合路径。
 * @return 直接从 ordered_path 选出的控制点。
 */
std::vector<cv::Point2d> SrafGeometry::sample_skeleton_points(
    const std::vector<cv::Point2d>& ordered_path,
    int interval,
    bool closed) {
    if (interval <= 0) {
        throw std::invalid_argument(
            "SrafGeometry::sample_skeleton_points: interval must be positive");
    }
    if (ordered_path.empty()) return {};

    std::vector<cv::Point2d> controls;
    controls.reserve(
        (ordered_path.size() + static_cast<std::size_t>(interval) - 1) /
        static_cast<std::size_t>(interval));

    for (std::size_t index = 0; index < ordered_path.size();index += static_cast<std::size_t>(interval)) {
        controls.push_back(ordered_path[index]);
    }

    // 开放曲线必须保留最后一个端点，即使它不在固定间隔下标上。
    const std::size_t last_index = ordered_path.size() - 1;
    if (!closed && last_index % static_cast<std::size_t>(interval) != 0) {
        controls.push_back(ordered_path.back());
    }
    return controls;
}

/**
 * @brief 执行完整的 SRAF 几何提取流程。
 *
 * 调用顺序为：
 * full mask 拆分 -> SRAF 二值化 -> 形态学/面积清理 -> 骨架化 ->
 * 按 SRAF 连通域拆分骨架 -> 路径排序 -> 按固定下标间隔提取控制点。
 *
 * 每个 SRAF 连通域独立生成 SrafCenterlineGeometry，其中包含面积、外接框、
 * 重心、独立骨架、有序路径和间隔控制点。拓扑不合法的连通域仍会保留在返回
 * 结果中，但 path.valid=false 且不会生成控制点，便于调用方诊断。
 *
 * @param full_mask 包含主图形和 SRAF 的完整灰度 mask。
 * @param target_mask 只包含主图形的参考 target mask。
 * @param config 拆分阈值、清理参数和控制点间隔。
 * @return 完整的拆分结果、清理图、总骨架及每个 SRAF 的几何信息。
 */
SrafGeometryResult SrafGeometry::extract(
    const Eigen::MatrixXd& full_mask,
    const Eigen::MatrixXd& target_mask,
    const SrafGeometryConfig& config) {
    if (config.opening_radius < 0 || config.minimum_component_area < 1 ||
        config.control_point_interval <= 0) {
        throw std::invalid_argument(
            "SrafGeometry::extract: invalid cleanup or sampling configuration");
    }

    SrafGeometryResult result;
    // Step 1：从完整 LSM mask 中拆出主图形与 SRAF 灰度图。
    result.split = split_main_and_sraf(
        full_mask, target_mask,
        config.foreground_threshold,
        config.target_threshold,
        config.overlap_ratio,
        config.fallback_dilate_radius);

    // Step 2：二值化、去噪并提取所有 SRAF 的一像素骨架。
    const cv::Mat binary_sraf = to_binary_mask(
        result.split.sraf_mask, config.foreground_threshold);
    result.cleaned_sraf_mask = clean_binary_sraf(
        binary_sraf,
        config.opening_radius,
        config.minimum_component_area);
    result.skeleton_mask = skeletonize(result.cleaned_sraf_mask);

    if (result.cleaned_sraf_mask.empty() ||
        cv::countNonZero(result.cleaned_sraf_mask) == 0) {
        return result;
    }

    // Step 3：在清理后的 SRAF 上重新标记连通域，保证一块对应一条中心线。
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    const int component_count = cv::connectedComponentsWithStats(
        result.cleaned_sraf_mask, labels, stats, centroids, 8, CV_32S);
    result.centerlines.reserve(static_cast<std::size_t>(component_count - 1));

    // Step 4：逐连通域收集属性、独立骨架、有序路径和间隔控制点。
    for (int label = 1; label < component_count; ++label) {
        SrafCenterlineGeometry geometry;
        geometry.component_id = label;
        geometry.area = stats.at<int>(label, cv::CC_STAT_AREA);
        geometry.bounding_box = cv::Rect(
            stats.at<int>(label, cv::CC_STAT_LEFT),
            stats.at<int>(label, cv::CC_STAT_TOP),
            stats.at<int>(label, cv::CC_STAT_WIDTH),
            stats.at<int>(label, cv::CC_STAT_HEIGHT));
        geometry.centroid = cv::Point2d(
            centroids.at<double>(label, 0),
            centroids.at<double>(label, 1));

        geometry.skeleton_mask = cv::Mat::zeros(
            result.skeleton_mask.size(), CV_8UC1);
        for (int y = 0; y < labels.rows; ++y) {
            for (int x = 0; x < labels.cols; ++x) {
                if (labels.at<int>(y, x) == label &&
                    result.skeleton_mask.at<uchar>(y, x) != 0) {
                    geometry.skeleton_mask.at<uchar>(y, x) = 255;
                }
            }
        }

        geometry.path = order_skeleton_path(geometry.skeleton_mask);
        if (geometry.path.valid) {
            if (geometry.path.edges.empty()) {
                geometry.control_points = sample_skeleton_points(geometry.path.points, config.control_point_interval, geometry.path.closed);
            } else {
                geometry.edge_control_points.reserve(geometry.path.edges.size());
                for (const auto& edge : geometry.path.edges) {
                    auto controls = sample_skeleton_points(edge, config.control_point_interval, false);
                    geometry.control_points.insert(geometry.control_points.end(), controls.begin(), controls.end());
                    geometry.edge_control_points.push_back(std::move(controls));
                }
            }
        }
        result.centerlines.push_back(std::move(geometry));
    }
    return result;
}

}  // namespace litho
