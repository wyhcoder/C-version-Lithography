#include "sraf_printing_repair.h"

#include <opencv2/imgproc.hpp>
// OpenCV 5 将距离类型移入 geometry；条件包含兼容现有 OpenCV 4/5 构建方式。
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>
#endif
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace litho {
namespace {

// 输入按 (y,x) 存储的二维向量，返回按 (x,y) 顺序定义的叉积，供射线求交与面积计算共用。
double cross_xy(const Eigen::Vector2d& a, const Eigen::Vector2d& b) {
    return a(1) * b(0) - a(0) * b(1);
}

// 周期向心 CR 在控制点处的解析切线旋转后得到单位法线，再用材料内外探测确定朝向。
// 输入为同一块的采样边界和一个控制点所在轮廓，坐标单位 pixel；退化点返回零向量。
Eigen::Vector2d control_inward_normal(const Polygon& controls, int point, const Polygons& curves) {
    const int count = static_cast<int>(controls.rows());
    const Eigen::Vector2d position = controls.row(point).transpose();
    const Eigen::Vector2d previous = controls.row((point + count - 1) % count).transpose();
    const Eigen::Vector2d next = controls.row((point + 1) % count).transpose();
    const double before_interval = std::sqrt((position - previous).norm());
    const double after_interval = std::sqrt((next - position).norm());
    if (before_interval < 1e-10 || after_interval < 1e-10) return Eigen::Vector2d::Zero();
    const Eigen::Vector2d tangent = (position - previous) / before_interval -
        (next - previous) / (before_interval + after_interval) + (next - position) / after_interval;
    if (tangent.norm() < 1e-10) return Eigen::Vector2d::Zero();
    Eigen::Vector2d normal(tangent(1), -tangent(0));
    normal.normalize();
    const bool plus_inside = inside_sraf_material(curves, position + 0.02 * normal);
    const bool minus_inside = inside_sraf_material(curves, position - 0.02 * normal);
    if (plus_inside == minus_inside) return Eigen::Vector2d::Zero();
    if (!plus_inside) normal = -normal;
    return normal;
}

// 从曲线控制点沿材料内法线与闭合边界求交，取首次离开材料的距离作为局部厚度，单位 pixel。
double normal_material_width(const Polygons& curves, const Eigen::Vector2d& position_yx, const Eigen::Vector2d& inward_normal_yx) {
    double width = std::numeric_limits<double>::infinity();
    // 解 position+t*normal=edge_start+u*segment_delta，t 为像素距离，u 位于线段的 [0,1]。
    // 跳过平行边和射线起点附近的交点，保留沿内法线第一次离开材料的位置。
    for (const auto& curve : curves) {
        for (int i = 0; i < curve.rows(); ++i) {
            const Eigen::Vector2d edge_start = curve.row(i).transpose();
            const Eigen::Vector2d segment_delta_yx = curve.row((i + 1) % curve.rows()).transpose() - edge_start;
            const double denominator = cross_xy(inward_normal_yx, segment_delta_yx);
            if (std::abs(denominator) < 1e-12) continue;
            const double ray_distance_px = cross_xy(edge_start - position_yx, segment_delta_yx) / denominator;
            const double segment_fraction = cross_xy(edge_start - position_yx, inward_normal_yx) / denominator;
            if (ray_distance_px > 1e-4 && segment_fraction >= -1e-9 && segment_fraction <= 1.0 + 1e-9) {
                width = std::min(width, ray_distance_px);
            }
        }
    }
    return std::isfinite(width) ? width : 0.0;
}

struct SrafSection {
    Eigen::Vector2d center_yx = Eigen::Vector2d::Zero(); // 最宽截面圆心，整图零基 (y,x) 浮点像素坐标。
    double diameter_px = 0.0; // 最大内切圆直径，用于限定局部厚度和候选点范围，单位 pixel。
};

// 输入本块闭合 CR 采样曲线，输出 1/8 pixel 网格估计的最宽截面，供多组点对统一限定候选范围。
SrafSection find_widest_sraf_section(const Polygons& curves) {
    SrafSection section;
    // 先在整图浮点像素坐标中求包围盒，外加 2 pixel 空白以建立材料到背景的距离。
    Eigen::Vector2d minimum_yx = curves.front().colwise().minCoeff().transpose();
    Eigen::Vector2d maximum_yx = curves.front().colwise().maxCoeff().transpose();
    for (const auto& curve : curves) {
        minimum_yx = minimum_yx.cwiseMin(curve.colwise().minCoeff().transpose());
        maximum_yx = maximum_yx.cwiseMax(curve.colwise().maxCoeff().transpose());
    }
    // 仅在本块的包围盒内以 1/8 pixel 网格估计局部厚度；填充按奇偶规则保留孔洞。
    constexpr int samples_per_pixel = 8;
    const Eigen::Vector2d origin_yx = minimum_yx.array().floor().matrix() - Eigen::Vector2d::Constant(2.0);
    // origin_yx 是整图坐标，grid_rows/grid_cols 为局部细网格点数，末端的 +1 纳入边界网格点。
    const int grid_rows = static_cast<int>(std::ceil(maximum_yx(0) - origin_yx(0) + 2.0)) * samples_per_pixel + 1;
    const int grid_cols = static_cast<int>(std::ceil(maximum_yx(1) - origin_yx(1) + 2.0)) * samples_per_pixel + 1;
    cv::Mat material_grid(grid_rows, grid_cols, CV_8UC1, cv::Scalar(0));
    for (const auto& curve : curves) {
        std::vector<cv::Point> grid_points;
        for (int row = 0; row < curve.rows(); ++row) {
            const int y = static_cast<int>(std::lround((curve(row, 0) - origin_yx(0)) * samples_per_pixel));
            const int x = static_cast<int>(std::lround((curve(row, 1) - origin_yx(1)) * samples_per_pixel));
            grid_points.emplace_back(x, y);
        }
        cv::Mat ring_grid(grid_rows, grid_cols, CV_8UC1, cv::Scalar(0));
        cv::fillPoly(ring_grid, std::vector<std::vector<cv::Point>>{grid_points}, cv::Scalar(255));
        cv::bitwise_xor(material_grid, ring_grid, material_grid);
    }
    // 距离变换的输入是几何材料覆盖，返回以细网格步长计的半径，再换算回整图 pixel。
    cv::Mat boundary_distances;
    cv::distanceTransform(material_grid, boundary_distances, cv::DIST_L2, cv::DIST_MASK_PRECISE);
    double radius_samples = 0.0;
    cv::Point center_sample;
    cv::minMaxLoc(boundary_distances, nullptr, &radius_samples, nullptr, &center_sample);
    const double radius_px = radius_samples / samples_per_pixel;
    section.diameter_px = 2.0 * radius_px;
    section.center_yx = origin_yx + Eigen::Vector2d(center_sample.y, center_sample.x) / samples_per_pixel;
    return section;
}

} // namespace

bool inside_sraf_material(const Polygons& curves, const Eigen::Vector2d& position_yx) {
    bool inside = false;
    for (const auto& curve : curves) {
        bool inside_ring = false;
        // 水平射线穿越边界的次数决定内外，逐轮廓翻转用于扣除外边界内的孔洞。
        int previous_point = static_cast<int>(curve.rows()) - 1;
        for (int current_point = 0; current_point < curve.rows(); ++current_point) {
            const double yi = curve(current_point, 0), xi = curve(current_point, 1);
            const double yj = curve(previous_point, 0), xj = curve(previous_point, 1);
            if ((yi > position_yx(0)) != (yj > position_yx(0))) {
                const double crossing_x = xi + (position_yx(0) - yi) * (xj - xi) / (yj - yi);
                if (position_yx(1) < crossing_x) inside_ring = !inside_ring;
            }
            previous_point = current_point;
        }
        if (inside_ring) inside = !inside;
    }
    return inside;
}

bool sraf_contour_self_intersects(const Polygon& curve) {
    // 只比较不相邻的边 AB/CD，先排除包围盒无交集的情况，再检查严格交叉。
    const int count = static_cast<int>(curve.rows());
    for (int i = 0; i < count; ++i) {
        const Eigen::Vector2d a = curve.row(i).transpose(), b = curve.row((i + 1) % count).transpose();
        for (int j = i + 2; j < count; ++j) {
            if (i == 0 && j == count - 1) continue;
            const Eigen::Vector2d c = curve.row(j).transpose(), d = curve.row((j + 1) % count).transpose();
            if ((a.cwiseMin(b).array() > c.cwiseMax(d).array()).any() ||
                (c.cwiseMin(d).array() > a.cwiseMax(b).array()).any()) continue;
            const double ab_c = cross_xy(b - a, c - a), ab_d = cross_xy(b - a, d - a);
            const double cd_a = cross_xy(d - c, a - c), cd_b = cross_xy(d - c, b - c);
            if (ab_c * ab_d < -1e-14 && cd_a * cd_b < -1e-14) return true;
        }
    }
    return false;
}

std::vector<SrafRegion> group_sraf_regions(const ControlPoints& controls, const ParametricDemo& renderer) {
    const Polygons curves = renderer.get_curve_points(controls);
    const int contour_count = static_cast<int>(curves.size());
    // 鞋带公式的叉积和是两倍有向面积，取绝对值后只用于比较轮廓大小，单位为 pixel² 的两倍。
    std::vector<double> doubled_areas(contour_count, 0.0);
    std::vector<int> hole_owners(contour_count, -1); // 孔洞所属外轮廓的全局编号，-1 表示当前轮廓本身为外边界。
    for (int contour = 0; contour < contour_count; ++contour) {
        const auto& curve = curves[contour];
        for (int point = 0; point < curve.rows(); ++point) {
            doubled_areas[contour] += cross_xy(curve.row(point).transpose(), curve.row((point + 1) % curve.rows()).transpose());
        }
        doubled_areas[contour] = std::abs(doubled_areas[contour]);
    }
    for (int contour = 0; contour < contour_count; ++contour) {
        if (curves[contour].rows() < 4 || doubled_areas[contour] < 1e-10) throw std::invalid_argument("Invalid closed SRAF contour");
        // 只建立“外边界包含一层孔洞”的归属，每条孔洞边界对应唯一外轮廓。
        for (int other = 0; other < contour_count; ++other) {
            if (doubled_areas[other] <= doubled_areas[contour]) continue;
            if (!inside_sraf_material({curves[other]}, curves[contour].row(0).transpose())) continue;
            if (hole_owners[contour] >= 0) throw std::invalid_argument("Only one level of SRAF holes is supported");
            hole_owners[contour] = other;
        }
    }
    std::vector<SrafRegion> regions;
    for (int outer = 0; outer < contour_count; ++outer) {
        if (hole_owners[outer] >= 0) continue;
        SrafRegion region;
        region.outer_contour = outer;
        region.contour_indices.push_back(outer);
        ControlPoints region_controls{controls[outer]};
        for (int hole = 0; hole < contour_count; ++hole) {
            if (hole_owners[hole] != outer) continue;
            region.contour_indices.push_back(hole);
            region_controls.push_back(controls[hole]);
        }
        region.initial_mask = renderer.render_curve_even_odd(region_controls);
        regions.push_back(std::move(region));
    }
    return regions;
}

double sraf_printing_loss(const Eigen::MatrixXd& aerial, const cv::Mat& window, double target) {
    if (window.type() != CV_8UC1 || window.rows != aerial.rows() || window.cols != aerial.cols() ||
        !aerial.allFinite() || !std::isfinite(target)) throw std::invalid_argument("Invalid printing-loss input");
    double sum = 0.0;
    int pixel_count = 0;
    for (int y = 0; y < aerial.rows(); ++y) {
        for (int x = 0; x < aerial.cols(); ++x) {
            if (window.at<unsigned char>(y, x) == 0) continue;
            const double excess = std::max(aerial(y, x) - target, 0.0);
            sum += excess * excess;
            ++pixel_count;
        }
    }
    if (pixel_count == 0) throw std::invalid_argument("Printing-loss window is empty");
    return sum / pixel_count;
}

std::vector<SrafShrinkPoint> sraf_control_normals(const ControlPoints& controls, const ParametricDemo& renderer,
                                                const SrafRegion& region) {
    ControlPoints local_controls;
    for (int contour : region.contour_indices) {
        if (contour < 0 || contour >= controls.size() || controls[contour].rows() < 4 || !controls[contour].allFinite()) {
            throw std::invalid_argument("Invalid SRAF region control indices");
        }
        local_controls.push_back(controls[contour]);
    }
    const Polygons curves = renderer.get_curve_points(local_controls, 800);
    std::vector<SrafShrinkPoint> points;
    for (int contour = 0; contour < local_controls.size(); ++contour) {
        for (int point = 0; point < local_controls[contour].rows(); ++point) {
            const Eigen::Vector2d normal = control_inward_normal(local_controls[contour], point, curves);
            if (normal.squaredNorm() == 0.0) continue;
            SrafShrinkPoint control;
            control.contour = region.contour_indices[contour];
            control.point = point;
            control.inward_yx = normal;
            const Eigen::Vector2d position = local_controls[contour].row(point).transpose();
            control.width_px = normal_material_width(curves, position, normal);
            points.push_back(control);
        }
    }
    return points;
}

int select_printing_sraf_from_aerial(const std::vector<SrafRegion>& regions, const Eigen::MatrixXd& aerial,
                                    double threshold, const cv::Mat& background_window, const std::vector<bool>& finished) {
    if (regions.size() != finished.size() || background_window.type() != CV_8UC1 || !std::isfinite(threshold) ||
        background_window.rows != aerial.rows() || background_window.cols != aerial.cols() || !aerial.allFinite()) {
        throw std::invalid_argument("Invalid aerial/SRAF selection input");
    }
    for (const auto& region : regions) {
        if (region.initial_mask.rows() != aerial.rows() || region.initial_mask.cols() != aerial.cols()) {
            throw std::invalid_argument("Invalid SRAF region mask");
        }
    }
    // 按强度筛选像素后查覆盖归属，最高可归属强度初始设为阈值，恰好相等时不触发。
    int selected_region_index = -1;
    double highest_intensity = threshold;
    for (int y = 0; y < aerial.rows(); ++y) {
        for (int x = 0; x < aerial.cols(); ++x) {
            const double intensity = aerial(y, x);
            if (background_window.at<unsigned char>(y, x) == 0 || intensity <= threshold || intensity < highest_intensity) continue;
            // 先用连续 AI 判断是否超阈值，再查该位置所属的 SRAF；未归属峰值不能压过其他可归属峰值。
            for (int region_index = 0; region_index < regions.size(); ++region_index) {
                if (finished[region_index] || regions[region_index].initial_mask(y, x) <= 0.0) continue;
                // 峰值相等时优先选择下标较小的块，使结果不受像素遍历顺序影响。
                if (intensity > highest_intensity || selected_region_index < 0 || region_index < selected_region_index) {
                    highest_intensity = intensity;
                    selected_region_index = region_index;
                }
                break;
            }
        }
    }
    return selected_region_index;
}

std::vector<SrafPointPair> find_sraf_pair_groups(const ControlPoints& controls, const ParametricDemo& renderer,
                                                const SrafRegion& region, const SrafShrinkConfig& geometry,
                                                const SrafPairGroupConfig& groups) {
    if (groups.maximum_pairs < 1 || !std::isfinite(groups.minimum_center_spacing_px) || groups.minimum_center_spacing_px < 0.0) {
        throw std::invalid_argument("Invalid pair-group parameters");
    }
    if (region.contour_indices.empty()) return {};
    // 本轮只取锁定块的外轮廓和孔洞控制点，800 是交给现有 CR 渲染器的采样点数请求。
    ControlPoints region_controls;
    for (int contour : region.contour_indices) {
        if (contour < 0 || contour >= controls.size() || controls[contour].rows() < 4 || !controls[contour].allFinite()) {
            throw std::invalid_argument("Invalid SRAF region control indices");
        }
        region_controls.push_back(controls[contour]);
    }
    const Polygons curves = renderer.get_curve_points(region_controls, 800);
    const auto widest = find_widest_sraf_section(curves);
    if (widest.diameter_px <= geometry.width_threshold_px) return {};

    // 先确定本轮最宽区域内哪些控制点具有有效材料内法线，再按法线穿过材料的厚度筛选。
    std::vector<SrafShrinkPoint> wide_controls;
    for (int local_contour = 0; local_contour < region_controls.size(); ++local_contour) {
        const auto& contour_controls = region_controls[local_contour];
        for (int point = 0; point < contour_controls.rows(); ++point) {
            const Eigen::Vector2d position = contour_controls.row(point).transpose();
            // 作用范围围绕本块当前最宽截面，不会选取其他 SRAF 的控制点。
            if ((position - widest.center_yx).norm() > widest.diameter_px) continue;
            const Eigen::Vector2d inward_normal_yx = control_inward_normal(contour_controls, point, curves);
            if (inward_normal_yx.squaredNorm() == 0.0) continue;
            const double normal_width_px = normal_material_width(curves, position, inward_normal_yx);
            if (normal_width_px <= geometry.width_threshold_px) continue;
            SrafShrinkPoint wide_control;
            wide_control.contour = region.contour_indices[local_contour];
            wide_control.point = point;
            wide_control.inward_yx = inward_normal_yx;
            wide_control.width_px = normal_width_px;
            wide_controls.push_back(wide_control);
        }
    }

    // 点对长度需位于 (厚度阈值,1.5*内切圆直径]，并检验两端法线相对和整条弦位于材料内。
    std::vector<SrafPointPair> candidates;
    for (int first = 0; first < wide_controls.size(); ++first) {
        for (int second = first + 1; second < wide_controls.size(); ++second) {
            const auto& first_control = wide_controls[first];
            const auto& second_control = wide_controls[second];
            const Eigen::Vector2d start = controls[first_control.contour].row(first_control.point).transpose();
            const Eigen::Vector2d finish = controls[second_control.contour].row(second_control.point).transpose();
            const double pair_distance_px = (finish - start).norm();
            if (pair_distance_px <= geometry.width_threshold_px || pair_distance_px > 1.5 * widest.diameter_px) continue;
            const Eigen::Vector2d pair_direction_yx = (finish - start) / pair_distance_px;
            if (first_control.inward_yx.dot(pair_direction_yx) < 0.5 || second_control.inward_yx.dot(-pair_direction_yx) < 0.5) continue;
            if (first_control.inward_yx.dot(second_control.inward_yx) > -0.5) continue;
            bool chord_inside_material = true;
            // 连线内部取 19 个位置，比例为 1/20 至 19/20，包含这两个比例端点。
            for (int sample = 1; sample < 20; ++sample) {
                const double fraction = sample / 20.0;
                if (!inside_sraf_material(curves, (1.0 - fraction) * start + fraction * finish)) {
                    chord_inside_material = false;
                    break;
                }
            }
            if (!chord_inside_material) continue;
            SrafPointPair pair;
            pair.valid = true;
            pair.first = first_control;
            pair.second = second_control;
            pair.width_px = pair_distance_px;
            pair.section_center_yx = 0.5 * (start + finish);
            pair.section_diameter_px = widest.diameter_px;
            candidates.push_back(pair);
        }
    }
    // 排序分数取两端法线厚度和最大内切圆直径的最小值，分数相同时再按点对距离排序。
    // 使用稳定排序保留完全相同分数的候选顺序，使贪心选组可重复。
    std::stable_sort(candidates.begin(), candidates.end(), [](const auto& first, const auto& second) {
        const double first_width_score_px = std::min({first.first.width_px, first.second.width_px, first.section_diameter_px});
        const double second_width_score_px = std::min({second.first.width_px, second.second.width_px, second.section_diameter_px});
        if (first_width_score_px != second_width_score_px) return first_width_score_px > second_width_score_px;
        return first.width_px > second.width_px;
    });
    auto compatible = [&](const SrafPointPair& candidate, const std::vector<SrafPointPair>& selected_groups) {
        for (const auto& existing : selected_groups) {
            for (const auto& point : {candidate.first, candidate.second}) {
                const bool shared_first = point.contour == existing.first.contour && point.point == existing.first.point;
                const bool shared_second = point.contour == existing.second.contour && point.point == existing.second.point;
                if (shared_first || shared_second) return false;
            }
            if ((candidate.section_center_yx - existing.section_center_yx).norm() >= groups.minimum_center_spacing_px) continue;
            const Eigen::Vector2d candidate_axis = controls[candidate.second.contour].row(candidate.second.point).transpose() -
                controls[candidate.first.contour].row(candidate.first.point).transpose();
            const Eigen::Vector2d existing_axis = controls[existing.second.contour].row(existing.second.point).transpose() -
                controls[existing.first.contour].row(existing.first.point).transpose();
            // 中点接近且轴向夹角小于 30 度的两条截面不重复选择；同一圆心的不同方向仍可联合收窄。
            if (std::abs(candidate_axis.normalized().dot(existing_axis.normalized())) > 0.8660254037844386) return false;
        }
        return true;
    };
    std::vector<SrafPointPair> best_groups;
    double best_width_score_sum_px = 0.0;
    // 轮换首选候选再贪心补齐，优先覆盖更多不同控制点，避免一条对角截面占用后续所有有效点。
    for (int seed = 0; seed < candidates.size(); ++seed) {
        std::vector<SrafPointPair> selected_groups{candidates[seed]};
        for (const auto& candidate : candidates) {
            if (selected_groups.size() >= groups.maximum_pairs) break;
            if (compatible(candidate, selected_groups)) selected_groups.push_back(candidate);
        }
        double width_score_sum_px = 0.0;
        for (const auto& pair : selected_groups) {
            width_score_sum_px += std::min({pair.first.width_px, pair.second.width_px, pair.section_diameter_px});
        }
        if (selected_groups.size() < best_groups.size() ||
            (selected_groups.size() == best_groups.size() && width_score_sum_px <= best_width_score_sum_px)) continue;
        best_groups = std::move(selected_groups);
        best_width_score_sum_px = width_score_sum_px;
    }
    return best_groups;
}

SrafShrinkProposal propose_sraf_group_shrink(const ControlPoints& controls, const std::vector<SrafPointPair>& pairs,
                                            const std::vector<double>& steps_px) {
    if (pairs.size() != steps_px.size()) throw std::invalid_argument("Pair-group step count mismatch");
    // 先确定每个点唯一的正位移，再统一写回候选，保证组间采用同一份输入控制点。
    SrafShrinkProposal proposal;
    proposal.control_points = controls;
    for (int group = 0; group < pairs.size(); ++group) {
        const double step = steps_px[group];
        if (!std::isfinite(step) || step < 0.0) throw std::invalid_argument("Invalid pair-group step");
        if (step == 0.0) continue;
        const auto& pair = pairs[group];
        if (!pair.valid) continue;
        if (pair.first.contour == pair.second.contour && pair.first.point == pair.second.point) {
            throw std::invalid_argument("Pair group must contain two distinct controls");
        }
        for (auto point : {pair.first, pair.second}) {
            if (point.contour < 0 || point.contour >= controls.size() || point.point < 0 ||
                point.point >= controls[point.contour].rows()) throw std::invalid_argument("Invalid pair-group control index");
            if (!point.inward_yx.allFinite() || std::abs(point.inward_yx.norm() - 1.0) > 1e-8) {
                throw std::invalid_argument("Pair-group normals must be finite unit vectors");
            }
            point.movement_px = step;
            auto duplicate = std::find_if(proposal.points.begin(), proposal.points.end(), [&](const auto& existing) {
                return point.contour == existing.contour && point.point == existing.point;
            });
            if (duplicate == proposal.points.end()) {
                proposal.points.push_back(point);
            } else {
                // 外部调用若传入共享点，只保留较小的正位移，防止同一点被多组重复累加。
                duplicate->movement_px = std::min(duplicate->movement_px, point.movement_px);
            }
        }
    }
    for (const auto& point : proposal.points) {
        proposal.control_points[point.contour].row(point.point) += point.movement_px * point.inward_yx.transpose();
    }
    proposal.moved_count = static_cast<int>(proposal.points.size());
    return proposal;
}

} // namespace litho
