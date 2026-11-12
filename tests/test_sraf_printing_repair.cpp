#include "sraf_printing_repair.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

// 八点矩形含四个边中点；局部横向厚度可与矩形的已知宽度直接比较，坐标单位 pixel。
litho::Polygon rectangle(double y0, double x0, double y1, double x1) {
    litho::Polygon points(8, 2);
    const double ym = 0.5 * (y0 + y1), xm = 0.5 * (x0 + x1);
    points << y0, x0, y0, xm, y0, x1, ym, x1, y1, x1, y1, xm, y1, x0, ym, x0;
    return points;
}

// 连续曲线按奇偶填充识别材料和孔洞，自交检测保护重新拟合后的闭合曲线。
void test_material_holes_and_intersection() {
    const Eigen::MatrixXd mask = Eigen::MatrixXd::Zero(100, 100);
    litho::ParametricDemo renderer("CR", mask, 16, "msaa");
    const litho::ControlPoints controls{rectangle(10, 10, 90, 90), rectangle(35, 35, 65, 65)};
    const auto curves = renderer.get_curve_points(controls);
    require(litho::inside_sraf_material(curves, Eigen::Vector2d(25, 50)), "outer ring material must remain inside");
    require(!litho::inside_sraf_material(curves, Eigen::Vector2d(50, 50)), "hole must remain outside material");
    litho::Polygon bow(4, 2);
    bow << 0, 0, 10, 10, 0, 10, 10, 0;
    require(litho::sraf_contour_self_intersects(bow), "crossed contour must be rejected");
    require(!litho::sraf_contour_self_intersects(controls[0]), "simple contour must not be rejected");
}

// 八个圆周控制点位于真实 CR 曲线上，解析切线方向已知，可核查各组法线及两端等距内缩。
litho::Polygon circle_controls(double center_y, double center_x, double radius) {
    litho::Polygon controls(8, 2);
    const double pi = std::acos(-1.0);
    for (int point = 0; point < 8; ++point) {
        const double angle = 2.0 * pi * point / 8;
        controls.row(point) << center_y + radius * std::sin(angle), center_x + radius * std::cos(angle);
    }
    return controls;
}

// 直接给空间像和背景范围，不构造 WI；核查严格阈值、逐块锁定、排除带和未归属热点。
void test_direct_aerial_selection() {
    std::vector<litho::SrafRegion> regions(2);
    // 6×10 整图中两块材料分别覆盖 y=[1,2]、x=[1,2] 和 y=[1,2]、x=[6,7]，区间包含端点。
    for (auto& region : regions) region.initial_mask = Eigen::MatrixXd::Zero(6, 10);
    regions[0].initial_mask.block(1, 1, 2, 2).setOnes();
    regions[1].initial_mask.block(1, 6, 2, 2).setOnes();
    Eigen::MatrixXd aerial = Eigen::MatrixXd::Zero(6, 10);
    cv::Mat background_window(6, 10, CV_8UC1, cv::Scalar(255));
    const double threshold = 0.25;
    aerial(1, 1) = 0.34;
    aerial(1, 6) = 0.32;
    aerial(4, 9) = 0.9; // 未落在任何 SRAF 内，不能遮蔽可归属的较弱热点。
    require(litho::select_printing_sraf_from_aerial(regions, aerial, threshold, background_window, {false, false}) == 0,
            "direct AI selection must choose the strongest assignable SRAF");
    require(litho::select_printing_sraf_from_aerial(regions, aerial, threshold, background_window, {true, false}) == 1,
            "direct AI selection must skip completed SRAFs");
    background_window.at<unsigned char>(1, 1) = 0;
    require(litho::select_printing_sraf_from_aerial(regions, aerial, threshold, background_window, {false, false}) == 1,
            "main-pattern exclusion must apply before AI selection");
    aerial(1, 6) = threshold;
    require(litho::select_printing_sraf_from_aerial(regions, aerial, threshold, background_window, {false, false}) == -1,
            "AI equal to threshold must not trigger a strict greater-than repair");
    aerial(1, 6) = 0.249999;
    require(litho::select_printing_sraf_from_aerial(regions, aerial, threshold, background_window, {false, false}) == -1,
            "subthreshold AI must not trigger repair");
    aerial(1, 6) = std::nextafter(threshold, std::numeric_limits<double>::infinity());
    require(litho::select_printing_sraf_from_aerial(regions, aerial, threshold, background_window, {false, false}) == 1,
            "AI just above threshold must trigger without WI rounding or connected-component filtering");
    aerial(1, 6) = aerial(2, 1) = 0.32;
    require(litho::select_printing_sraf_from_aerial(regions, aerial, threshold, background_window, {false, false}) == 0,
            "equal peaks must keep region order even if the later region is encountered at an earlier pixel");
}

// 固定窗口内既统计超目标像素，也统计未超目标像素作为分母；窗口外热点不得影响损失。
void test_printing_loss_in_fixed_window() {
    Eigen::MatrixXd aerial(2, 3);
    aerial << 0.30, 0.20, 0.90, 0.25, 0.10, 0.80;
    cv::Mat window(2, 3, CV_8UC1, cv::Scalar(0));
    window.at<unsigned char>(0, 0) = window.at<unsigned char>(0, 1) = 255;
    require(std::abs(litho::sraf_printing_loss(aerial, window, 0.25) - 0.00125) < 1e-14,
            "printing loss must average the positive excess over the entire fixed window");
    aerial(0, 0) = 0.25;
    require(litho::sraf_printing_loss(aerial, window, 0.25) == 0.0, "at or below target must have zero penalty");
    window.setTo(0);
    bool rejected = false;
    try {
        litho::sraf_printing_loss(aerial, window, 0.25);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "empty printing-loss window must not be treated as successful repair");
}

// 验证梯度方法会取得本块全部控制点，孔洞法线进入环形材料，其他 SRAF 不在变量中。
void test_all_control_normals_for_gradient() {
    const Eigen::MatrixXd mask = Eigen::MatrixXd::Zero(100, 100);
    litho::ParametricDemo renderer("CR", mask, 16, "msaa");
    const litho::ControlPoints controls{circle_controls(30, 30, 15), circle_controls(30, 30, 8), circle_controls(70, 70, 7)};
    const auto regions = litho::group_sraf_regions(controls, renderer);
    const auto points = litho::sraf_control_normals(controls, renderer, regions[0]);
    require(points.size() == 16, "gradient variables must include every outer and hole control in the locked SRAF");
    for (const auto& point : points) {
        require(point.contour == 0 || point.contour == 1, "normal variables must stay in the locked SRAF");
        const Eigen::Vector2d position = controls[point.contour].row(point.point).transpose();
        Eigen::Vector2d expected = (Eigen::Vector2d(30, 30) - position).normalized();
        if (point.contour == 1) expected = -expected;
        require(point.inward_yx.dot(expected) > 1.0 - 1e-10, "outer and hole normals must both enter material");
        require(point.width_px > 0.0, "valid ring controls must have positive material thickness");
    }
    // 宽度小于几何筛选下限的圆仍提供全部八个法线，梯度选点不使用最宽截面过滤。
    const litho::ControlPoints small{circle_controls(30, 30, 2)};
    const auto small_regions = litho::group_sraf_regions(small, renderer);
    require(litho::sraf_control_normals(small, renderer, small_regions[0]).size() == 8,
            "normal-gradient variables must not be restricted to geometrically wide controls");
}

void test_group_holes_and_reject_long_axis_width() {
    const Eigen::MatrixXd mask = Eigen::MatrixXd::Zero(100, 100);
    litho::ParametricDemo renderer("CR", mask, 16, "msaa");
    // 第一块为一个外轮廓和一层孔洞，第二块为独立实心轮廓，坐标和半径单位均为 pixel。
    const litho::ControlPoints controls{circle_controls(30, 30, 15), circle_controls(30, 30, 8), circle_controls(70, 70, 7)};
    const auto regions = litho::group_sraf_regions(controls, renderer);
    require(regions.size() == 2, "one outer contour with a hole and one separate contour must form two SRAFs");
    require(regions[0].contour_indices == std::vector<int>({0, 1}), "hole must remain assigned to its outer SRAF");
    require(regions[1].contour_indices == std::vector<int>({2}), "separate SRAF must retain its own contour");
    // 多层包含属于不支持输入，必须明确拒绝，避免静默漏掉轮廓。
    auto nested_controls = controls;
    nested_controls.push_back(circle_controls(30, 30, 3));
    bool nested_rejected = false;
    try {
        litho::group_sraf_regions(nested_controls, renderer);
    } catch (const std::invalid_argument&) {
        nested_rejected = true;
    }
    require(nested_rejected, "multiple levels of enclosed contours must be rejected");
    const litho::ControlPoints narrow{rectangle(30, 10, 34, 90)};
    const auto narrow_regions = litho::group_sraf_regions(narrow, renderer);
    require(litho::find_sraf_pair_groups(narrow, renderer, narrow_regions[0], {}, {}).empty(), "thin long SRAF has no wide groups");
}

void test_joint_groups_and_shared_control_deduplication() {
    // 两块圆形 SRAF 的整图中心为 (20,20) 和 (20,55) pixel，只锁定前一块执行联合更新。
    const Eigen::MatrixXd mask = Eigen::MatrixXd::Zero(80, 80);
    litho::ParametricDemo renderer("CR", mask, 16, "msaa");
    const litho::ControlPoints controls{circle_controls(20, 20, 7), circle_controls(20, 55, 7)};
    const auto regions = litho::group_sraf_regions(controls, renderer);
    const auto pairs = litho::find_sraf_pair_groups(controls, renderer, regions[0], {}, {});
    require(pairs.size() == 4, "circle must support four distinct opposing directions without repeated controls");
    std::set<int> used_points;
    for (const auto& pair : pairs) {
        require(pair.first.contour == 0 && pair.second.contour == 0, "joint groups must stay in the locked SRAF");
        require(used_points.insert(pair.first.point).second && used_points.insert(pair.second.point).second, "groups share a control");
    }
    const auto proposal = litho::propose_sraf_group_shrink(controls, pairs, std::vector<double>(pairs.size(), 0.25));
    require(proposal.moved_count == 8, "joint circle update must move eight distinct controls once");
    require((proposal.control_points[1] - controls[1]).cwiseAbs().maxCoeff() == 0.0, "other SRAF must remain unchanged");
    for (const auto& point : proposal.points) {
        const Eigen::Vector2d position = controls[point.contour].row(point.point).transpose();
        const Eigen::Vector2d expected = (Eigen::Vector2d(20, 20) - position).normalized();
        require(point.inward_yx.dot(expected) > 1.0 - 1e-10, "joint groups must use inward analytic CR normals");
    }
    for (int point = 0; point < 8; ++point) {
        const Eigen::Vector2d moved = proposal.control_points[0].row(point).transpose();
        require(std::abs((moved - Eigen::Vector2d(20, 20)).norm() - 6.75) < 1e-10, "joint update must uniformly shrink circle radius");
    }
    const auto repeated = litho::propose_sraf_group_shrink(controls, {pairs[0], pairs[0]}, {0.25, 0.125});
    require(repeated.moved_count == 2, "repeated controls must be deduplicated");
    for (const auto& point : repeated.points) {
        const double distance = (repeated.control_points[point.contour].row(point.point) - controls[point.contour].row(point.point)).norm();
        require(std::abs(distance - 0.125) < 1e-10, "shared control steps must use minimum, not accumulate");
    }
    const auto paused = litho::propose_sraf_group_shrink(controls, pairs, std::vector<double>(pairs.size(), 0.0));
    require(paused.moved_count == 0, "zero-step groups must stay paused");
    require((paused.control_points[0] - controls[0]).cwiseAbs().maxCoeff() == 0.0, "paused proposal must preserve original coordinates");
}

} // namespace

int main() {
    try {
        test_material_holes_and_intersection();
        test_direct_aerial_selection();
        test_printing_loss_in_fixed_window();
        test_all_control_normals_for_gradient();
        test_group_holes_and_reject_long_axis_width();
        test_joint_groups_and_shared_control_deduplication();
        std::cout << "SRAF printing geometry tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
