#include "sraf_curve.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

litho::SrafCenterlineGeometry make_open_centerline() {
    litho::SrafCenterlineGeometry centerline;
    centerline.component_id = 1;
    centerline.path.valid = true;
    centerline.path.closed = false;
    centerline.control_points = {{4.0, 8.0}, {9.0, 8.0}, {15.0, 8.0}, {20.0, 8.0}};
    return centerline;
}

double nearest_curve_point(
    const std::vector<cv::Point2d>& curve,
    const cv::Point2d& point) {
    double minimum = std::numeric_limits<double>::infinity();
    for (const auto& sample : curve) minimum = std::min(minimum, cv::norm(sample - point));
    return minimum;
}

void test_open_curve_and_gray_render() {
    const auto centerline = make_open_centerline();
    const auto curve = litho::SrafCurve::fit(centerline, 0.25);
    require(!curve.closed, "open centerline must produce an open curve");
    require(curve.degree == 3, "four fitting points must produce a cubic B-spline");
    require(curve.spline_control_points.size() == centerline.control_points.size(), "B-spline control points are missing");
    require(curve.points.size() > centerline.control_points.size(), "curve must contain dense samples");
    require(cv::norm(curve.points.front() - centerline.control_points.front()) < 1e-12, "curve must preserve first endpoint");
    require(cv::norm(curve.points.back() - centerline.control_points.back()) < 1e-12, "curve must preserve last endpoint");
    for (std::size_t i = 0; i < centerline.control_points.size(); ++i) {
        require(cv::norm(curve.spline_control_points[i] - centerline.control_points[i]) < 1e-12, "skeleton samples must be used directly as B-spline control points");
    }

    const auto cache = litho::SrafCurve::build_distance_cache(curve, cv::Size(32, 24), 3.0, 4);
    const Eigen::MatrixXd mask = litho::SrafCurve::render(cache, 2.0);
    require(std::abs(mask(8, 10) - 1.0) < 1e-12, "center pixel must be fully covered");
    require(mask(6, 10) > 0.0 && mask(6, 10) < 1.0, "boundary pixel must have gray MSAA coverage");
    require(mask(1, 10) == 0.0, "far pixel must remain background");
}

void test_middle_controls_do_not_require_interpolation() {
    litho::SrafCenterlineGeometry centerline;
    centerline.component_id = 4;
    centerline.path.valid = true;
    centerline.control_points = {{4.0, 12.0}, {8.0, 4.0}, {16.0, 4.0}, {20.0, 12.0}};

    const auto curve = litho::SrafCurve::fit(centerline, 0.1);
    require(nearest_curve_point(curve.points, centerline.control_points[1]) > 0.5, "curve must not be forced through the first middle control point");
    require(nearest_curve_point(curve.points, centerline.control_points[2]) > 0.5, "curve must not be forced through the second middle control point");
}

void test_width_changes_only_threshold() {
    const auto curve = litho::SrafCurve::fit(make_open_centerline(), 0.25);
    const auto cache = litho::SrafCurve::build_distance_cache(curve, cv::Size(32, 24), 3.0, 4);
    const Eigen::MatrixXd narrow = litho::SrafCurve::render(cache, 1.0);
    const Eigen::MatrixXd wide = litho::SrafCurve::render(cache, 3.0);
    require(wide.sum() > narrow.sum(), "larger half width must produce a larger SRAF");
}

void test_point_sraf() {
    litho::SrafCenterlineGeometry centerline;
    centerline.component_id = 2;
    centerline.path.valid = true;
    centerline.control_points = {{10.0, 10.0}};

    const auto curve = litho::SrafCurve::fit(centerline);
    const auto cache = litho::SrafCurve::build_distance_cache(curve, cv::Size(24, 24), 2.0, 4);
    const Eigen::MatrixXd mask = litho::SrafCurve::render(cache, 1.0);
    require(mask(10, 10) == 1.0, "point SRAF center must be fully covered");
    require(mask(10, 13) == 0.0, "point SRAF must remain local");
}

void test_closed_curve() {
    litho::SrafCenterlineGeometry centerline;
    centerline.component_id = 3;
    centerline.path.valid = true;
    centerline.path.closed = true;
    centerline.control_points = {{6.0, 6.0}, {14.0, 6.0}, {14.0, 14.0}, {6.0, 14.0}};

    const auto curve = litho::SrafCurve::fit(centerline, 0.25);
    require(curve.closed, "closed centerline must produce a closed curve");
    require(curve.degree == 3, "closed curve must use a periodic cubic B-spline");
    require(curve.spline_control_points.size() == centerline.control_points.size(), "periodic B-spline control points are missing");
    require(cv::norm(curve.points.front() - curve.points.back()) > 1e-12, "closed curve must not duplicate its first point");
    for (std::size_t i = 0; i < centerline.control_points.size(); ++i) {
        require(cv::norm(curve.spline_control_points[i] - centerline.control_points[i]) < 1e-12, "closed skeleton samples must remain the B-spline control points");
    }

    const auto cache = litho::SrafCurve::build_distance_cache(curve, cv::Size(24, 24), 2.0, 4);
    const Eigen::MatrixXd mask = litho::SrafCurve::render(cache, 1.0);
    require(mask(7, 10) > 0.0, "closed periodic curve must be rendered");
    require(mask(10, 10) == 0.0, "closed curve center must remain empty");
}

void test_render_all() {
    auto first = make_open_centerline();
    auto second = make_open_centerline();
    second.component_id = 2;
    for (auto& point : second.control_points) point.y += 8.0;

    const auto first_curve = litho::SrafCurve::fit(first);
    const auto second_curve = litho::SrafCurve::fit(second);
    const std::vector<litho::SrafDistanceCache> caches = {
        litho::SrafCurve::build_distance_cache(first_curve, cv::Size(32, 24), 2.0, 4),
        litho::SrafCurve::build_distance_cache(second_curve, cv::Size(32, 24), 2.0, 4)};
    const Eigen::MatrixXd mask = litho::SrafCurve::render_all(caches, {1.5});
    require(mask(8, 10) == 1.0 && mask(16, 10) == 1.0, "render_all must combine all SRAFs");
}

void test_branch_graph_width() {
    cv::Mat skeleton = cv::Mat::zeros(15, 15, CV_8UC1);
    for (int y = 3; y <= 11; ++y) skeleton.at<uchar>(y, 7) = 255;
    for (int x = 4; x <= 10; ++x) skeleton.at<uchar>(7, x) = 255;
    litho::SrafCenterlineGeometry centerline;
    centerline.component_id = 7;
    centerline.path = litho::SrafGeometry::order_skeleton_path(skeleton);
    require(centerline.path.valid && centerline.path.edges.size() == 4, "branched test skeleton must have four edges");
    for (const auto& edge : centerline.path.edges) {
        const auto controls = litho::SrafGeometry::sample_skeleton_points(edge, 2, false);
        centerline.control_points.insert(centerline.control_points.end(), controls.begin(), controls.end());
        centerline.edge_control_points.push_back(controls);
    }
    const auto curve = litho::SrafCurve::fit(centerline);
    require(curve.component_id == 7 && curve.edges.size() == 4, "four graph edges must remain one width component");
    for (std::size_t i = 0; i < curve.edges.size(); ++i) {
        const auto& controls = centerline.edge_control_points[i];
        const auto& fitted = curve.edges[i];
        require(fitted.size() > controls.size(), "graph edge must be densely fitted from its controls");
        require(cv::norm(fitted.front() - controls.front()) < 1e-12 && cv::norm(fitted.back() - controls.back()) < 1e-12,
                "fitted graph edge must preserve its endpoint and junction");
        for (const auto& control : controls) require(nearest_curve_point(fitted, control) < 1e-10, "Catmull-Rom edge must pass every sampled control");
    }
    const auto cache = litho::SrafCurve::build_distance_cache(curve, cv::Size(15, 15), 2.0, 4);
    const Eigen::MatrixXd mask = litho::SrafCurve::render(cache, 1.0);
    require(mask(7, 7) == 1.0 && mask(3, 7) > 0.0 && mask(7, 4) > 0.0, "junction and all graph arms must be rendered");
    require(mask(3, 3) == 0.0, "separate arms must not be connected by an artificial diagonal");
    require(litho::SrafCurve::render(cache, 2.0).sum() > mask.sum(), "branched SRAF must respond to its shared width");
}

void test_curved_graph_edge() {
    litho::SrafCenterlineGeometry centerline;
    centerline.component_id = 8;
    centerline.path.valid = true;
    centerline.edge_control_points = {
        {{2.0, 10.0}, {4.0, 4.0}, {8.0, 4.0}, {10.0, 10.0}},
        {{10.0, 10.0}, {14.0, 10.0}},
        {{10.0, 10.0}, {10.0, 14.0}}};
    centerline.path.edges = centerline.edge_control_points;
    for (const auto& controls : centerline.edge_control_points) {
        centerline.control_points.insert(centerline.control_points.end(), controls.begin(), controls.end());
    }
    const auto curve = litho::SrafCurve::fit(centerline, 0.1);
    require(curve.edges.size() == 3 && curve.degree == 3, "curved graph must keep its three separate fitted edges");
    bool curved_between_controls = false;
    for (const auto& point : curve.edges[0]) {
        if (point.x > 5.0 && point.x < 7.0 && point.y < 3.9) curved_between_controls = true;
    }
    require(curved_between_controls, "graph edge must follow a cubic interpolating curve, not just its control polygon");
    for (const auto& edge : curve.edges) require(cv::norm(edge.back() - cv::Point2d(10.0, 10.0)) < 1e-12 ||
                                              cv::norm(edge.front() - cv::Point2d(10.0, 10.0)) < 1e-12,
                                              "all fitted edges must keep the shared junction");
}

}  // namespace

int main() {
    try {
        test_open_curve_and_gray_render();
        test_middle_controls_do_not_require_interpolation();
        test_width_changes_only_threshold();
        test_point_sraf();
        test_closed_curve();
        test_render_all();
        test_branch_graph_width();
        test_curved_graph_edge();
        std::cout << "sraf_curve tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "sraf_curve test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
