#include "sraf_geometry.h"
#include "sraf_curve.h"
#include "save_txt.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void test_split_and_extract_open_sraf() {
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(32, 32);
    Eigen::MatrixXd full = Eigen::MatrixXd::Zero(32, 32);

    target.block(12, 13, 8, 6).setOnes();
    full.block(12, 13, 8, 6).setConstant(0.8);
    full.block(5, 5, 3, 22).setConstant(0.6);

    litho::SrafGeometryConfig config;
    config.opening_radius = 0;
    config.minimum_component_area = 3;
    config.control_point_interval = 4;

    const auto geometry = litho::SrafGeometry::extract(full, target, config);
    require(std::abs(geometry.split.main_mask.sum() - 48.0 * 0.8) < 1e-12,
            "main mask gray values were not preserved");
    require(std::abs(geometry.split.sraf_mask.sum() - 66.0 * 0.6) < 1e-12,
            "SRAF mask gray values were not preserved");
    require(geometry.centerlines.size() == 1,
            "expected exactly one SRAF centerline");

    const auto& centerline = geometry.centerlines.front();
    require(centerline.path.valid, centerline.path.diagnostic);
    require(!centerline.path.closed, "horizontal SRAF must be an open path");
    require(centerline.path.endpoint_count == 2,
            "open SRAF must have two endpoints");
    require(centerline.path.branchpoint_count == 0,
            "open SRAF must not have branchpoints");
    require(centerline.control_points.size() >= 2,
            "open SRAF must produce at least two control points");
}

void test_closed_path() {
    cv::Mat ring = cv::Mat::zeros(12, 12, CV_8UC1);
    for (int x = 3; x <= 8; ++x) {
        ring.at<uchar>(3, x) = 255;
        ring.at<uchar>(8, x) = 255;
    }
    for (int y = 4; y <= 7; ++y) {
        ring.at<uchar>(y, 3) = 255;
        ring.at<uchar>(y, 8) = 255;
    }

    const auto path = litho::SrafGeometry::order_skeleton_path(ring);
    require(path.valid, path.diagnostic);
    require(path.closed, "ring skeleton must be closed");
    require(path.points.size() == 20,
            "closed path must visit every ring pixel exactly once");

    const auto controls = litho::SrafGeometry::sample_skeleton_points(
        path.points, 5, true);
    require(controls.size() >= 4,
            "closed path must produce at least four control points");
    require(cv::norm(controls.front() - controls.back()) > 1e-12,
            "closed control points must not duplicate the first point at the end");
}

void test_branch_graph() {
    cv::Mat branch = cv::Mat::zeros(11, 11, CV_8UC1);
    for (int y = 2; y <= 8; ++y) branch.at<uchar>(y, 5) = 255;
    for (int x = 3; x <= 7; ++x) branch.at<uchar>(5, x) = 255;

    const auto path = litho::SrafGeometry::order_skeleton_path(branch);
    require(path.valid, path.diagnostic);
    require(path.branchpoint_count == 1 && path.endpoint_count == 4, "cross skeleton must have one junction and four endpoints");
    require(path.edges.size() == 4, "cross skeleton must split into four graph edges");
    std::vector<std::size_t> edge_sizes;
    for (const auto& edge : path.edges) {
        edge_sizes.push_back(edge.size());
        require(cv::norm(edge.front() - cv::Point2d(5.0, 5.0)) < 1e-12 ||
                cv::norm(edge.back() - cv::Point2d(5.0, 5.0)) < 1e-12, "every graph edge must meet the junction");
    }
    std::sort(edge_sizes.begin(), edge_sizes.end());
    require(edge_sizes == std::vector<std::size_t>({3, 3, 4, 4}), "each graph edge must retain all pixels of its arm");
}

void test_point_sraf() {
    cv::Mat point = cv::Mat::zeros(7, 7, CV_8UC1);
    point.at<uchar>(3, 3) = 255;

    const auto path = litho::SrafGeometry::order_skeleton_path(point);
    require(path.valid, "single-point SRAF must remain a valid geometry");
    require(!path.closed, "single-point SRAF must not be marked as closed");
    require(path.points.size() == 1,
            "single-point SRAF must preserve its center point");

    const auto controls = litho::SrafGeometry::sample_skeleton_points(
        path.points, 10, false);
    require(controls.size() == 1,
            "single-point SRAF must produce one control point");
}

void test_direct_interval_sampling() {
    const std::vector<cv::Point2d> diagonal = {
        {0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}};
    const auto controls = litho::SrafGeometry::sample_skeleton_points(
        diagonal, 2, false);

    require(controls.size() == 3,
            "interval=2 must select indices 0, 2 and 4");
    require(cv::norm(controls.front() - diagonal.front()) < 1e-12,
            "open sampling must preserve the first endpoint");
    require(cv::norm(controls[1] - diagonal[2]) < 1e-12,
            "control point must be selected directly from the skeleton");
    require(cv::norm(controls.back() - diagonal.back()) < 1e-12,
            "open sampling must preserve the last endpoint");
}

void test_real_branched_sraf(const std::string& lsm_path, const std::string& target_path) {
    Eigen::MatrixXd lsm;
    Eigen::MatrixXd target;
    litho::SaveTxt::load_txt(lsm_path, lsm);
    litho::SaveTxt::load_txt(target_path, target);
    const auto geometry = litho::SrafGeometry::extract(lsm, target);
    int branched_components = 0;
    int graph_edges = 0;
    for (const auto& centerline : geometry.centerlines) {
        require(centerline.path.valid, "real SRAF component was dropped: " + centerline.path.diagnostic);
        if (centerline.path.edges.empty()) continue;
        ++branched_components;
        graph_edges += static_cast<int>(centerline.path.edges.size());
        require(centerline.edge_control_points.size() == centerline.path.edges.size(), "every graph edge must have its own controls");
        const auto curve = litho::SrafCurve::fit(centerline);
        require(curve.edges.size() == centerline.path.edges.size(), "every real graph edge must be fitted separately");
        for (std::size_t i = 0; i < curve.edges.size(); ++i) {
            require(cv::norm(curve.edges[i].front() - centerline.path.edges[i].front()) < 1e-12 &&
                    cv::norm(curve.edges[i].back() - centerline.path.edges[i].back()) < 1e-12,
                    "real fitted graph edge moved a shared endpoint or junction");
        }
        const auto cache = litho::SrafCurve::build_distance_cache(curve, cv::Size(lsm.cols(), lsm.rows()), 5.0, 4);
        const double narrow_area = litho::SrafCurve::render(cache, 1.0).sum();
        const double wide_area = litho::SrafCurve::render(cache, 3.0).sum();
        require(narrow_area > 0.0 && wide_area > narrow_area, "real branched SRAF must render and respond to width");
    }
    require(branched_components > 0, "real input has no branched SRAF components");
    std::cout << "real SRAF: " << geometry.centerlines.size() << " components, " << branched_components
              << " branched components, " << graph_edges << " graph edges\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        test_split_and_extract_open_sraf();
        test_closed_path();
        test_branch_graph();
        test_point_sraf();
        test_direct_interval_sampling();
        if (argc == 3) test_real_branched_sraf(argv[1], argv[2]);
        else require(argc == 1, "usage: test_sraf_geometry [lsm_mask.txt target_mask.txt]");
        std::cout << "sraf_geometry tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "sraf_geometry test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
