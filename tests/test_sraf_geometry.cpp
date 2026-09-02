#include "sraf_geometry.h"

#include <opencv2/core.hpp>

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

void test_branch_detection() {
    cv::Mat branch = cv::Mat::zeros(11, 11, CV_8UC1);
    for (int y = 2; y <= 8; ++y) branch.at<uchar>(y, 5) = 255;
    for (int x = 3; x <= 7; ++x) branch.at<uchar>(5, x) = 255;

    const auto path = litho::SrafGeometry::order_skeleton_path(branch);
    require(!path.valid, "branched skeleton must be rejected");
    require(path.branchpoint_count > 0,
            "branched skeleton must report branchpoints");
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

}  // namespace

int main() {
    try {
        test_split_and_extract_open_sraf();
        test_closed_path();
        test_branch_detection();
        test_point_sraf();
        test_direct_interval_sampling();
        std::cout << "sraf_geometry tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "sraf_geometry test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
