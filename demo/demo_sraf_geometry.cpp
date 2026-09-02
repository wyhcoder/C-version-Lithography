#include "mask.h"
#include "save_txt.h"
#include "sraf_curve.h"
#include "sraf_geometry.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

cv::Mat eigen_mask_to_u8(const Eigen::MatrixXd& mask) {
    cv::Mat image(
        static_cast<int>(mask.rows()),
        static_cast<int>(mask.cols()),
        CV_8UC1,
        cv::Scalar(0));
    for (int y = 0; y < mask.rows(); ++y) {
        for (int x = 0; x < mask.cols(); ++x) {
            const double value = std::clamp(mask(y, x), 0.0, 1.0);
            image.at<uchar>(y, x) =
                static_cast<uchar>(std::lround(value * 255.0));
        }
    }
    return image;
}

void save_geometry_images(
    const Eigen::MatrixXd& lsm_mask,
    const litho::SrafGeometryResult& result,
    const std::filesystem::path& output_dir) {
    std::filesystem::create_directories(output_dir);

    const auto skeleton_path = output_dir / "sraf_skeleton.png";
    if (!cv::imwrite(skeleton_path.string(), result.skeleton_mask)) {
        throw std::runtime_error(
            "failed to save skeleton image: " + skeleton_path.string());
    }

    cv::Mat overlay;
    cv::cvtColor(eigen_mask_to_u8(lsm_mask), overlay, cv::COLOR_GRAY2BGR);

    // 红色表示一像素宽骨架。
    for (int y = 0; y < result.skeleton_mask.rows; ++y) {
        for (int x = 0; x < result.skeleton_mask.cols; ++x) {
            if (result.skeleton_mask.at<uchar>(y, x) != 0) {
                overlay.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 255);
            }
        }
    }

    // 绿色圆点表示从骨架按固定下标间隔选出的控制点；单点型 SRAF 用黄色标记。
    for (const auto& centerline : result.centerlines) {
        const bool point_sraf = centerline.path.points.size() == 1;
        const cv::Scalar color = point_sraf
            ? cv::Scalar(0, 255, 255)
            : cv::Scalar(0, 255, 0);
        for (const auto& point : centerline.control_points) {
            cv::circle(
                overlay,
                cv::Point(
                    static_cast<int>(std::lround(point.x)),
                    static_cast<int>(std::lround(point.y))),
                point_sraf ? 3 : 2,
                color,
                cv::FILLED,
                cv::LINE_AA);
        }
    }

    const auto overlay_path = output_dir / "sraf_skeleton_overlay.png";
    if (!cv::imwrite(overlay_path.string(), overlay)) {
        throw std::runtime_error(
            "failed to save skeleton overlay: " + overlay_path.string());
    }

    std::cout << "\nSaved visualization:\n"
              << "  " << skeleton_path << '\n'
              << "  " << overlay_path << '\n';
}

void save_curve_images(
    const Eigen::MatrixXd& lsm_mask,
    const std::vector<litho::SrafParametricCurve>& curves,
    const Eigen::MatrixXd& reconstructed_sraf,
    const std::filesystem::path& output_dir) {
    const auto reconstructed_path = output_dir / "sraf_reconstructed.png";
    if (!cv::imwrite(reconstructed_path.string(), eigen_mask_to_u8(reconstructed_sraf))) {
        throw std::runtime_error("failed to save reconstructed SRAF: " + reconstructed_path.string());
    }

    cv::Mat overlay;
    cv::cvtColor(eigen_mask_to_u8(lsm_mask), overlay, cv::COLOR_GRAY2BGR);
    for (const auto& curve : curves) {
        if (curve.points.size() == 1) {
            cv::circle(overlay, curve.points.front(), 2, cv::Scalar(255, 0, 0), cv::FILLED, cv::LINE_AA);
            continue;
        }

        std::vector<cv::Point> integer_points;
        integer_points.reserve(curve.points.size());
        for (const auto& point : curve.points) {
            integer_points.emplace_back(
                static_cast<int>(std::lround(point.x)),
                static_cast<int>(std::lround(point.y)));
        }
        cv::polylines(overlay, integer_points, curve.closed, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
    }

    const auto curve_path = output_dir / "sraf_fitted_curve_overlay.png";
    if (!cv::imwrite(curve_path.string(), overlay)) {
        throw std::runtime_error("failed to save fitted curve overlay: " + curve_path.string());
    }

    std::cout << "  " << curve_path << '\n'
              << "  " << reconstructed_path << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3 || argc > 9) {
        std::cerr
            << "Usage: demo_sraf_geometry <lsm_mask.txt> <target.bmp> "
               "[control_point_interval=10] [minimum_area=3] [opening_radius=0] "
               "[output_dir=outputs/sraf_geometry] [half_width=2.5] "
               "[samples_per_axis=4]\n";
        return EXIT_FAILURE;
    }

    try {
        Eigen::MatrixXd lsm_mask;
        litho::SaveTxt::load_txt(argv[1], lsm_mask);
        if (lsm_mask.rows() != lsm_mask.cols()) {
            throw std::invalid_argument(
                "demo_sraf_geometry currently expects a square LSM mask");
        }

        litho::Mask target(argv[2], static_cast<int>(lsm_mask.rows()), true);

        litho::SrafGeometryConfig config;
        if (argc >= 4) config.control_point_interval = std::stoi(argv[3]);
        if (argc >= 5) config.minimum_component_area = std::stoi(argv[4]);
        if (argc >= 6) config.opening_radius = std::stoi(argv[5]);
        const std::filesystem::path output_dir = argc >= 7
            ? std::filesystem::path(argv[6])
            : std::filesystem::path("outputs/sraf_geometry");
        const double half_width = argc >= 8 ? std::stod(argv[7]) : 2.5;
        const int samples_per_axis = argc >= 9 ? std::stoi(argv[8]) : 4;

        const auto result = litho::SrafGeometry::extract(
            lsm_mask, target.data(), config);

        const int main_pixels = static_cast<int>(
            (result.split.main_mask.array() > config.foreground_threshold).count());
        const int sraf_pixels = static_cast<int>(
            (result.split.sraf_mask.array() > config.foreground_threshold).count());

        std::cout << "\nSRAF geometry summary\n"
                  << "  mask size        : " << lsm_mask.rows() << " x "
                  << lsm_mask.cols() << '\n'
                  << "  main pixels      : " << main_pixels << '\n'
                  << "  raw SRAF pixels  : " << sraf_pixels << '\n'
                  << "  cleaned pixels   : "
                  << cv::countNonZero(result.cleaned_sraf_mask) << '\n'
                  << "  skeleton pixels  : "
                  << cv::countNonZero(result.skeleton_mask) << '\n'
                  << "  SRAF components  : " << result.centerlines.size() << "\n\n";

        bool all_valid = true;
        for (const auto& centerline : result.centerlines) {
            all_valid = all_valid && centerline.path.valid;
            std::cout << "  component " << std::setw(3)
                      << centerline.component_id
                      << " | area=" << std::setw(5) << centerline.area
                      << " | box=" << centerline.bounding_box.width << 'x'
                      << centerline.bounding_box.height
                      << " | skeleton=" << std::setw(4)
                      << cv::countNonZero(centerline.skeleton_mask)
                      << " | endpoints=" << centerline.path.endpoint_count
                      << " | branches=" << centerline.path.branchpoint_count
                      << " | controls=" << centerline.control_points.size()
                      << " | " << centerline.path.diagnostic << '\n';
        }

        save_geometry_images(lsm_mask, result, output_dir);

        if (!all_valid) {
            std::cerr << "\nAt least one SRAF skeleton is not a single valid path.\n";
            return 2;
        }
        if (result.centerlines.empty()) {
            std::cout << "\nNo SRAF centerline needs curve rendering.\n";
            return EXIT_SUCCESS;
        }

        std::vector<litho::SrafParametricCurve> curves;
        std::vector<litho::SrafDistanceCache> caches;
        curves.reserve(result.centerlines.size());
        caches.reserve(result.centerlines.size());
        const cv::Size mask_size(static_cast<int>(lsm_mask.cols()), static_cast<int>(lsm_mask.rows()));
        for (const auto& centerline : result.centerlines) {
            curves.push_back(litho::SrafCurve::fit(centerline));
            caches.push_back(litho::SrafCurve::build_distance_cache(
                curves.back(), mask_size, half_width, samples_per_axis));
        }

        const Eigen::MatrixXd reconstructed_sraf = litho::SrafCurve::render_all(caches, {half_width});
        save_curve_images(lsm_mask, curves, reconstructed_sraf, output_dir);
        std::cout << "\nSRAF curve rendering\n"
                  << "  half width       : " << half_width << " pixels\n"
                  << "  full width       : " << 2.0 * half_width << " pixels\n"
                  << "  MSAA samples     : " << samples_per_axis << " x " << samples_per_axis << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "demo_sraf_geometry failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
