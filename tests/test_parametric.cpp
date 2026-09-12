#include "parametric.h"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

double binary_iou(const Eigen::MatrixXd& lhs, const Eigen::MatrixXd& rhs) {
    int intersection = 0;
    int union_count = 0;
    for (int row = 0; row < lhs.rows(); ++row) {
        for (int col = 0; col < lhs.cols(); ++col) {
            const bool a = lhs(row, col) >= 0.5;
            const bool b = rhs(row, col) >= 0.5;
            intersection += a && b;
            union_count += a || b;
        }
    }
    return union_count == 0
        ? 1.0
        : static_cast<double>(intersection) / union_count;
}

double cosine_similarity(const Eigen::MatrixXd& lhs,
                         const Eigen::MatrixXd& rhs) {
    const double denominator = lhs.norm() * rhs.norm();
    return denominator > 1e-15
        ? (lhs.array() * rhs.array()).sum() / denominator
        : 0.0;
}

void save_matrix(const std::filesystem::path& path,
                 const Eigen::MatrixXd& matrix) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("cannot write comparison matrix: " + path.string());
    }
    output << std::setprecision(12);
    for (int row = 0; row < matrix.rows(); ++row) {
        for (int col = 0; col < matrix.cols(); ++col) {
            if (col != 0) output << ' ';
            output << matrix(row, col);
        }
        output << '\n';
    }
}

void test_periodic_cubic_b_spline() {
    litho::Polygon controls(4, 2);
    controls << 4.0, 4.0,
                4.0, 20.0,
                20.0, 20.0,
                20.0, 4.0;

    const Eigen::MatrixXd canvas = Eigen::MatrixXd::Zero(25, 25);
    const litho::ParametricDemo parametric("BS", canvas, 16);
    const litho::Polygons curves = parametric.get_curve_points({controls}, 80);

    require(curves.size() == 1, "one control contour must produce one curve");
    require(curves.front().rows() == 80, "B-spline sample count is incorrect");

    // 三次周期均匀 B 样条在第一段 t=0 时为 (P[-1]+4P[0]+P[1])/6。
    const Eigen::Vector2d expected_first(20.0 / 3.0, 20.0 / 3.0);
    require(
        (curves.front().row(0).transpose() - expected_first).norm() < 1e-12,
        "main contour is not using the cubic uniform B-spline basis");
    require(
        (curves.front().row(0) - controls.row(0)).norm() > 1.0,
        "B-spline must treat input points as controls rather than interpolation points");

    constexpr int curve_samples = 200;
    const Eigen::MatrixXd mask_msaa =
        parametric.render_curve({controls}, curve_samples);
    const Eigen::MatrixXd mask_dirac =
        parametric.render_curve_dirac({controls}, curve_samples);
    require(mask_msaa(12, 12) == 1.0,
            "closed B-spline must fill its interior");
    require(mask_msaa(1, 1) == 0.0,
            "pixels outside the B-spline must remain empty");
    require(mask_dirac(12, 12) > 0.99,
            "Dirac-rendered B-spline must fill its interior");
    require(mask_dirac(1, 1) < 1e-12,
            "pixels outside the Dirac-rendered B-spline must remain empty");

    const Eigen::MatrixXd difference = mask_dirac - mask_msaa;
    const double area_difference =
        std::abs(mask_dirac.sum() - mask_msaa.sum()) /
        std::max(mask_msaa.sum(), 1e-12);
    const double mae = difference.cwiseAbs().mean();
    const double iou = binary_iou(mask_dirac, mask_msaa);
    require(area_difference < 0.01,
            "small-grid Dirac/MSAA area difference is excessive");
    require(mae < 0.03,
            "small-grid Dirac/MSAA MAE is excessive");
    require(iou > 0.90,
            "small-grid Dirac/MSAA binary overlap is insufficient");

    std::cout << "small-grid same-control-point comparison\n"
              << "  canvas / samples    : 25x25 / " << curve_samples << '\n'
              << "  relative area diff  : " << area_difference << '\n'
              << "  MAE                 : " << mae << '\n'
              << "  IoU @ 0.5           : " << iou << '\n';

}

void test_same_control_point_rasterizers(
    const std::filesystem::path* output_directory) {
    litho::Polygon controls(8, 2);
    controls << 30.0, 20.0,
                20.0, 64.0,
                30.0, 108.0,
                64.0, 115.0,
                100.0, 105.0,
                112.0, 64.0,
                100.0, 22.0,
                64.0, 12.0;

    const Eigen::MatrixXd canvas = Eigen::MatrixXd::Zero(128, 128);
    const litho::ParametricDemo parametric("BS", canvas, 16);
    constexpr int curve_samples = 200;

    // 两个接口接收完全相同的控制点，并在内部生成完全相同的 200 个曲线点；
    // 唯一变量是最后一步使用 MSAA 还是论文 Dirac 指示函数光栅化。
    const Eigen::MatrixXd mask_msaa =
        parametric.render_curve({controls}, curve_samples);
    const Eigen::MatrixXd mask_dirac =
        parametric.render_curve_dirac({controls}, curve_samples);
    const Eigen::MatrixXd difference = mask_dirac - mask_msaa;
    const double mae = difference.cwiseAbs().mean();
    const double rmse = std::sqrt(
        difference.squaredNorm() / static_cast<double>(difference.size()));
    const double max_abs = difference.cwiseAbs().maxCoeff();
    const double iou = binary_iou(mask_dirac, mask_msaa);
    const double relative_area_difference =
        std::abs(mask_dirac.sum() - mask_msaa.sum()) /
        std::max(mask_msaa.sum(), 1e-12);

    std::cout << "same-control-point rasterizer comparison\n"
              << "  curve samples       : " << curve_samples << '\n'
              << "  MSAA area           : " << mask_msaa.sum() << '\n'
              << "  Dirac area          : " << mask_dirac.sum() << '\n'
              << "  relative area diff  : " << relative_area_difference << '\n'
              << "  MAE                 : " << mae << '\n'
              << "  RMSE                : " << rmse << '\n'
              << "  max abs difference  : " << max_abs << '\n'
              << "  IoU @ 0.5           : " << iou << '\n';

    require(relative_area_difference < 0.01,
            "Dirac and MSAA B-spline masks have excessive area difference");
    require(mae < 0.01,
            "Dirac and MSAA B-spline masks have excessive MAE");
    require(rmse < 0.06,
            "Dirac and MSAA B-spline masks have excessive RMSE");
    require(iou > 0.97,
            "Dirac and MSAA B-spline masks have insufficient binary IoU");

    if (output_directory != nullptr) {
        std::filesystem::create_directories(*output_directory);
        const litho::Polygons curve_points =
            parametric.get_curve_points({controls}, curve_samples);
        save_matrix(*output_directory / "control_points_yx.txt", controls);
        save_matrix(*output_directory / "curve_points_yx.txt", curve_points.front());
        save_matrix(*output_directory / "mask_msaa.txt", mask_msaa);
        save_matrix(*output_directory / "mask_dirac.txt", mask_dirac);
        save_matrix(*output_directory / "mask_difference.txt", difference);
        std::cout << "  comparison output   : "
                  << output_directory->string() << '\n';
    }
}

void test_dirac_shape_derivative() {
    litho::Polygon controls(8, 2);
    controls << 30.0, 20.0,
                20.0, 64.0,
                30.0, 108.0,
                64.0, 115.0,
                100.0, 105.0,
                112.0, 64.0,
                100.0, 22.0,
                64.0, 12.0;

    const Eigen::MatrixXd canvas = Eigen::MatrixXd::Zero(128, 128);
    const litho::ParametricDemo parametric("BS", canvas, 16, "dirac");
    constexpr int curve_samples = 200;
    constexpr int control_index = 2;
    constexpr double delta = 0.05;

    const litho::RasterDerivativeXY analytic =
        parametric.render_curve_dirac_derivative(
            {controls}, 0, control_index, curve_samples);
    auto central_difference = [&](int coordinate) {
        litho::Polygon plus = controls;
        litho::Polygon minus = controls;
        plus(control_index, coordinate) += delta;
        minus(control_index, coordinate) -= delta;
        return (parametric.render_curve_dirac({plus}, curve_samples) -
                parametric.render_curve_dirac({minus}, curve_samples)) /
               (2.0 * delta);
    };
    // 控制点存储顺序为 (y,x)，解析接口则按优化器的 (x,y) 命名返回。
    const Eigen::MatrixXd fd_x = central_difference(1);
    const Eigen::MatrixXd fd_y = central_difference(0);
    const double cosine_x = cosine_similarity(analytic.dx, fd_x);
    const double cosine_y = cosine_similarity(analytic.dy, fd_y);
    const double ratio_x = analytic.dx.norm() / std::max(fd_x.norm(), 1e-15);
    const double ratio_y = analytic.dy.norm() / std::max(fd_y.norm(), 1e-15);

    std::cout << "Dirac mask derivative comparison\n"
              << "  central delta       : " << delta << '\n'
              << "  cosine x / y        : " << cosine_x << " / " << cosine_y << '\n'
              << "  norm ratio x / y    : " << ratio_x << " / " << ratio_y << '\n';
    require(analytic.dx.allFinite() && analytic.dy.allFinite(),
            "Dirac analytic derivatives must be finite");
    require(cosine_x > 0.80 && cosine_y > 0.80,
            "Dirac analytic derivatives do not agree with central differences");
    require(ratio_x > 0.75 && ratio_x < 1.25 &&
            ratio_y > 0.75 && ratio_y < 1.25,
            "Dirac analytic derivative magnitude is inconsistent");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        test_periodic_cubic_b_spline();
        test_dirac_shape_derivative();
        const std::filesystem::path output_directory =
            argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::path{};
        test_same_control_point_rasterizers(
            argc > 1 ? &output_directory : nullptr);
        std::cout << "parametric B-spline tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "parametric B-spline test failed: " << error.what()
                  << '\n';
        return EXIT_FAILURE;
    }
}
