#include "msaa.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

litho::Polygon make_ellipse(int samples) {
    litho::Polygon polygon(samples, 2);
    constexpr double pi = 3.14159265358979323846;
    for (int i = 0; i < samples; ++i) {
        const double angle = 2.0 * pi * i / samples;
        const double x = 31.5 + 19.5 * std::cos(angle);
        const double y = 31.5 + 17.0 * std::sin(angle);
        polygon(i, 0) = y;
        polygon(i, 1) = x;
    }
    return polygon;
}

double polygon_area(const litho::Polygon& polygon) {
    double twice_area = 0.0;
    for (int i = 0; i < polygon.rows(); ++i) {
        const int next = (i + 1) % polygon.rows();
        twice_area += polygon(i, 1) * polygon(next, 0) -
                      polygon(next, 1) * polygon(i, 0);
    }
    return 0.5 * std::abs(twice_area);
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

}  // namespace

int main() {
    try {
        constexpr int size = 64;
        const Eigen::MatrixXd mask_template = Eigen::MatrixXd::Zero(size, size);
        const litho::Polygon ellipse = make_ellipse(256);
        const litho::Polygons contours{ellipse};
        const litho::AntiAliasRenderer renderer(16);

        const Eigen::MatrixXd dirac = renderer.rasterize_dirac_indicator(
            contours, mask_template, "gray", 1.0, 0.25);
        const Eigen::MatrixXd reference = renderer.MSAA(
            contours, mask_template, "gray");

        require(dirac.allFinite(), "Dirac indicator contains non-finite values");
        require(dirac.minCoeff() >= 0.0 && dirac.maxCoeff() <= 1.0,
                "Dirac indicator must stay in [0,1]");
        require(dirac(32, 32) > 0.95, "ellipse center must be inside");
        require(dirac(2, 2) < 1e-12, "far corner must be outside");
        require(((dirac.array() > 0.0) && (dirac.array() < 1.0)).any(),
                "curved boundary must contain gray pixels");

        const double analytic_area = polygon_area(ellipse);
        const double area_error =
            std::abs(dirac.sum() - analytic_area) / analytic_area;
        const double mae = (dirac - reference).cwiseAbs().mean();
        const double rmse = std::sqrt((dirac - reference).squaredNorm() /
                                      static_cast<double>(dirac.size()));
        const double iou = binary_iou(dirac, reference);
        require(area_error < 0.02, "Dirac indicator area error is too large");
        require(mae < 0.02, "Dirac indicator MAE against MSAA is too large");
        require(rmse < 0.08, "Dirac indicator RMSE against MSAA is too large");
        require(iou > 0.96, "Dirac indicator IoU against MSAA is too small");

        litho::Polygon reversed = ellipse.colwise().reverse().eval();
        const Eigen::MatrixXd reversed_dirac =
            renderer.rasterize_dirac_indicator(
                {reversed}, mask_template, "gray", 1.0, 0.25);
        require((dirac - reversed_dirac).cwiseAbs().maxCoeff() < 1e-10,
                "Dirac indicator must not depend on contour orientation");

        const Eigen::MatrixXd binary = renderer.rasterize_dirac_indicator(
            contours, mask_template, "binary", 1.0, 0.25);
        require(((binary.array() == 0.0) || (binary.array() == 1.0)).all(),
                "binary Dirac indicator must contain only 0 and 1");

        litho::Polygon inner = ellipse;
        inner.col(0) = (31.5 + 0.35 * (inner.col(0).array() - 31.5)).matrix();
        inner.col(1) = (31.5 + 0.35 * (inner.col(1).array() - 31.5)).matrix();
        const Eigen::MatrixXd ring = renderer.rasterize_dirac_indicator(
            {ellipse, inner}, mask_template, "binary", 1.0, 0.25);
        require(ring(32, 32) == 0.0,
                "binary odd-even fill must preserve an inner hole");
        require(ring(32, 45) == 1.0,
                "binary odd-even fill must preserve the outer ring");

        std::cout << "Dirac rasterization metrics\n"
                  << "  polygon area : " << analytic_area << '\n'
                  << "  grid area    : " << dirac.sum() << '\n'
                  << "  area error   : " << area_error << '\n'
                  << "  MAE vs MSAA  : " << mae << '\n'
                  << "  RMSE vs MSAA : " << rmse << '\n'
                  << "  IoU @ 0.5    : " << iou << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "test_dirac_rasterization failed: " << error.what() << '\n';
        return 1;
    }
}
