#include "parametric.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
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

    const Eigen::MatrixXd mask = parametric.render_curve({controls});
    require(mask(12, 12) == 1.0, "closed B-spline must fill its interior");
    require(mask(1, 1) == 0.0, "pixels outside the B-spline must remain empty");
}

}  // namespace

int main() {
    try {
        test_periodic_cubic_b_spline();
        std::cout << "parametric B-spline tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "parametric B-spline test failed: " << error.what()
                  << '\n';
        return EXIT_FAILURE;
    }
}
