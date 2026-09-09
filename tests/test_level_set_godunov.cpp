#include "level_set_utils.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void require_near(double actual, double expected, double tolerance,
                  const std::string& message) {
    if (std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            message + ": actual=" + std::to_string(actual) +
            ", expected=" + std::to_string(expected));
    }
}

Eigen::MatrixXd make_x_plane(int rows, int cols, double slope) {
    Eigen::MatrixXd phi(rows, cols);
    const double center = 0.5 * static_cast<double>(cols - 1);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            phi(i, j) = slope * (static_cast<double>(j) - center);
        }
    }
    return phi;
}

void test_planar_signed_distance() {
    constexpr int rows = 25;
    constexpr int cols = 29;
    const Eigen::MatrixXd speed_positive =
        Eigen::MatrixXd::Constant(rows, cols, 2.0);
    const Eigen::MatrixXd speed_negative =
        Eigen::MatrixXd::Constant(rows, cols, -2.0);

    const auto increasing = litho::LevelSetUtils::evolve_normal_WENO_godunov(
        make_x_plane(rows, cols, 1.0), speed_positive, 1.0, 1.0);
    const auto decreasing = litho::LevelSetUtils::evolve_normal_WENO_godunov(
        make_x_plane(rows, cols, -1.0), speed_positive, 1.0, 1.0);
    const auto reverse_speed = litho::LevelSetUtils::evolve_normal_WENO_godunov(
        make_x_plane(rows, cols, 1.0), speed_negative, 1.0, 1.0);

    const int row = rows / 2;
    const int col = cols / 2;
    require_near(increasing.delta(row, col), 2.0, 1e-12,
                 "positive x slope must have unit Godunov gradient");
    require_near(decreasing.delta(row, col), 2.0, 1e-12,
                 "negative x slope must have unit Godunov gradient");
    require_near(reverse_speed.delta(row, col), -2.0, 1e-12,
                 "negative normal speed must reverse the Hamiltonian sign");
    require_near(increasing.H1_abs(row, col), 2.0, 1e-12,
                 "x CFL bound must use absolute normal speed");
    require_near(increasing.H2_abs(row, col), 2.0, 1e-12,
                 "y CFL bound must use absolute normal speed");
    require(increasing.delta.rows() == rows && increasing.delta.cols() == cols,
            "Godunov result must preserve rectangular input dimensions");
}

void test_godunov_at_one_dimensional_extremum() {
    constexpr int rows = 17;
    constexpr int cols = 25;
    constexpr int center = cols / 2;
    Eigen::MatrixXd minimum(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            minimum(i, j) = std::abs(j - center);
        }
    }

    const Eigen::MatrixXd positive_speed = Eigen::MatrixXd::Ones(rows, cols);
    const Eigen::MatrixXd negative_speed = -positive_speed;
    const auto expanding = litho::LevelSetUtils::evolve_normal_WENO_godunov(
        minimum, positive_speed, 1.0, 1.0);
    const auto contracting = litho::LevelSetUtils::evolve_normal_WENO_godunov(
        minimum, negative_speed, 1.0, 1.0);

    // At a local minimum Dx-=-1 and Dx+=+1. Godunov selects neither for
    // positive speed, but both squared contributions for negative speed.
    require_near(expanding.delta(rows / 2, center), 0.0, 1e-12,
                 "positive speed at a local minimum must use the Godunov zero state");
    require_near(contracting.delta(rows / 2, center), -std::sqrt(2.0), 1e-6,
                 "negative speed at a local minimum must combine both one-sided slopes");
}

void test_invalid_inputs_are_rejected() {
    bool rejected = false;
    try {
        const Eigen::MatrixXd phi = Eigen::MatrixXd::Zero(5, 7);
        const Eigen::MatrixXd speed = Eigen::MatrixXd::Zero(5, 6);
        (void)litho::LevelSetUtils::evolve_normal_WENO_godunov(
            phi, speed, 1.0, 1.0);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "mismatched phi and speed dimensions must be rejected");
}

void test_circle_symmetry() {
    constexpr int size = 33;
    constexpr double radius = 10.0;
    const double center = 0.5 * static_cast<double>(size - 1);
    Eigen::MatrixXd phi(size, size);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            const double x = static_cast<double>(j) - center;
            const double y = static_cast<double>(i) - center;
            phi(i, j) = radius - std::hypot(x, y);
        }
    }

    const Eigen::MatrixXd speed = Eigen::MatrixXd::Ones(size, size);
    const auto result = litho::LevelSetUtils::evolve_normal_WENO_godunov(
        phi, speed, 1.0, 1.0);

    double maximum_mirror_error = 0.0;
    int error_i = 0;
    int error_j = 0;
    int mirror_i = 0;
    int mirror_j = 0;
    // edge-replication controls the outer boundary and a radial signed-distance
    // field is non-differentiable at its center. Level-set accuracy matters in
    // the narrow band around phi=0, so verify symmetry there.
    constexpr int margin = 4;
    for (int i = margin; i < size - margin; ++i) {
        for (int j = margin; j < size - margin; ++j) {
            if (std::abs(phi(i, j)) > 3.0) continue;
            const double horizontal_error = std::abs(
                result.delta(i, j) - result.delta(i, size - 1 - j));
            if (horizontal_error > maximum_mirror_error) {
                maximum_mirror_error = horizontal_error;
                error_i = i;
                error_j = j;
                mirror_i = i;
                mirror_j = size - 1 - j;
            }
            const double vertical_error = std::abs(
                result.delta(i, j) - result.delta(size - 1 - i, j));
            if (vertical_error > maximum_mirror_error) {
                maximum_mirror_error = vertical_error;
                error_i = i;
                error_j = j;
                mirror_i = size - 1 - i;
                mirror_j = j;
            }
        }
    }
    if (maximum_mirror_error >= 1e-12) {
        const Eigen::MatrixXd data_ext = litho::LevelSetUtils::pad_edge(
            litho::LevelSetUtils::pad_edge(phi, 3, 0), 3, 1);
        const Eigen::MatrixXd phi_y_minus =
            litho::LevelSetUtils::der_weno5(
                data_ext.transpose(), 1.0, "minus", 1).transpose();
        const Eigen::MatrixXd phi_y_plus =
            litho::LevelSetUtils::der_weno5(
                data_ext.transpose(), 1.0, "plus", 1).transpose();
        const int ext_i = error_i + 3;
        const int ext_j = error_j + 3;
        const int ext_mirror_i = mirror_i + 3;
        const int ext_mirror_j = mirror_j + 3;
        throw std::runtime_error(
            "constant-speed circular evolution must preserve mirror symmetry: error=" +
            std::to_string(maximum_mirror_error) + " at (" +
            std::to_string(error_i) + "," + std::to_string(error_j) +
            ") versus (" + std::to_string(mirror_i) + "," +
            std::to_string(mirror_j) + "), values=" +
            std::to_string(result.delta(error_i, error_j)) + "," +
            std::to_string(result.delta(mirror_i, mirror_j)) +
            ", y-minus/y-plus=" +
            std::to_string(phi_y_minus(ext_i, ext_j)) + "/" +
            std::to_string(phi_y_plus(ext_i, ext_j)) + " versus " +
            std::to_string(phi_y_minus(ext_mirror_i, ext_mirror_j)) + "/" +
            std::to_string(phi_y_plus(ext_mirror_i, ext_mirror_j)));
    }
}

}  // namespace

int main() {
    try {
        test_planar_signed_distance();
        test_godunov_at_one_dimensional_extremum();
        test_invalid_inputs_are_rejected();
        test_circle_symmetry();
        std::cout << "level-set Godunov tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "level-set Godunov test failed: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
