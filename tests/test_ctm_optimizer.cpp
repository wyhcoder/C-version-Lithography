#include "CTM_Optimizer.h"
#include "gradient.h"
#include "imaging.h"
#include "litho_prepare.h"
#include "lithography_simulator.h"
#include "loss.h"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

struct TestDirectory {
    std::filesystem::path original = std::filesystem::current_path();
    std::filesystem::path temporary = std::filesystem::temp_directory_path() /
        ("litho_ctm_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));

    TestDirectory() {
        std::filesystem::create_directories(temporary / "target_pattern");
        std::filesystem::current_path(temporary);
    }

    ~TestDirectory() {
        std::filesystem::current_path(original);
        std::error_code error;
        std::filesystem::remove_all(temporary, error);
    }
};

void check_pe_gradient(litho::LithographySimulator& simulator, litho::ImagingCache& cache, bool with_penalty) {
    const auto& target = simulator._mask.data();
    const double threshold = simulator._params.resist.threshold;
    const double alpha = with_penalty ? 10.0 : simulator._params.resist.alpha;
    constexpr double pi = 3.14159265358979323846;
    constexpr double penalty = 2.0;
    constexpr double penalty_threshold = 0.01;
    const Eigen::MatrixXd theta = (target.array() > 0.0).select(
        Eigen::MatrixXd::Constant(target.rows(), target.cols(), 0.3 * pi),
        Eigen::MatrixXd::Constant(target.rows(), target.cols(), 0.7 * pi));
    litho::Imaging imaging(cache);
    litho::Gradient gradient(cache, threshold, alpha);
    const Eigen::MatrixXd mask = 0.5 * (1.0 + theta.array().cos()).matrix();
    const auto baseline = imaging.compute(mask, threshold, alpha);
    if (with_penalty) {
        require(litho::Loss::pe_loss_with_penalty(baseline.wafer_image, target, penalty, penalty_threshold) >
                litho::Loss::pe_loss(baseline.wafer_image, target), "penalty test has no active penalty pixels");
    }
    const Eigen::MatrixXd mask_gradient = with_penalty
        ? gradient.pe_gradient_with_penalty(baseline.wafer_image, target, imaging.get_electric_field(), penalty, penalty_threshold)
        : gradient.pe_gradient(baseline.wafer_image, target, imaging.get_electric_field());
    const Eigen::MatrixXd theta_gradient = (-0.5 * mask_gradient.array() * theta.array().sin()).matrix();
    require(theta_gradient.norm() > 1e-8, "PE gradient is unexpectedly zero");
    const Eigen::MatrixXd direction = -theta_gradient / theta_gradient.norm();
    auto evaluate = [&](const Eigen::MatrixXd& trial_theta) {
        const Eigen::MatrixXd trial_mask = 0.5 * (1.0 + trial_theta.array().cos()).matrix();
        const auto result = imaging.compute(trial_mask, threshold, alpha);
        return with_penalty ? litho::Loss::pe_loss_with_penalty(result.wafer_image, target, penalty, penalty_threshold)
                            : litho::Loss::pe_loss(result.wafer_image, target);
    };
    constexpr double step = 1e-3;
    const double numeric = (evaluate(theta + step * direction) - evaluate(theta - step * direction)) / (2.0 * step);
    const double analytic = theta_gradient.cwiseProduct(direction).sum();
    require(std::abs(numeric - analytic) <= 1e-3 * std::max(1.0, std::abs(numeric)),
            with_penalty ? "penalty PE gradient disagrees with finite differences" : "PE gradient disagrees with finite differences");
}

void diagnose_real_input() {
    const litho::SimulationParameters params = litho::SimulationParameters::from_yaml("config.yaml");
    litho::LithographySimulator simulator(params);
    litho::LithoPrepare prepare(simulator._grid, simulator._pupil, simulator._source, true);
    litho::ImagingCache cache = prepare.cache();
    litho::Imaging imaging(cache);
    litho::Gradient gradient(cache, params.resist.threshold, params.resist.alpha);
    const Eigen::MatrixXd& target = simulator._mask.data();
    constexpr double pi = 3.14159265358979323846;
    const Eigen::MatrixXd theta = (target.array() > 0.0).select(
        Eigen::MatrixXd::Constant(target.rows(), target.cols(), 0.3 * pi),
        Eigen::MatrixXd::Constant(target.rows(), target.cols(), 0.7 * pi));
    const Eigen::MatrixXd mask = 0.5 * (1.0 + theta.array().cos()).matrix();
    const auto baseline = imaging.compute(mask, params.resist.threshold, params.resist.alpha);
    const Eigen::MatrixXd gm = gradient.pe_gradient(baseline.wafer_image, target, imaging.get_electric_field());
    const Eigen::MatrixXd gt = (-0.5 * gm.array() * theta.array().sin()).matrix();
    const Eigen::MatrixXd direction = -gt / gt.norm();
    auto evaluate = [&](const Eigen::MatrixXd& theta_value) {
        const Eigen::MatrixXd trial_mask = 0.5 * (1.0 + theta_value.array().cos()).matrix();
        const auto result = imaging.compute(trial_mask, params.resist.threshold, params.resist.alpha);
        return litho::Loss::pe_loss(result.wafer_image, target);
    };
    std::cout << "Real CTM PE=" << litho::Loss::pe_loss(baseline.wafer_image, target)
              << " grad_norm=" << gt.norm() << " analytic_directional=" << gt.cwiseProduct(direction).sum() << std::endl;
    for (double h : {1e-1, 1e-2, 1e-3}) {
        const double fplus = evaluate(theta + h * direction);
        const double fminus = evaluate(theta - h * direction);
        const double numeric = (fplus - fminus) / (2.0 * h);
        std::cout << "h=" << h << " numeric_directional=" << numeric << std::endl;
        if (h == 1e-3) require(std::abs(numeric + gt.norm()) <= 1e-3 * gt.norm(), "real PE gradient disagrees with finite differences");
    }
    Eigen::MatrixXd orthogonal = Eigen::MatrixXd::Random(target.rows(), target.cols());
    orthogonal -= gt * (orthogonal.cwiseProduct(gt).sum() / gt.squaredNorm());
    orthogonal /= orthogonal.norm();
    const double orthogonal_h = 0.1;
    const double orthogonal_numeric = (evaluate(theta + orthogonal_h * orthogonal) -
                                       evaluate(theta - orthogonal_h * orthogonal)) / (2.0 * orthogonal_h);
    std::cout << "orthogonal analytic=" << gt.cwiseProduct(orthogonal).sum() << " numeric=" << orthogonal_numeric << std::endl;
    Eigen::Index row = 0, col = 0;
    gt.cwiseAbs().maxCoeff(&row, &col);
    Eigen::MatrixXd pixel = Eigen::MatrixXd::Zero(target.rows(), target.cols());
    pixel(row, col) = 1.0;
    const double pixel_h = 1e-3;
    std::cout << "pixel (" << row << ',' << col << ") analytic=" << gt(row, col)
              << " numeric=" << (evaluate(theta + pixel_h * pixel) - evaluate(theta - pixel_h * pixel)) / (2.0 * pixel_h)
              << std::endl;

    if (std::getenv("CTM_DIAG_NO_OPT") != nullptr) return;

    litho::CTM_Optimizer optimizer(simulator, cache, 50, 0.9, "lbfgs", 10, 1e-2);
    const Eigen::MatrixXd optimized_mask = optimizer.optimize(true, false);
    const auto optimized = imaging.compute(optimized_mask, params.resist.threshold, params.resist.alpha);
    require(litho::Loss::pe_loss(optimized.wafer_image, target) < litho::Loss::pe_loss(baseline.wafer_image, target),
            "real L-BFGS optimization did not reduce PE");
}

void test_synthetic() {
    TestDirectory directory;
    cv::Mat pattern(65, 65, CV_8UC1, cv::Scalar(0));
    pattern(cv::Rect(23, 23, 19, 19)).setTo(255);
    require(cv::imwrite("target_pattern/ctm_smoke.bmp", pattern), "failed to write synthetic mask");

    litho::SimulationParameters params;
    params.system = {193.0, 4.0, 65};
    params.mask = {"ctm_smoke", 64};
    params.optics.na = 1.35;
    params.optics.refractive_index = 1.44;
    params.source = {"annular", 0.6, 0.9};
    params.resist = {"threshold", 0.25, 85};

    litho::LithographySimulator simulator(params);
    litho::LithoPrepare prepare(simulator._grid, simulator._pupil, simulator._source, true);
    litho::ImagingCache cache = prepare.cache();
    check_pe_gradient(simulator, cache, false);
    check_pe_gradient(simulator, cache, true);

    litho::CTM_Optimizer gradient_descent(simulator, cache, 3, 0.9, "gradient_descent", 10, 1e100);
    const Eigen::MatrixXd gd_mask = gradient_descent.optimize(false, false);
    require(gradient_descent.error_history().size() == 1, "gradient descent did not stop on gradient norm");
    require(gd_mask.allFinite(), "gradient descent returned a non-finite mask");

    litho::CTM_Optimizer lbfgs_early(simulator, cache, 3, 0.9, "lbfgs", 10, 1e100);
    const Eigen::MatrixXd lbfgs_early_mask = lbfgs_early.optimize(false, false);
    require(lbfgs_early.error_history().size() == 1, "L-BFGS did not stop on gradient norm");
    require(lbfgs_early_mask.allFinite(), "L-BFGS returned a non-finite mask");
    require(gd_mask.isApprox(lbfgs_early_mask, 1e-12), "both methods should return the same initial mask when stopped immediately");

    litho::CTM_Optimizer lbfgs_step(simulator, cache, 1, 0.9, "lbfgs", 10, 0.0);
    const Eigen::MatrixXd lbfgs_step_mask = lbfgs_step.optimize(false, false);
    require(lbfgs_step.error_history().size() >= 2, "L-BFGS did not evaluate a line-search step");
    require(lbfgs_step.error_history().back() <= lbfgs_step.error_history().front(), "L-BFGS line search did not reduce PE");
    require(lbfgs_step_mask.allFinite(), "L-BFGS step returned a non-finite mask");
}

}  // namespace

int main() {
    try {
        test_synthetic();
        if (std::getenv("CTM_DIAG_REAL") != nullptr) diagnose_real_input();

        std::cout << "CTM gradient stopping and both optimizer modes passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "CTM optimizer test failed: " << error.what() << '\n';
        return 1;
    }
}
