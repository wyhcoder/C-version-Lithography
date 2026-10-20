#include "MEEF_Optimizer.h"
#include "epe_utils.h"
#include "loss.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void require_close(const Eigen::MatrixXd& actual, const Eigen::MatrixXd& expected, double tolerance, const std::string& message) {
    require(actual.rows() == expected.rows() && actual.cols() == expected.cols(), message + ": shape mismatch");
    require((actual - expected).cwiseAbs().maxCoeff() <= tolerance, message);
}

void check_legacy_mode(const Eigen::MatrixXd& target, const Eigen::MatrixXd& aerial) {
    using namespace litho;
    MEEFPipelineConfig config;
    config.epe_mode = "legacy";
    config.interval_line = 10;
    config.interval_corner = 2;
    config.mid_weight = 4.0;
    config.other_weight = 1.0;

    const EpsResult selected = select_meef_evaluation_points(target, config);
    const EpsResult expected = EpSelect(target, config.mid_weight, config.other_weight)
                                   .select_eps_others(config.interval_line, config.interval_corner);
    require_close(selected.eps, expected.eps, 0.0, "legacy EP coordinates changed");
    require_close(selected.weight_epe, expected.weight_epe, 0.0, "legacy EPE weights changed");
    require_close(selected.weight_meef, expected.weight_meef, 0.0, "legacy MEEF weights changed");

    constexpr double threshold = 0.4, pixel_size_nm = 6.0;
    const Eigen::RowVectorXd actual = evaluate_meef_epe_vector(aerial, selected, threshold, pixel_size_nm, config.epe_mode);
    const Eigen::RowVectorXd legacy = Loss::evaluate_epe(aerial, selected.eps, threshold, pixel_size_nm).epe_vector;
    require_close(actual, legacy, 0.0, "legacy EPE calculation changed");
}

void check_subpixel_mode(const Eigen::MatrixXd& target, const Eigen::MatrixXd& aerial) {
    using namespace litho;
    MEEFPipelineConfig config;
    config.epe_mode = "subpixel";
    config.interval_line = 10;
    config.interval_corner = 2;
    config.mid_weight = 4.0;
    config.other_weight = 1.0;

    const EpsResult selected = select_meef_evaluation_points(target, config);
    const EpsResult expected = EpSelect(target, config.mid_weight, config.other_weight)
                                   .select_eps_subpixel(config.interval_line, config.interval_corner);
    require_close(selected.eps, expected.eps, 0.0, "subpixel EP selection was not used");
    require_close(selected.weight_epe, expected.weight_epe, 0.0, "subpixel EPE weights differ");
    require_close(selected.weight_meef, expected.weight_meef, 0.0, "subpixel MEEF weights differ");

    int fractional_index = -1, inactive_index = -1;
    for (int i = 0; i < selected.eps.rows(); ++i) {
        const bool fractional = std::abs(selected.eps(i, 0) - std::round(selected.eps(i, 0))) > 1e-9 ||
                                std::abs(selected.eps(i, 1) - std::round(selected.eps(i, 1))) > 1e-9;
        if (fractional && selected.weight_epe(i) > 0.0 && fractional_index < 0) fractional_index = i;
        if (selected.weight_epe(i) == 0.0 && inactive_index < 0) inactive_index = i;
    }
    require(fractional_index >= 0, "subpixel mode produced no fractional EP coordinate");
    require(inactive_index >= 0, "subpixel fixture produced no inactive candidate EP");

    constexpr double threshold = 0.4, pixel_size_nm = 6.0;
    const Eigen::RowVectorXd actual = evaluate_meef_epe_vector(aerial, selected, threshold, pixel_size_nm, config.epe_mode);
    const Eigen::RowVectorXd unit_pixel = evaluate_meef_epe_vector(aerial, selected, threshold, 1.0, config.epe_mode);
    require(actual.allFinite(), "inactive candidate produced NaN in the MEEF EPE vector");

    const SparseEpeEvaluation direct = Loss::evaluate_sparse_epe(aerial, selected, threshold, pixel_size_nm);
    Eigen::RowVectorXd expected_epe = direct.absolute_epe_nm;
    for (int i = 0; i < expected_epe.size(); ++i) {
        if (selected.weight_epe(i) <= 0.0) expected_epe(i) = 0.0;
    }
    require_close(actual, expected_epe, 1e-12, "subpixel MEEF did not restrict EPE to WEP positions");
    require(actual(inactive_index) == 0.0, "inactive candidate entered the MEEF EPE vector");

    // 线性光强场的梯度恒为 (0.01, 0.02)，用于同时验证空间差分、亚像素插值及 pixel→nm 只换算一次。
    for (int i = 0; i < selected.eps.rows(); ++i) {
        if (selected.weight_epe(i) <= 0.0) continue;
        require(std::abs(actual(i) - pixel_size_nm * unit_pixel(i)) <= 1e-10,
                "subpixel EPE was not converted from pixel to nm exactly once");
        const double y = selected.eps(i, 0), x = selected.eps(i, 1);
        const double expected_nm = std::abs(0.10 + 0.01 * x + 0.02 * y - threshold) / std::hypot(0.01, 0.02) * pixel_size_nm;
        require(std::abs(actual(i) - expected_nm) <= 1e-10, "subpixel EPE used the wrong spatial gradient or interpolation");
    }

    const SparseEpeEvaluation weighted = Loss::evaluate_sparse_epe(aerial, selected, threshold, pixel_size_nm);
    require(std::isnan(weighted.absolute_epe_nm(inactive_index)), "test fixture did not exercise inactive sparse EPE behavior");
}

void check_pytorch_mode() {
    using namespace litho;
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(8, 9);
    target.block(2, 3, 4, 4).setOnes();
    MEEFPipelineConfig config;
    config.epe_mode = "pytorch";
    config.interval_line = 2.0;
    config.interval_corner = 0.5;

    const EpsResult selected = select_meef_evaluation_points(target, config);
    Eigen::MatrixXd expected_points(8, 2);
    expected_points << 2.0, 2.5, 5.0, 2.5, 2.0, 6.5, 5.0, 6.5,
                       1.5, 3.0, 1.5, 6.0, 5.5, 3.0, 5.5, 6.0;
    require_close(selected.eps, expected_points, 1e-12, "edge_sampling.py point order or coordinates changed");
    require(selected.weight_epe.minCoeff() == 1.0 && selected.weight_meef.minCoeff() == 1.0,
            "pytorch mode did not mark every sampled point as an EP");

    Eigen::MatrixXd aerial(8, 9);
    for (int y = 0; y < aerial.rows(); ++y) {
        for (int x = 0; x < aerial.cols(); ++x) aerial(y, x) = 0.10 + 0.01 * x + 0.02 * y;
    }
    constexpr double threshold = 0.4, pixel_size_nm = 6.0, epsilon = 1e-7;
    const PytorchEpeEvaluation evaluation = Loss::evaluate_pytorch_epe(aerial, selected.eps, threshold, pixel_size_nm, epsilon);
    const Eigen::RowVectorXd solve_vector = evaluate_meef_epe_vector(aerial, selected, threshold, pixel_size_nm, config.epe_mode);
    require_close(solve_vector, evaluation.signed_epe_nm, 1e-12, "pytorch MEEF vector is not signed delta_nm");

    // 未归一化 Sobel 对线性场给出 (8*0.01, 8*0.02)，对应 epe.py 的 conv2d 结果。
    const double gradient = std::hypot(8.0 * 0.01, 8.0 * 0.02);
    double expected_loss = 0.0;
    for (int i = 0; i < selected.eps.rows(); ++i) {
        const double y = selected.eps(i, 0), x = selected.eps(i, 1);
        const double expected_nm = (0.10 + 0.01 * x + 0.02 * y - threshold) / (gradient + epsilon) * pixel_size_nm;
        require(std::abs(evaluation.signed_epe_nm(i) - expected_nm) <= 1e-10, "pytorch EPE differs from Sobel formula");
        expected_loss += expected_nm * expected_nm;
    }
    require(std::abs(evaluation.squared_loss_nm2 - expected_loss) <= 1e-9, "pytorch EPE loss is not sum(delta_nm^2)");

    const SobelGradientMaps padded = sobel_gradient_maps(Eigen::MatrixXd::Ones(3, 3));
    require(std::abs(padded.x(0, 0) - 3.0) <= 1e-12 && std::abs(padded.y(0, 0) - 3.0) <= 1e-12,
            "Sobel boundary does not match conv2d padding=1");
}

} // namespace

int main() {
    try {
        Eigen::MatrixXd target = Eigen::MatrixXd::Zero(32, 32);
        target.block(8, 8, 16, 16).setOnes();
        Eigen::MatrixXd aerial(32, 32);
        for (int y = 0; y < aerial.rows(); ++y) {
            for (int x = 0; x < aerial.cols(); ++x) aerial(y, x) = 0.10 + 0.01 * x + 0.02 * y;
        }
        check_legacy_mode(target, aerial);
        check_subpixel_mode(target, aerial);
        check_pytorch_mode();
        std::cout << "MEEF EPE mode tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "MEEF EPE mode tests failed: " << error.what() << '\n';
        return 1;
    }
}
