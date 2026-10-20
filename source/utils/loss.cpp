#include "loss.h"
#include "epe_utils.h"
#include "level_set_utils.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace litho {

    double Loss::get_pe_loss(){
        return pe_loss(_wafer_image, _target_image);
    }

    double Loss::pe_loss(const Eigen::MatrixXd& w,
                         const Eigen::MatrixXd& t){
        return (w.array() - t.array()).square().sum();
    }

    double Loss::pe_loss(const Eigen::MatrixXd& w, const Eigen::MatrixXd& t, const Eigen::MatrixXd& loss_window) {
        return ((w.array() - t.array()).square() * loss_window.array()).sum();
    }

    double Loss::pe_loss_with_penalty(
        const Eigen::MatrixXd& wafer_image,
        const Eigen::MatrixXd& target_image,
        double penalty,
        double penalty_threshold)
    {
        // 主项: 全图像素误差
        double pe = (wafer_image - target_image).array().square().sum();

        // 非主图形区域（target==0）的显影惩罚:
        //   β · Σ_{target==0} max(0, wafer - τ)²
        double penalty_loss = 0.0;
        for (int i = 0; i < wafer_image.size(); ++i) {
            if (target_image(i) < 0.5) {  // 非主图形区域
                double excess = std::max(0.0, wafer_image(i) - penalty_threshold);
                penalty_loss += excess * excess;
            }
        }

        return pe + penalty * penalty_loss;
    }

    double Loss::pe_loss_with_penalty(const Eigen::MatrixXd& wafer_image, const Eigen::MatrixXd& target_image,
                                     double penalty, double penalty_threshold, const Eigen::MatrixXd& loss_window) {
        return pe_loss(wafer_image, target_image, loss_window) +
               penalty * background_penalty_loss(wafer_image, target_image, penalty_threshold, loss_window);
    }

    double Loss::background_penalty_loss(const Eigen::MatrixXd& wafer_image, const Eigen::MatrixXd& target_image,
                                        double penalty_threshold, const Eigen::MatrixXd& loss_window) {
        double background = 0.0;
        // 独立累计背景项；全芯片 PEPenalty 可用主图窗口评价 PE、背景窗口评价本项。
        for (int i = 0; i < wafer_image.size(); ++i) {
            if (target_image(i) < 0.5) {
                const double excess = std::max(0.0, wafer_image(i) - penalty_threshold);
                background += loss_window(i) * excess * excess;
            }
        }
        return background;
    }

    EpeEvaluation Loss::evaluate_epe(const Eigen::MatrixXd& aerial_image,
                                     const Eigen::MatrixXd& eps,
                                     double threshold,
                                     double dx)
    {
        EpeEvaluation result;
        result.epe_vector = Eigen::RowVectorXd::Zero(eps.rows());
        if (eps.rows() == 0) return result;
        if (eps.cols() < 2) {
            throw std::invalid_argument(
                "Loss::evaluate_epe: eps must have at least 2 columns (y, x)");
        }
        if (aerial_image.size() == 0) {
            throw std::invalid_argument(
                "Loss::evaluate_epe: aerial_image must not be empty");
        }

        // dx 是 nm/px；斜率已按物理间距计算，因此 EPE = |I-threshold|/|∇I| 的单位直接是 nm。
        Eigen::MatrixXd aerial_delta_x =
            LevelSetUtils::_gradient_1d(aerial_image, dx, 1);
        Eigen::MatrixXd aerial_delta_y =
            LevelSetUtils::_gradient_1d(aerial_image, dx, 0);
        Eigen::MatrixXd G_magnitude =
            (aerial_delta_x.array().square() +
             aerial_delta_y.array().square()).sqrt();
        G_magnitude.array() += 1e-12;

        for (int i = 0; i < eps.rows(); ++i) {
            int y = std::clamp(static_cast<int>(eps(i, 0)),
                               0, static_cast<int>(aerial_image.rows()) - 1);
            int x = std::clamp(static_cast<int>(eps(i, 1)),
                               0, static_cast<int>(aerial_image.cols()) - 1);
            result.epe_vector(i) =
                std::abs(aerial_image(y, x) - threshold) /
                G_magnitude(y, x) ;
        }
        result.total_epe = result.epe_vector.sum();
        return result;
    }

    double Loss::weighted_epe(const Eigen::RowVectorXd& weights,
                              const Eigen::RowVectorXd& epe_vector)
    {
        if (weights.size() != epe_vector.size()) {
            throw std::invalid_argument(
                "Loss::weighted_epe: weights and epe_vector sizes must match");
        }
        return (weights.array() * epe_vector.array()).sum();
    }

    double Loss::epe_loss(const Eigen::MatrixXd& aerial_image,
                          const EpsResult& eps_result,
                          double threshold,
                          double dx)
    {
        return evaluate_epe(
            aerial_image, eps_result.eps, threshold, dx).total_epe;
    }

    SparseEpeEvaluation Loss::evaluate_sparse_epe(const Eigen::MatrixXd& aerial_image, const EpsResult& eps_result,
                                                   double threshold, double pixel_size_nm) {
        const int count = static_cast<int>(eps_result.eps.rows());
        const double nan = std::numeric_limits<double>::quiet_NaN();
        SparseEpeEvaluation result;
        result.signed_epe_pixel = Eigen::RowVectorXd::Constant(count, nan);
        result.absolute_epe_nm = Eigen::RowVectorXd::Constant(count, nan);
        result.aerial_samples = Eigen::RowVectorXd::Constant(count, nan);
        result.gradient_samples = Eigen::RowVectorXd::Constant(count, nan);
        result.point_losses = Eigen::RowVectorXd::Zero(count);

        // 复用项目已有的一维空间差分，先逐像素求 |∇AI|，再分别插值 AI 和梯度幅值。
        const Eigen::MatrixXd gradient_x = LevelSetUtils::_gradient_1d(aerial_image, 1.0, 1);
        const Eigen::MatrixXd gradient_y = LevelSetUtils::_gradient_1d(aerial_image, 1.0, 0);
        Eigen::MatrixXd gradient_magnitude = (gradient_x.array().square() + gradient_y.array().square()).sqrt().matrix();
        gradient_magnitude = gradient_magnitude.array().max(1e-6).matrix();

        for (int i = 0; i < count; ++i) {
            const double weight = eps_result.weight_epe(i);
            if (weight <= 0.0) continue;
            const auto stencil = make_bilinear_stencil(
                aerial_image.rows(), aerial_image.cols(), eps_result.eps(i, 0), eps_result.eps(i, 1));
            const double aerial = bilinear_sample(aerial_image, stencil);
            const double magnitude = std::max(bilinear_sample(gradient_magnitude, stencil), 1e-6);
            const double signed_epe = (aerial - threshold) / magnitude;
            // magnitude 按 d=1 pixel 求得，signed_epe 的单位是 pixel；这里只在绝对 EPE 统计中换算一次 nm。
            const double absolute_nm = std::abs(signed_epe) * pixel_size_nm;
            const double point_loss = weight * signed_epe * signed_epe;
            result.signed_epe_pixel(i) = signed_epe;
            result.absolute_epe_nm(i) = absolute_nm;
            result.aerial_samples(i) = aerial;
            result.gradient_samples(i) = magnitude;
            result.point_losses(i) = point_loss;
            result.weighted_squared_loss += point_loss;
            result.absolute_sum_nm += absolute_nm;
            ++result.active_count;
        }
        return result;
    }

    PytorchEpeEvaluation Loss::evaluate_pytorch_epe(const Eigen::MatrixXd& aerial_image, const Eigen::MatrixXd& eps,
                                                     double threshold, double pixel_size_nm, double epsilon) {
        const int count = static_cast<int>(eps.rows());
        PytorchEpeEvaluation result;
        result.signed_epe_nm.resize(count);
        result.absolute_epe_nm.resize(count);
        result.aerial_samples.resize(count);
        result.gradient_samples.resize(count);
        result.denominators.resize(count);
        result.point_losses_nm2.resize(count);

        const SobelGradientMaps gradient = sobel_gradient_maps(aerial_image);
        for (int i = 0; i < count; ++i) {
            // edge_sampling.py 输出 (x,y)，EpsResult 统一保存为 (y,x)，这里直接按矩阵行列构造双线性模板。
            const BilinearStencil stencil = make_bilinear_stencil(aerial_image.rows(), aerial_image.cols(), eps(i, 0), eps(i, 1));
            const double aerial = bilinear_sample(aerial_image, stencil);
            const double magnitude = bilinear_sample(gradient.magnitude, stencil);
            const double denominator = magnitude + epsilon;
            const double signed_nm = (aerial - threshold) / denominator * pixel_size_nm;
            const double point_loss_nm2 = signed_nm * signed_nm;
            result.signed_epe_nm(i) = signed_nm;
            result.absolute_epe_nm(i) = std::abs(signed_nm);
            result.aerial_samples(i) = aerial;
            result.gradient_samples(i) = magnitude;
            result.denominators(i) = denominator;
            result.point_losses_nm2(i) = point_loss_nm2;
            result.squared_loss_nm2 += point_loss_nm2;
        }
        return result;
    }


}
