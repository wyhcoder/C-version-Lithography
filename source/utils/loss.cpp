#include "loss.h"
#include "level_set_utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace litho {

    double Loss::get_pe_loss(){
        return pe_loss(_wafer_image, _target_image);
    }

    double Loss::pe_loss(const Eigen::MatrixXd& w,
                         const Eigen::MatrixXd& t){
        return (w.array() - t.array()).square().sum();
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

        // 以原 Loss::epe_loss 为准：EPE = |I - threshold| / |gradient(I)| * dx。
        Eigen::MatrixXd aerial_delta_x =
            LevelSetUtils::_gradient_1d(aerial_image, 1.0, 1);
        Eigen::MatrixXd aerial_delta_y =
            LevelSetUtils::_gradient_1d(aerial_image, 1.0, 0);
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
                G_magnitude(y, x) * dx;
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


}