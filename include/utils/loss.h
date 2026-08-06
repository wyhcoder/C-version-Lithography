#pragma once
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "ep_select.h"

namespace litho {

    // 单次 EPE 评估结果：逐点值不加权，便于用不同权重计算 WEPE。
    struct EpeEvaluation {
        double total_epe = 0.0;
        Eigen::RowVectorXd epe_vector;
    };

    class Loss {
        public:
            Loss(const Eigen::MatrixXd& wafer_image,
                 const Eigen::MatrixXd& target_image,
                 double threshold,
                 double alpha)
                : _wafer_image(wafer_image), _target_image(target_image),
                  _threshold(threshold), _alpha(alpha) {}

            double get_pe_loss();
            Eigen::VectorXd get_epe_loss();

            // 无状态版本：不构造对象、不持引用，热点循环里直接用
            static double pe_loss(const Eigen::MatrixXd& wafer_image,
                                  const Eigen::MatrixXd& target_image);

            // 使用 loss.cpp 的 EPE 公式，同时返回总 EPE 和逐点 EPE。
            // EPE = |I-threshold| / |gradient(I)| * dx；epe_vector 不加权。
            static EpeEvaluation evaluate_epe(
                const Eigen::MatrixXd& aerial_image,
                const Eigen::MatrixXd& eps,
                double threshold,
                double dx = 1.0);

            // WEPE = Σ weights[i] * epe_vector[i]。
            static double weighted_epe(
                const Eigen::RowVectorXd& weights,
                const Eigen::RowVectorXd& epe_vector);

            // 保留原 Loss API：从 eps_result 读取 EP 坐标，返回未加权总 EPE。
            // WEPE 请显式调用 weighted_epe。
            static double epe_loss(
                const Eigen::MatrixXd& aerial_image,
                const EpsResult& eps_result,
                double threshold,
                double dx);

            // 带非主图形区域显影惩罚的 PE loss
            //   wafer_image   : N×N，光刻胶显影结果
            //   target_image  : N×N，目标图（1=主图形应显影，0=非主图形不应显影）
            //   penalty       : 惩罚权重 β（非主图形区域 wafer 超过阈值就惩罚）
            //   penalty_threshold: 非主图形区域显影阈值（wafer 超过此值才惩罚）
            static double pe_loss_with_penalty(
                const Eigen::MatrixXd& wafer_image,
                const Eigen::MatrixXd& target_image,
                double penalty,
                double penalty_threshold = 0.01);

        protected:
            const Eigen::MatrixXd& _wafer_image;
            const Eigen::MatrixXd& _target_image;
            double _threshold;
            double _alpha;
    };
}

