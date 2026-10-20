#pragma once
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "ep_select.h"
#include "epe_utils.h"

namespace litho {

    // 单次 EPE 评估结果：逐点值不加权，便于用不同权重计算 WEPE。
    struct EpeEvaluation {
        double total_epe = 0.0;
        Eigen::RowVectorXd epe_vector;
    };

    // Python 全芯片 EPE 的一次评价结果；优化目标使用像素平方，绝对 EPE 仅用于 nm 统计和逐点导出。
    struct SparseEpeEvaluation {
        double weighted_squared_loss = 0.0; // Σ weight_epe * signed_epe_pixel²，单位 pixel²。
        double absolute_sum_nm = 0.0; // 仅激活 EP 的 Σ|signed_epe_pixel| * pixel_size_nm。
        int active_count = 0; // weight_epe > 0 且参与损失、梯度和统计的 EP 数量。
        Eigen::RowVectorXd signed_epe_pixel; // 每个候选 EP 的有符号 EPE；未激活位置为 NaN，坐标单位 pixel。
        Eigen::RowVectorXd absolute_epe_nm; // 每个候选 EP 的绝对 EPE；未激活位置为 NaN，单位 nm。
        Eigen::RowVectorXd aerial_samples; // EP 处双线性插值得到的空中像；未激活位置为 NaN。
        Eigen::RowVectorXd gradient_samples; // EP 处双线性插值得到的 |∇AI|，单位光强/pixel；未激活位置为 NaN。
        Eigen::RowVectorXd point_losses; // 每个候选 EP 的 weight_epe * signed_epe_pixel²；未激活位置为 0。
    };

    // 对齐 epe.py 的单图 EPE 结果：所有传入点均有效，Sobel 分母加 epsilon，平方目标单位为 nm²。
    struct PytorchEpeEvaluation {
        double squared_loss_nm2 = 0.0; // Σ delta_nm²。
        Eigen::RowVectorXd signed_epe_nm; // delta_nm=(I-threshold)/(|Sobel(I)|+epsilon)*pixel_size_nm。
        Eigen::RowVectorXd absolute_epe_nm;
        Eigen::RowVectorXd aerial_samples;
        Eigen::RowVectorXd gradient_samples; // EP 处双线性插值的未归一化 Sobel 幅值。
        Eigen::RowVectorXd denominators; // gradient_samples + epsilon。
        Eigen::RowVectorXd point_losses_nm2;
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

            // 三个矩阵同尺寸；loss_window 为 0/1 输出像素窗口，只累计窗口内的 PE，不裁切光学输入。
            static double pe_loss(const Eigen::MatrixXd& wafer_image, const Eigen::MatrixXd& target_image,
                                  const Eigen::MatrixXd& loss_window);

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

            // 全芯片 Python 对齐接口：AI 和 |∇AI| 都在浮点 EP 坐标处双线性插值，梯度幅值下限为 1e-6。
            // 平方目标保持像素单位；pixel_size_nm 只把绝对 EPE 统计从 pixel 换算为 nm，不改变优化目标。
            static SparseEpeEvaluation evaluate_sparse_epe(const Eigen::MatrixXd& aerial_image, const EpsResult& eps_result,
                                                            double threshold, double pixel_size_nm);

            // 对齐 epe.py：使用未归一化 Sobel、零填充和双线性采样，所有 eps 行都参与计算。
            static PytorchEpeEvaluation evaluate_pytorch_epe(const Eigen::MatrixXd& aerial_image, const Eigen::MatrixXd& eps,
                                                              double threshold, double pixel_size_nm,
                                                              double epsilon = kPytorchEpeEpsilon);

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

            // PE 与背景惩罚均限制在 loss_window 内；背景沿用 target < 0.5，返回 PE + penalty*BG_raw。
            static double pe_loss_with_penalty(const Eigen::MatrixXd& wafer_image, const Eigen::MatrixXd& target_image,
                                              double penalty, double penalty_threshold, const Eigen::MatrixXd& loss_window);

            // 同尺寸 WI/target/窗口；返回窗口内原始背景 Σ max(WI-threshold,0)^2，不含 PE 或权重。
            static double background_penalty_loss(const Eigen::MatrixXd& wafer_image, const Eigen::MatrixXd& target_image,
                                                  double penalty_threshold, const Eigen::MatrixXd& loss_window);

        protected:
            const Eigen::MatrixXd& _wafer_image;
            const Eigen::MatrixXd& _target_image;
            double _threshold;
            double _alpha;
    };
}
