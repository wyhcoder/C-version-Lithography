#pragma once

#include "litho_prepare.h"
#include "ep_select.h"
#include <Eigen/Dense>
#include <vector>

namespace litho {

class Gradient {
public:
    Gradient(const ImagingCache& cache, double threshold, double alpha);
    ~Gradient();

    // 像素误差对掩模的梯度
    //   wafer_image    : N×N，光刻胶显影结果（已 sigmoid）
    //   target_image   : N×N，目标图
    //   electric_field : K 个 N×N 复电场（compute_aerial 内部缓存）
    Eigen::MatrixXd pe_gradient(const Eigen::MatrixXd& wafer_image,
                                const Eigen::MatrixXd& target_image,
                                const std::vector<Eigen::MatrixXcd>& electric_field);
    
    Eigen::MatrixXd epe_gradient(const Eigen::MatrixXd& aerial_image, EpsResult eps_result, double threshold, double dx, const std::vector<Eigen::MatrixXcd>& electric_field);
    // 带非主图形区域显影惩罚的梯度
    //   penalty           : 惩罚权重 β
    //   penalty_threshold : 非主图形区域显影阈值
    Eigen::MatrixXd pe_gradient_with_penalty(
        const Eigen::MatrixXd& wafer_image,
        const Eigen::MatrixXd& target_image,
        const std::vector<Eigen::MatrixXcd>& electric_field,
        double penalty,
        double penalty_threshold = 0.01);

    // EPE 损失对掩模的梯度
    //   aerial_image   : N×N 空间像（未经 sigmoid）
    //   eps_result     : EPE 测量点（含坐标和权重）
    //   threshold      : 显影阈值 τ
    //   dx             : 像素尺寸
    Eigen::MatrixXd epe_gradient(
        const Eigen::MatrixXd& aerial_image,
        const std::vector<Eigen::MatrixXcd>& electric_field,
        const EpsResult& eps_result,
        double threshold,
        double dx);

private:
    // 将损失对 aerial image 的导数通过所有光学核反传到 mask。
    // 各核并行计算，最后按 k 固定顺序归约，保证串并行数值一致。
    Eigen::MatrixXd _backpropagate(
        const Eigen::MatrixXd& dLdI,
        const std::vector<Eigen::MatrixXcd>& electric_field,
        double output_scale);

    const ImagingCache& _cache;
    double _threshold;
    double _alpha;

    
    

    // FFTW 资源
    fftw_complex* _fft_in  = nullptr;
    fftw_complex* _fft_out = nullptr;
    fftw_plan     _plan_fwd;
    fftw_plan     _plan_inv;
};

}  // namespace litho
