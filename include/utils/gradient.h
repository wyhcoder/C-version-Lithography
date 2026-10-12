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

    // 加权 PE：L = Σ loss_window * (wafer_image - target_image)²，窗口尺寸与图像相同。
    // 全芯片分块时 Core 为 1、Halo 为 0；窗口作用于反传的误差信号，返回的 mask 梯度仍覆盖完整局部区域。
    Eigen::MatrixXd pe_gradient(const Eigen::MatrixXd& wafer_image, const Eigen::MatrixXd& target_image,
                               const std::vector<Eigen::MatrixXcd>& electric_field, const Eigen::MatrixXd& loss_window);

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

    // 与带惩罚 PE 使用同一公式，但在反传误差信号处乘 loss_window，限定评价区域。
    Eigen::MatrixXd pe_gradient_with_penalty(const Eigen::MatrixXd& wafer_image, const Eigen::MatrixXd& target_image,
                                           const std::vector<Eigen::MatrixXcd>& electric_field, double penalty,
                                           double penalty_threshold, const Eigen::MatrixXd& loss_window);

    // 独立加权：L = Σ loss_window * [pe_weight*(WI-target)^2 + penalty*背景*max(WI-penalty_threshold,0)^2]。
    // 沿用原 PE/背景惩罚反传，合并后只反传一次；支持 pe_weight=0 时只优化背景惩罚。
    Eigen::MatrixXd pe_gradient_with_penalty(const Eigen::MatrixXd& wafer_image, const Eigen::MatrixXd& target_image,
                                           const std::vector<Eigen::MatrixXcd>& electric_field, double penalty,
                                           double penalty_threshold, const Eigen::MatrixXd& loss_window, double pe_weight);

    // PE/背景窗口均为完整局部尺寸：先分别限定输出误差并乘各自系数，再共用一次光学反传。
    // PEPenalty 传入未膨胀主图 PE 窗口和原始背景窗口；返回完整 Halo 梯度，不在 mask 域裁切。
    Eigen::MatrixXd pe_gradient_with_penalty(const Eigen::MatrixXd& wafer_image, const Eigen::MatrixXd& target_image,
                                           const std::vector<Eigen::MatrixXcd>& electric_field, double penalty,
                                           double penalty_threshold, const Eigen::MatrixXd& pe_window, double pe_weight,
                                           const Eigen::MatrixXd& penalty_window);

    // 平方 EPE 的近似 mask 驱动：Σ weight_epe*(I-threshold)²*dx²/G²，反传时冻结每像素斜率 G。
    // 输出为完整 N×N mask 梯度；Loss::epe_loss() 独立报告绝对 EPE，本接口不求该统计量的严格导数。
    //   aerial_image   : N×N 空间像（未经 sigmoid）
    //   electric_field : 当前 mask 成像得到的各核复电场，须与 aerial_image 同一次成像。
    //   eps_result     : EP 的局部像素索引 (y,x) 及 weight_epe；允许多个 EP 的贡献累加到同一像素。
    //   threshold      : 显影阈值 τ
    //   dx             : 各向同性物理长度/像素；1 表示以像素计量，传入 nm/像素时平方 EPE 使用 nm²。
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
