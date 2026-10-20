#include "gradient.h"
#include "epe_utils.h"
#include "fft.h"
#include <fftw3.h>
#include "level_set_utils.h"
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace litho {

Gradient::Gradient(const ImagingCache& cache, double threshold, double alpha)
    : _cache(cache), _threshold(threshold), _alpha(alpha)
{
    const int N  = _cache.N;
    const int sz = N * N;
    _fft_in   = fftw_alloc_complex(sz);
    _fft_out  = fftw_alloc_complex(sz);
    _plan_fwd = fftw_plan_dft_2d(N, N, _fft_in, _fft_out,
                                 FFTW_FORWARD,  FFTW_MEASURE); //FFTW_PATIENT FFTW_MEASURE
    _plan_inv = fftw_plan_dft_2d(N, N, _fft_in, _fft_out,
                                 FFTW_BACKWARD, FFTW_MEASURE);
}

Gradient::~Gradient() {
    fftw_destroy_plan(_plan_fwd);
    fftw_destroy_plan(_plan_inv);
    fftw_free(_fft_in);
    fftw_free(_fft_out);
}

Eigen::MatrixXd Gradient::_backpropagate(
    const Eigen::MatrixXd& dLdI,
    const std::vector<Eigen::MatrixXcd>& electric_field,
    double output_scale)
{
    const int N = _cache.N;
    const int K = static_cast<int>(electric_field.size());
    const double invN = 1.0 / (static_cast<double>(N) * N);
    const auto& optical_kernels_frequency = _cache.optical_kernels_frequency;
    const auto& ws = _cache.source_ws;
    const auto& socs_eigenvalues = _cache.socs_eigenvalues;
    const bool use_socs = !socs_eigenvalues.empty();

    if (dLdI.rows() != N || dLdI.cols() != N) {
        throw std::invalid_argument("Gradient: dLdI dimensions do not match cache");
    }
    if (static_cast<int>(optical_kernels_frequency.size()) != K ||
        (use_socs && static_cast<int>(socs_eigenvalues.size()) != K) ||
        (!use_socs && ws.size() != K)) {
        throw std::invalid_argument("Gradient: optical kernel weights are incomplete");
    }

    // 每个 k 写独立矩阵，避免共享 gradient 的数据竞争。
    // 并行区结束后固定按 k=0..K-1 求和，使 1 线程与多线程结果一致。
    std::vector<Eigen::MatrixXd> kernel_gradients(K);

    // 外层已按 Micro/MEEF 任务并行时，单个任务内部顺序处理核，避免再创建一组工作线程。
#ifdef _OPENMP
    const bool enable_inner_parallel = (omp_in_parallel() == 0);
    #pragma omp parallel if(enable_inner_parallel)
#endif
    {
        // FFTW plan 只读共享；执行所用的输入、输出 buffer 必须每线程私有。
        fftw_complex* local_in  = fftw_alloc_complex(N * N);
        fftw_complex* local_out = fftw_alloc_complex(N * N);
        Eigen::MatrixXcd local_buf(N, N);

        #pragma omp for schedule(static)
        for (int k = 0; k < K; ++k) {
            const double weight = use_socs ? socs_eigenvalues[k] : ws(k);

            // 复数伴随的输入是 E_k · dL/dI；内积本身会对它取共轭。
            local_buf = electric_field[k].array() * dLdI.array();

            FFT::ifftshift_inplace(local_buf);
            FFT::to_fftw(local_buf, local_in);
            fftw_execute_dft(_plan_fwd, local_in, local_out);
            FFT::from_fftw(local_out, local_buf);

            local_buf.array() *= optical_kernels_frequency[k].conjugate().array();

            FFT::to_fftw(local_buf, local_in);
            fftw_execute_dft(_plan_inv, local_in, local_out);
            FFT::from_fftw(local_out, local_buf);
            FFT::fftshift_inplace(local_buf);

            kernel_gradients[k] =
                (output_scale * weight * invN * local_buf.real().array()).matrix();
        }

        fftw_free(local_in);
        fftw_free(local_out);
    }

    Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(N, N);
    for (int k = 0; k < K; ++k) {
        gradient += kernel_gradients[k];
    }

    const double source_weight_sum = ws.sum();
    if (source_weight_sum == 0.0) {
        throw std::runtime_error("Gradient: source weight sum is zero");
    }
    gradient /= source_weight_sum;
    return gradient;
}

// 梯度推导（链式法则）：
//   L     = Σ (W - T)²                   （像素误差能量）
//   W     = σ(α(I - threshold))          （光刻胶显影模型）
//   I(x)  = Σ_k weight_k |E_k(x)|² / Σ source_ws
//   E_k   = ifft( M_f · K_k )           （compute_aerial 里缓存的光学核）
//
//   dL/dM(x) = 4α / Σ source_ws · Σ_k weight_k Re{ ifft( fft(E_k·dL/dI) · conj(K_k) ) }
//   其中 dL/dI = (W - T) · W · (1 - W)
//
// optical_kernels_frequency 的 DC 在角点（与 FFT(M) 对齐）；伴随计算使用其复共轭。
Eigen::MatrixXd Gradient::pe_gradient(
    const Eigen::MatrixXd& wafer_image,
    const Eigen::MatrixXd& target_image,
    const std::vector<Eigen::MatrixXcd>& electric_field)
{
    // dL/dI 中暂不含 PE 平方项的 2、强度 |E|² 的 2 和 sigmoid 的 alpha；
    // 统一由 output_scale = 4*alpha 在反传末端乘入。
    Eigen::MatrixXd dLdI =
        (wafer_image - target_image).array() *
         wafer_image.array() * (1.0 - wafer_image.array());

    return _backpropagate(dLdI, electric_field, 4.0 * _alpha);
}

Eigen::MatrixXd Gradient::pe_gradient(const Eigen::MatrixXd& wafer_image, const Eigen::MatrixXd& target_image,
                                    const std::vector<Eigen::MatrixXcd>& electric_field, const Eigen::MatrixXd& loss_window) {
    // 在成像误差处限定统计区域，再经过光学伴随反传；不能先算全区域梯度再裁成 Core。
    const Eigen::MatrixXd dLdI = ((wafer_image - target_image).array() * wafer_image.array() *
                                  (1.0 - wafer_image.array()) * loss_window.array()).matrix();
    return _backpropagate(dLdI, electric_field, 4.0 * _alpha);
}

Eigen::MatrixXd Gradient::pe_gradient_with_penalty(
    const Eigen::MatrixXd& wafer_image,
    const Eigen::MatrixXd& target_image,
    const std::vector<Eigen::MatrixXcd>& electric_field,
    double penalty,
    double penalty_threshold)
{
    // ── 1. 主项 dL/dI = (W - T) · W · (1 - W) ──────────────────────────
    Eigen::MatrixXd dLdI_main =
        (wafer_image - target_image).array() *
         wafer_image.array() * (1.0 - wafer_image.array());

    // ── 2. 非主图形区域（target==0）显影惩罚 ──────────────────────────
    //    L_pen = β · Σ_{target==0} max(0, W - τ)²
    //    dL_pen/dW = 2β · max(0, W - τ)            (仅在 target==0 区域)
    //    dW/dI     = α · W · (1 - W)              (sigmoid 导数)
    // 公共的 4α 系数已含平方损失和 |E|² 导数的两个 2，因此这里只保留 β。
    Eigen::MatrixXd dLdI_pen =
        penalty *
        (wafer_image.array() - penalty_threshold).cwiseMax(0.0) *
        wafer_image.array() * (1.0 - wafer_image.array());
    // 只在非主图形区域（target < 0.5）生效
    dLdI_pen.array() *= (target_image.array() < 0.5).cast<double>();

    Eigen::MatrixXd dLdI = dLdI_main + dLdI_pen;
    return _backpropagate(dLdI, electric_field, 4.0 * _alpha);
}

Eigen::MatrixXd Gradient::pe_gradient_with_penalty(const Eigen::MatrixXd& wafer_image, const Eigen::MatrixXd& target_image,
                                                const std::vector<Eigen::MatrixXcd>& electric_field, double penalty,
                                                double penalty_threshold, const Eigen::MatrixXd& loss_window) {
    return pe_gradient_with_penalty(wafer_image, target_image, electric_field, penalty, penalty_threshold, loss_window, 1.0);
}

Eigen::MatrixXd Gradient::pe_gradient_with_penalty(const Eigen::MatrixXd& wafer_image, const Eigen::MatrixXd& target_image,
                                                const std::vector<Eigen::MatrixXcd>& electric_field, double penalty,
                                                double penalty_threshold, const Eigen::MatrixXd& loss_window, double pe_weight) {
    return pe_gradient_with_penalty(wafer_image, target_image, electric_field, penalty, penalty_threshold,
                                    loss_window, pe_weight, loss_window);
}

Eigen::MatrixXd Gradient::pe_gradient_with_penalty(const Eigen::MatrixXd& wafer_image, const Eigen::MatrixXd& target_image,
                                                const std::vector<Eigen::MatrixXcd>& electric_field, double penalty,
                                                double penalty_threshold, const Eigen::MatrixXd& pe_window, double pe_weight,
                                                const Eigen::MatrixXd& penalty_window) {
    const Eigen::ArrayXXd residual = wafer_image.array() - target_image.array();
    const Eigen::ArrayXXd background = (target_image.array() < 0.5).cast<double>();
    const Eigen::ArrayXXd excess = (wafer_image.array() - penalty_threshold).cwiseMax(0.0);
    // 先按各项自己的窗口与权重合成 WI 误差，再经过 sigmoid 和光学伴随，不增加第二次 FFT 反传。
    const Eigen::MatrixXd dLdI = ((pe_weight * residual * pe_window.array() +
                                  penalty * background * excess * penalty_window.array()) *
                                  wafer_image.array() * (1.0 - wafer_image.array())).matrix();
    return _backpropagate(dLdI, electric_field, 4.0 * _alpha);
}

Eigen::MatrixXd Gradient::epe_gradient(
    const Eigen::MatrixXd& aerial_image,
    const std::vector<Eigen::MatrixXcd>& electric_field,
    const EpsResult& eps_result,
    double threshold)
{
    const int N = _cache.N;

    // 复用项目已有的一维空间差分，先求每个像素的梯度幅值，再在 EP 处双线性插值。
    const Eigen::MatrixXd aerial_dx = LevelSetUtils::_gradient_1d(aerial_image, 1.0, 1);
    const Eigen::MatrixXd aerial_dy = LevelSetUtils::_gradient_1d(aerial_image, 1.0, 0);
    Eigen::MatrixXd gradient_magnitude = (aerial_dx.array().square() + aerial_dy.array().square()).sqrt().matrix();
    gradient_magnitude = gradient_magnitude.array().max(1e-6).matrix();

    // L=Σw*[(I_ep-τ)/G_ep]²；冻结 G_ep 后，dL/dI_ep=2w*(I_ep-τ)/G_ep²。
    Eigen::MatrixXd dLdI = Eigen::MatrixXd::Zero(N, N);
    const auto& eps = eps_result.eps;
    const auto& weights = eps_result.weight_epe;
    for (int i = 0; i < eps.rows(); ++i) {
        if (weights(i) <= 0.0) continue;
        const auto stencil = make_bilinear_stencil(N, N, eps(i, 0), eps(i, 1));
        const double aerial = bilinear_sample(aerial_image, stencil);
        const double magnitude = std::max(bilinear_sample(gradient_magnitude, stencil), 1e-6);
        const double sample_gradient = 2.0 * weights(i) * (aerial - threshold) / (magnitude * magnitude);
        bilinear_scatter(dLdI, stencil, sample_gradient);
    }

    // dLdI 已包含平方 EPE 的 2；I=|E|² 对实值 mask 求导再产生一个 2。
    return _backpropagate(dLdI, electric_field, 2.0);
}

}  // namespace litho
