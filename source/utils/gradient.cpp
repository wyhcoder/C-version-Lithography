#include "gradient.h"
#include "fft.h"
#include <fftw3.h>
#include "level_set_utils.h"
#include <stdexcept>

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
    const auto& Hf = _cache.H_k_frequence;
    const auto& ws = _cache.source_ws;
    const auto& sv = _cache.socs_vals;
    const bool use_socs = !sv.empty();

    if (dLdI.rows() != N || dLdI.cols() != N) {
        throw std::invalid_argument("Gradient: dLdI dimensions do not match cache");
    }
    if (static_cast<int>(Hf.size()) != K ||
        (use_socs && static_cast<int>(sv.size()) != K) ||
        (!use_socs && ws.size() != K)) {
        throw std::invalid_argument("Gradient: optical kernel weights are incomplete");
    }

    // 每个 k 写独立矩阵，避免共享 gradient 的数据竞争。
    // 并行区结束后固定按 k=0..K-1 求和，使 1 线程与多线程结果一致。
    std::vector<Eigen::MatrixXd> kernel_gradients(K);

    #pragma omp parallel
    {
        // FFTW plan 只读共享；执行所用的输入、输出 buffer 必须每线程私有。
        fftw_complex* local_in  = fftw_alloc_complex(N * N);
        fftw_complex* local_out = fftw_alloc_complex(N * N);
        Eigen::MatrixXcd local_buf(N, N);

        #pragma omp for schedule(static)
        for (int k = 0; k < K; ++k) {
            const double weight = use_socs ? sv[k] : ws(k);

            // 复数伴随的输入是 E_k · dL/dI；内积本身会对它取共轭。
            local_buf = electric_field[k].array() * dLdI.array();

            FFT::ifftshift_inplace(local_buf);
            FFT::to_fftw(local_buf, local_in);
            fftw_execute_dft(_plan_fwd, local_in, local_out);
            FFT::from_fftw(local_out, local_buf);

            local_buf.array() *= Hf[k].conjugate().array();

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
//   E_k   = ifft( M_f · H_k )           （compute_aerial 里缓存的）
//
//   dL/dM(x) = 4α / Σ source_ws · Σ_k weight_k Re{ ifft( fft(E_k·dL/dI) · conj(H_k) ) }
//   其中 dL/dI = (W - T) · W · (1 - W)
//
// H_k_frequence 的 DC 在角点（与 FFT(M) 对齐）；伴随计算使用其复共轭。
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

Eigen::MatrixXd Gradient::epe_gradient(
    const Eigen::MatrixXd& aerial_image,
    EpsResult eps_result,
    double threshold,
    double dx,
    const std::vector<Eigen::MatrixXcd>& electric_field)
{
    // 保留旧接口的原有等效公式；该接口未使用 eps_result 和 dx。
    (void)eps_result;
    (void)dx;
    Eigen::MatrixXd dLdI =
        (2.0 * (aerial_image.array() - threshold)).matrix();
    return _backpropagate(dLdI, electric_field, 1.0);
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

Eigen::MatrixXd Gradient::epe_gradient(
    const Eigen::MatrixXd& aerial_image,
    const std::vector<Eigen::MatrixXcd>& electric_field,
    const EpsResult& eps_result,
    double threshold,
    double dx)
{
    const int N = _cache.N;

    // ── 计算 aerial image 的梯度模 G = |∇I|（视为常量）────────────────
    // 梯度差分步长=1（像素单位），EPE 最后乘 dx 转 nm
    Eigen::MatrixXd aerial_dx = LevelSetUtils::_gradient_1d(aerial_image, 1.0, 1);
    Eigen::MatrixXd aerial_dy = LevelSetUtils::_gradient_1d(aerial_image, 1.0, 0);
    Eigen::MatrixXd G_magnitude =
        (aerial_dx.array().square() + aerial_dy.array().square()).sqrt();

    // ── EPE 对 aerial image 的梯度 dL/dI ──────────────────────────────
    //   EPE²(p) = (I-τ)² · dx² / G²       (G = |∇I|，视为常量)
    //   d(EPE²)/dI = 2·(I-τ) · dx² / G²
    //
    //   只在 EPE 测量点处非零
    Eigen::MatrixXd dLdI = Eigen::MatrixXd::Zero(N, N);

    const auto& eps = eps_result.eps;
    const auto& w   = eps_result.weight_epe;
    int M = (int)eps.rows();
    for (int i = 0; i < M; ++i) {
        int y = (int)eps(i, 0);
        int x = (int)eps(i, 1);
        double diff = aerial_image(y, x) - threshold;
        double G = G_magnitude(y, x);
        if (G < 1e-12) G = 1e-12;   // 防止除零
        double G2 = G * G;
        dLdI(y, x) += w(i) * 2.0 * diff  / G2;
    }

    // EPE 直接对 aerial image 求导，不经过 sigmoid，因此 output_scale=1。
    return _backpropagate(dLdI, electric_field, 1.0);
}

}  // namespace litho
