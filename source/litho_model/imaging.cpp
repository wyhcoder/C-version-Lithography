#include "imaging.h"

#include "fft.h"

#include <fftw3.h>

#include <cmath>
#include <complex>
#include <cstddef>
#include <new>
#include <stdexcept>
#include <utility>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace litho {

Imaging::Imaging(const ImagingCache& cache)
    : _cache(cache) {
    const int N = _cache.N;
    if (N <= 0) {
        throw std::invalid_argument("Imaging: cache.N must be positive");
    }

    const std::size_t element_count = static_cast<std::size_t>(N) * N;
    _fft_in = fftw_alloc_complex(element_count);
    _fft_out = fftw_alloc_complex(element_count);

    if (_fft_in == nullptr || _fft_out == nullptr) {
        if (_fft_in != nullptr) fftw_free(_fft_in);
        if (_fft_out != nullptr) fftw_free(_fft_out);
        throw std::bad_alloc();
    }

    _plan_fwd = fftw_plan_dft_2d(N, N, _fft_in, _fft_out,
                                 FFTW_FORWARD, FFTW_ESTIMATE);
    _plan_inv = fftw_plan_dft_2d(N, N, _fft_in, _fft_out,
                                 FFTW_BACKWARD, FFTW_ESTIMATE);

    if (_plan_fwd == nullptr || _plan_inv == nullptr) {
        if (_plan_fwd != nullptr) fftw_destroy_plan(_plan_fwd);
        if (_plan_inv != nullptr) fftw_destroy_plan(_plan_inv);
        fftw_free(_fft_in);
        fftw_free(_fft_out);
        throw std::runtime_error("Imaging: failed to create FFTW plans");
    }

    _mask_frequency.resize(N, N);
}

Imaging::~Imaging() {
    fftw_destroy_plan(_plan_fwd);
    fftw_destroy_plan(_plan_inv);
    fftw_free(_fft_in);
    fftw_free(_fft_out);
}

Imaging_Result Imaging::compute(const Eigen::MatrixXd& mask,
                                double threshold,
                                double alpha) {
    const int N = _cache.N;
    const auto& source_weights = _cache.source_ws;
    const auto& kernels_frequency = _cache.H_k_frequence;
    const auto& socs_values = _cache.socs_vals;
    const int kernel_count = static_cast<int>(kernels_frequency.size());
    // 防御性代码
    if (mask.rows() != N || mask.cols() != N) {
        throw std::invalid_argument("Imaging: mask dimensions must match cache.N");
    }
    if (!mask.allFinite() || !std::isfinite(threshold) || !std::isfinite(alpha)) {
        throw std::invalid_argument("Imaging: mask, threshold, and alpha must be finite");
    }
    if (kernel_count == 0) {
        throw std::invalid_argument("Imaging: no optical kernels available");
    }

    // Abbe 模式不生成 socs_vals；SOCS 模式为每个相干核保存一个特征值。
    const bool use_socs = !socs_values.empty();
    if (use_socs && static_cast<int>(socs_values.size()) != kernel_count) {
        throw std::invalid_argument("Imaging: SOCS kernel/value counts do not match");
    }
    if (!use_socs && source_weights.size() != kernel_count) {
        throw std::invalid_argument("Imaging: Abbe kernel/source counts do not match");
    }

    const double source_weight_sum = source_weights.sum();
    if (!source_weights.allFinite() ||
        (source_weights.array() < 0.0).any() ||
        !std::isfinite(source_weight_sum) || source_weight_sum <= 0.0) {
        throw std::invalid_argument(
            "Imaging: source weights must be finite, nonnegative, and have a positive sum");
    }
    for (double value : socs_values) {
        if (!std::isfinite(value) || value < 0.0) {
            throw std::invalid_argument(
                "Imaging: SOCS values must be finite and nonnegative");
        }
    }

    for (const auto& kernel : kernels_frequency) {
        if (kernel.rows() != N || kernel.cols() != N) {
            throw std::invalid_argument(
                "Imaging: optical kernel dimensions must match cache.N");
        }
    }

    _electric_field.resize(kernel_count);
    const std::size_t element_count = static_cast<std::size_t>(N) * N;
    const double inverse_element_count = 1.0 / static_cast<double>(element_count);

    // 掩模频谱只计算一次，之后与每个 Abbe/SOCS 频域核逐元素相乘。
    Eigen::MatrixXcd mask_complex = mask.cast<std::complex<double>>();
    FFT::ifftshift_inplace(mask_complex);  // DC 从中心移到角点
    FFT::to_fftw(mask_complex, _fft_in);
    fftw_execute_dft(_plan_fwd, _fft_in, _fft_out);
    FFT::from_fftw(_fft_out, _mask_frequency);

    Eigen::MatrixXd aerial_image = Eigen::MatrixXd::Zero(N, N);

    // 每个光学核相互独立。若已经处于 MEEF 外层并行区，则不再创建内层线程组，
    // 避免“外层线程数 × 内层线程数”造成线程过度订阅。
#ifdef _OPENMP
    const bool enable_inner_parallel = (omp_in_parallel() == 0);
#pragma omp parallel if(enable_inner_parallel)
#endif
    {
        // FFTW plan 可共享执行，但输入和输出缓冲区必须由每个线程独立持有。
        fftw_complex* local_in = fftw_alloc_complex(element_count);
        fftw_complex* local_out = fftw_alloc_complex(element_count);
        Eigen::MatrixXcd local_temp(N, N);

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (int k = 0; k < kernel_count; ++k) {
            local_temp = _mask_frequency.array() * kernels_frequency[k].array();

            FFT::to_fftw(local_temp, local_in);
            fftw_execute_dft(_plan_inv, local_in, local_out);

            // 每个线程写入不同的 k，不会产生数据竞争。
            Eigen::MatrixXcd& electric_field = _electric_field[k];
            electric_field.resize(N, N);
            FFT::from_fftw(local_out, electric_field);
            electric_field *= inverse_element_count;
            FFT::fftshift_inplace(electric_field); // 脚点搬到中心去
        }

        fftw_free(local_in); // 并行结束后销毁内存
        fftw_free(local_out); // 并行结束后销毁内存
    }

    // 固定按 k 顺序累加，保证单线程和多线程采用相同的浮点求和顺序。
    for (int k = 0; k < kernel_count; ++k) {
        const double weight = use_socs ? socs_values[k] : source_weights(k);
        const Eigen::MatrixXcd& electric_field = _electric_field[k];
        aerial_image.array() += weight *
            (electric_field.real().array().square() +
             electric_field.imag().array().square());
    }

    aerial_image /= source_weight_sum;
    Eigen::MatrixXd wafer_image =
        (1.0 /
         (1.0 + (-alpha * (aerial_image.array() - threshold)).exp()))
            .matrix();

    return {std::move(aerial_image), std::move(wafer_image)};
}

}  // namespace litho
