#include "imaging.h"
#include "fft.h"
#include "litho_prepare.h"

#include <fftw3.h>
#ifdef _OPENMP
#include <omp.h>
#endif


namespace litho {

Imaging::Imaging(const ImagingCache& cache)
    : _cache(cache)
{
    int N  = _cache.N;
    int sz = N * N;

    _fft_in  = fftw_alloc_complex(sz);
    _fft_out = fftw_alloc_complex(sz);
    _plan_fwd = fftw_plan_dft_2d(N, N, _fft_in, _fft_out,
                                  FFTW_FORWARD,  FFTW_ESTIMATE);
    _plan_inv = fftw_plan_dft_2d(N, N, _fft_in, _fft_out,
                                  FFTW_BACKWARD, FFTW_ESTIMATE);

    _MASK.resize(N, N);
    _E.resize(N, N);
    _TEMP.resize(N, N);
    _PSF.resize(N, N);


    int Ns = static_cast<int>(_cache.source_fs_phys.size());
    
}

Imaging::~Imaging() {
    fftw_destroy_plan(_plan_fwd);
    fftw_destroy_plan(_plan_inv);
    fftw_free(_fft_in);
    fftw_free(_fft_out);
}

// helpers 已统一到 FFT 类（fft.h）：FFT::to_fftw / FFT::from_fftw / FFT::fftshift_inplace

Imaging_Result Imaging::compute( const Eigen::MatrixXd& mask,double threshold, double alpha)  {
    const int    N   = _cache.N;
    const double ivN = 1.0 / (double(N) * double(N));
    const auto&  ws  = _cache.source_ws;
    const auto&  h_k = _cache.H_k_frequence;
    const int    K   = (int)h_k.size();

    _electric_field.resize(K);
    // 路径判别：核数 == 源点数 → Abbe（每核对应 1 个源点，需乘 w_s）
    //          核数 <  源点数 → SOCS（核已吸收 √w·σ，直接 |E|² 累加）
    const bool is_abbe = (K == (int)ws.size());

    // ── 1. mask 的 FFT 只算一次 ───────────────────────────────────────
    Eigen::MatrixXcd mask_c = mask.cast<std::complex<double>>();
    FFT::ifftshift_inplace(mask_c); // DC 中心 -> 角点
    FFT::to_fftw(mask_c, _fft_in);
    fftw_execute_dft(_plan_fwd, _fft_in, _fft_out);
    FFT::from_fftw(_fft_out, _MASK);          // DC 在角点

    Eigen::MatrixXd aerial = Eigen::MatrixXd::Zero(N, N);

    // 每个 SOCS 核相互独立：平常在内层并行；若已处于 MEEF 外层并行区，
    // 此处退化为单线程，避免 线程数×线程数 的嵌套过度订阅。
    bool enable_inner_parallel = true;
#ifdef _OPENMP
    enable_inner_parallel = (omp_in_parallel() == 0);
#endif
    #pragma omp parallel if(enable_inner_parallel)
    {
        // FFTW plan 只读共享；输入/输出 buffer 必须每线程独立。
        fftw_complex* local_in  = fftw_alloc_complex(N * N);
        fftw_complex* local_out = fftw_alloc_complex(N * N);
        Eigen::MatrixXcd local_temp(N, N);

        #pragma omp for schedule(static)
        for (int s = 0; s < K; ++s) {
            local_temp = _MASK.array() * h_k[s].array();

            FFT::to_fftw(local_temp, local_in);
            fftw_execute_dft(_plan_inv, local_in, local_out);

            // 每个线程写不同的 s，不存在写冲突。
            Eigen::MatrixXcd& electric_field = _electric_field[s];
            electric_field.resize(N, N);
            FFT::from_fftw(local_out, electric_field);
            electric_field *= ivN;
            FFT::fftshift_inplace(electric_field);
        }

        fftw_free(local_in);
        fftw_free(local_out);
    }

    // 固定按 s 顺序累加，保证 1 线程和多线程采用相同浮点求和顺序。
    for (int s = 0; s < K; ++s) {
        const double w = is_abbe ? ws(s) : _cache.socs_vals[s];
        const Eigen::MatrixXcd& electric_field = _electric_field[s];
        aerial.array() += w *
            (electric_field.real().array().square() +
             electric_field.imag().array().square());
    }

    aerial /= _cache.source_ws.sum();
    Eigen::MatrixXd wafer =
        (1.0 / (1.0 + (-alpha * (aerial.array() - threshold)).exp())).matrix();

    return {aerial, wafer};
}




}  // namespace litho
