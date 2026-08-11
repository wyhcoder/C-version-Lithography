#include "litho_prepare.h"
#include "fftw3.h"
#include "fft.h"
#include "pupil.h"
#include <algorithm>
#include <complex>
#include <iostream>
#include <opencv2/objdetect.hpp>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace litho {

LithoPrepare::LithoPrepare(const Grid& grid, const Pupil& pupil, const Source& source, bool use_socs) {
    // ── 1. 空间网格 ──────────────────────────────────────────────────────
    const auto& gc   = grid.grid_coords();
    _cache.N         = grid.size();
    _cache.X         = gc.x;
    _cache.Y         = gc.y;
    _cache.Fx_2d     = gc.Fx_2d;
    _cache.Fy_2d     = gc.Fy_2d;

    //socs计算
    _use_socs = use_socs;
    // _num_socs = use_socs;

    
    // ── 2. 复光瞳 H(f) ──────────────────────────────────────────────────
    _cache.H = pupil.get_pupil();

    // ── 3. 光源点 → 物理频率 (fs, gs) + 权重 ────────────────────────────
    //   f_max = NA / λ，用于归一化坐标 → 物理频率
    // double f_max = pupil.NA() / pupil.wavelength();
    // _extract_source_points(source.source_map(), f_max);
    _extract_source_points(source.source_map());

    _fft_in  = fftw_alloc_complex(_cache.N * _cache.N);
    _fft_out = fftw_alloc_complex(_cache.N * _cache.N);
    // 用 FFTW_ESTIMATE 快速构造 plan（MEASURE 每次要预热 1-3s）
    _fft_plan = fftw_plan_dft_2d(_cache.N, _cache.N, _fft_in, _fft_out,
                                 FFTW_FORWARD, FFTW_ESTIMATE);
    _ifft_plan = fftw_plan_dft_2d(_cache.N, _cache.N, _fft_in, _fft_out,
                                 FFTW_BACKWARD, FFTW_ESTIMATE);

    // 预计算所有光源点的频移 PSF
    if (_use_socs) {
        _compute_socs_kernels();}
    else
        {_compute_psf_vectors();}
    
}

LithoPrepare::~LithoPrepare() {
    fftw_free(_fft_in);
    fftw_free(_fft_out);
    fftw_destroy_plan(_fft_plan);
    fftw_destroy_plan(_ifft_plan);
}



void LithoPrepare::_extract_source_points(const SourceMap& smap,
                                           double f_max,
                                           double min_weight) {
    int ny = static_cast<int>(smap.source_coords.size());
    int nx = static_cast<int>(smap.source_coords.size());

    // 先统计非零点数量
    int count = 0;
    for (int i = 0; i < ny; ++i)
        for (int j = 0; j < nx; ++j)
            if (smap.source_weight_map(i, j) > min_weight)
                ++count;

    if (count == 0)
        throw std::runtime_error(
            "source_weight_map 无非零点，请检查 sigma 参数与频域网格范围");

    _cache.source_fs_phys.resize(count);
    _cache.source_gs_phys.resize(count);
    _cache.source_ws.resize(count);
    _cache.source_no_ws.resize(count);

    // 展开（indexing="ij" 约定：行 → fy, 列 → fx）
    int idx = 0;
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            double w = smap.source_weight_map(i, j);
            double no_w = smap.source_weight_map(i, j);
            if (w > min_weight) {
                _cache.source_fs_phys(idx) = smap.source_coords(j) ;
                _cache.source_gs_phys(idx) = smap.source_coords(i) ;
                _cache.source_ws(idx)       = w;
                _cache.source_no_ws(idx)    = no_w;
                ++idx;
            }
        }
    }
}

/// 旧的频率轴由傅立叶变换得到的
// void LithoPrepare::_extract_source_points(const SourceMap& smap,
//                                            double f_max,
//                                            double min_weight) {
//     int ny = static_cast<int>(smap.fy_norm.size());
//     int nx = static_cast<int>(smap.fx_norm.size());

//     // 先统计非零点数量
//     int count = 0;
//     for (int i = 0; i < ny; ++i)
//         for (int j = 0; j < nx; ++j)
//             if (smap.source_weight_map(i, j) > min_weight)
//                 ++count;

//     if (count == 0)
//         throw std::runtime_error(
//             "source_weight_map 无非零点，请检查 sigma 参数与频域网格范围");

//     _cache.source_fs_phys.resize(count);
//     _cache.source_gs_phys.resize(count);
//     _cache.source_ws.resize(count);
//     _cache.source_no_ws.resize(count);

//     // 展开（indexing="ij" 约定：行 → fy, 列 → fx）
//     int idx = 0;
//     for (int i = 0; i < ny; ++i) {
//         for (int j = 0; j < nx; ++j) {
//             double w = smap.source_weight_map(i, j);
//             double no_w = smap.source_map(i, j);
//             if (w > min_weight) {
//                 _cache.source_fs_phys(idx) = smap.fx_norm(j) * f_max;
//                 _cache.source_gs_phys(idx) = smap.fy_norm(i) * f_max;
//                 _cache.source_ws(idx)       = w;
//                 _cache.source_no_ws(idx)    = no_w;
//                 ++idx;
//             }
//         }
//     }
// }


// helpers 已统一到 FFT 类（fft.h）：FFT::to_fftw / FFT::from_fftw / FFT::fftshift_inplace



// 为每个光源点预计算频移光瞳 H(f + f_s)（频域，DC 在角点，与 FFT(mask) 对齐）：
//   shift_pupil_centered(r,c) = H(r + dr, c + dc)   (双线性插值，DC 居中)
//   psf_vectors[i] = fftshift(shift_pupil_centered)  ← DC 搬到角点，直接可用
//
// 这样 compute_aerial_abbe 里可以省掉一次 FFT 和一次拷贝（直接乘）
void LithoPrepare::_compute_psf_vectors() {
    if (_is_shift) return;   // 幂等保护

    const int N  = _cache.N;
    const int Ns = (int)_cache.source_fs_phys.size();

    // 频率步长从 Fx_2d/Fy_2d 取
    const double dfx = _cache.Fx_2d(0, 1) - _cache.Fx_2d(0, 0);
    const double dfy = _cache.Fy_2d(1, 0) - _cache.Fy_2d(0, 0);

    const auto& H  = _cache.H;
    const auto& fs = _cache.source_fs_phys;
    const auto& gs = _cache.source_gs_phys;

    _cache.psf_vectors_spatial.resize(Ns);
    _cache.psf_vectors_frequence.resize(Ns);
    // 匿名函数
    auto H_at = [&](int r, int c) -> std::complex<double> {
        if (r < 0 || r >= N || c < 0 || c >= N) return {0.0, 0.0};
        return H(r, c);
    };

    for (int i = 0; i < Ns; ++i) {
        // 光源点频率 -> 像素偏移（亚像素）
        double dc = fs(i) / dfx;
        double dr = gs(i) / dfy;
        int    c0 = (int)std::floor(dc);
        int    r0 = (int)std::floor(dr);
        double tc = dc - c0;
        double tr = dr - r0;

        Eigen::MatrixXcd shift_pupil(N, N);

        // H(f + f_s) 双线性插值（DC 在中心）
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                int ra = r + r0, rb = r + r0 + 1;
                int ca = c + c0, cb = c + c0 + 1;
                shift_pupil(r, c) =
                    (1 - tr) * (1 - tc) * H_at(ra, ca) +
                    (1 - tr) *      tc  * H_at(ra, cb) +
                         tr  * (1 - tc) * H_at(rb, ca) +
                         tr  *      tc  * H_at(rb, cb);
            }
        }

        // 与 Python 对齐: H_k = fftshift(ifft2(ifftshift(shift_pupil)))
        // 1) ifftshift: DC 从中心 -> 角点，供 ifft2 使用
        FFT::ifftshift_inplace(shift_pupil);  // DC 从中心 -> 角点
        FFT::to_fftw(shift_pupil, _fft_in);
        fftw_execute_dft(_ifft_plan, _fft_in, _fft_out);

        Eigen::MatrixXcd H_spatial(N, N);
        FFT::from_fftw(_fft_out, H_spatial);
        H_spatial *= 1.0 / (double(N) * double(N)); // ifft 归一化
        // 2) fftshift: 空域结果居中（真正的 PSF，DC 峰值在图像中心）
        FFT::fftshift_inplace(H_spatial); // DC 从角点 -> 中心
        _cache.psf_vectors_spatial[i] = H_spatial;

        // 频域核 = ifftshift(shift_pupil)（DC 居中 → 角点）
        // 成像时 _MASK 也是 DC 在角点（见 imaging.cpp:56），两者才能逐元素相乘
        // 注: fft2(ifft2c(X)) 往返等价于 ifftshift(X)，无需再做 FFT
        Eigen::MatrixXcd H_freq = shift_pupil;
        FFT::ifftshift_inplace(H_freq);           // DC 居中 → 角点
        _cache.psf_vectors_frequence[i] = std::move(H_freq);
    }
    _cache.H_k_frequence = _cache.psf_vectors_frequence;

    _is_shift = true;
}

void LithoPrepare::_compute_socs_kernels(){
    const int N  = _cache.N;
    const int Ns = (int)_cache.source_fs_phys.size();

    // 频率步长从 Fx_2d/Fy_2d 取
    const double dfx = _cache.Fx_2d(0, 1) - _cache.Fx_2d(0, 0);
    const double dfy = _cache.Fy_2d(1, 0) - _cache.Fy_2d(0, 0);

    const auto& H  = _cache.H;
    const auto& fs = _cache.source_fs_phys;
    const auto& gs = _cache.source_gs_phys;
    const auto& ws = _cache.source_ws;

    const int Nr = (int)H.rows();
    const int Nc = (int)H.cols();
    Eigen::MatrixXcd A(Nr * Nc, Ns);
    // lambda函数
    auto H_at = [&](int r, int c) -> std::complex<double> {
        if (r < 0 || r >= N || c < 0 || c >= N) return {0.0, 0.0};
        return H(r, c);
    };

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < Ns; ++i) {
        // 光源点频率 -> 像素偏移（亚像素）
        double dc = fs(i) / dfx;
        double dr = gs(i) / dfy;
        int    c0 = (int)std::floor(dc);
        int    r0 = (int)std::floor(dr);
        double tc = dc - c0;
        double tr = dr - r0;

        Eigen::MatrixXcd shift_pupil(N, N);

        // H(f + f_s) 双线性插值（DC 在中心）
        for (int r = 0; r < N; ++r) {
            for (int c = 0; c < N; ++c) {
                int ra = r + r0, rb = r + r0 + 1;
                int ca = c + c0, cb = c + c0 + 1;
                shift_pupil(r, c) =
                    (1 - tr) * (1 - tc) * H_at(ra, ca) +
                    (1 - tr) *      tc  * H_at(ra, cb) +
                         tr  * (1 - tc) * H_at(rb, ca) +
                         tr  *      tc  * H_at(rb, cb);
            }
        }

        const double w_scale = std::sqrt(ws(i));
        Eigen::Map<const Eigen::VectorXcd> flat(shift_pupil.data(), Nr * Nc);
        A.col(i) = w_scale * flat;
    }

    // ── 2. Thin SVD: A = U Σ Vᴴ ─────────────────────────────────────
    // BDCSVD 比 JacobiSVD 快 10-100 倍，精度对成像足够
    Eigen::BDCSVD<Eigen::MatrixXcd,Eigen::ComputeThinU> svd(A);
    const auto& U = svd.matrixU();          // [N² × Ns]
    const auto& S = svd.singularValues();   // [Ns]，降序

    // ── 3. 取前 K 个核 ──────────────────────────────────────────────
    const int K_request = _num_socs;
    const int K_use = std::min(K_request > 0 ? K_request : Ns, (int)S.size());

    _cache.socs_vals.resize(K_use);
    _cache.socs_kernels_frequence.resize(K_use);
    _cache.socs_kernels_spatial.resize(K_use);
    // 统计能量占比
    double total_energy = 0.0, top_energy = 0.0;
    for (int k = 0; k < S.size(); ++k) total_energy += S(k) * S(k);

    #ifdef _OPENMP
    std::cout << "OpenMP 已启用，版本：" << _OPENMP << '\n';
    #else
    std::cout << "OpenMP 未启用，当前为单线程\n";
    #endif

    #pragma omp parallel
    {
        fftw_complex* local_in = fftw_alloc_complex(N * N);
        fftw_complex* local_out = fftw_alloc_complex(N * N);
        
        // 为分配循环任务，reduction避免公共的 top_energy 发生数据冲突，schedule(static) 表示静态分配的线程数
        #pragma omp for reduction(+:top_energy) schedule(static)
        
        for (int k = 0; k < K_use; ++k){
            _cache.socs_vals[k] = S(k) * S(k);          // λ_k = σ_k²
            top_energy += _cache.socs_vals[k];
            Eigen::Map<const Eigen::MatrixXcd> kernel_freq(
            U.col(k).data(), N, N);
            Eigen::MatrixXcd kernel = kernel_freq;
            FFT::ifftshift_inplace(kernel);
            // 不能使用共享的_fft_in/_fft_out
            FFT::to_fftw(kernel, local_in);
            fftw_execute_dft(_ifft_plan, local_in, local_out);
            FFT::from_fftw(local_out, kernel);

            kernel *= 1.0 / (double(N) * double(N));
            FFT::fftshift_inplace(kernel);
            _cache.socs_kernels_spatial[k] = std::move(kernel);
            
            Eigen::MatrixXcd kernel_fft = kernel_freq;
            FFT::ifftshift_inplace(kernel_fft);          // DC 居中 → 角点
            _cache.socs_kernels_frequence[k] = std::move(kernel_fft);

        }
        fftw_free(local_in);
        fftw_free(local_out);

    }
    // 串行版本的实现
    // for (int k = 0; k < K_use; ++k) {
    //     _cache.socs_vals[k] = S(k) * S(k);          // λ_k = σ_k²
    //     top_energy += _cache.socs_vals[k];

    //     // U[:,k] reshape 为 N×N（列优先，居中频域核）
    //     Eigen::Map<const Eigen::MatrixXcd> kernel_freq(
    //         U.col(k).data(), N, N);
        
    //     // 空域核 = ifft2c(kernel_freq) = fftshift(ifft2(ifftshift(kernel_freq))/N²)
    //     Eigen::MatrixXcd kernel = kernel_freq;
    //     FFT::ifftshift_inplace(kernel);               // DC 居中 → 角点
    //     FFT::to_fftw(kernel, _fft_in);
    //     fftw_execute_dft(_ifft_plan, _fft_in, _fft_out);
    //     FFT::from_fftw(_fft_out, kernel);
    //     kernel *= 1.0 / (double(N) * double(N));
    //     FFT::fftshift_inplace(kernel);               // DC 角点 → 中心
    //     _cache.socs_kernels_spatial[k] = kernel;

    //     // 频域核 = ifftshift(kernel_freq)（DC 居中 → 角点）
    //     // 成像时 _MASK 也是 DC 在角点（见 imaging.cpp:56），两者才能逐元素相乘
    //     // 注: fft2(ifft2c(X)) 往返等价于 ifftshift(X)，无需再做 2 次 FFT
    //     Eigen::MatrixXcd kernel_fft = kernel_freq;
    //     FFT::ifftshift_inplace(kernel_fft);          // DC 居中 → 角点
    //     _cache.socs_kernels_frequence[k] = std::move(kernel_fft);
    // }
    _cache.H_K_spatial = _cache.socs_kernels_spatial;        // 共享引用（无拷贝）
    _cache.H_k_frequence = _cache.socs_kernels_frequence;
    std::cout << "SOCS: " << Ns << " source pts → top " << K_use
              << " kernels capture "
              << (100.0 * top_energy / total_energy) << "% energy"
              << "  (σ₁=" << S(0) << ", σₖ=" << S(K_use - 1) << ")"
              << std::endl;
}



}  // namespace litho
