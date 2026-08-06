#pragma once

#include <Eigen/Dense>
#include <vector>
#include "litho_prepare.h"

typedef double fftw_complex[2];
typedef struct fftw_plan_s* fftw_plan;

namespace litho {


struct Imaging_Result{
    Eigen::MatrixXd aerial_image;
    Eigen::MatrixXd wafer_image;
};
class Imaging {
public:
    explicit Imaging(const ImagingCache& cache);
    ~Imaging();

    // ── 成像方法 ─────────────────────────────────────────────────
    // Abbe（精确, 慢）: 对每个光源点单独累加
   

   

    
    // 光刻胶显影
    Imaging_Result compute(const Eigen::MatrixXd& mask,
                                  double threshold = 0.25,
                                  double alpha      = 50);

    // ── 计算的中间变量 ──────────────────────────────────────────────
    // 返回 const 引用，避免每次 CTM 迭代复制全部 K 个复电场
    const std::vector<Eigen::MatrixXcd>& get_electric_field() const noexcept {
        return _electric_field;
    }

private:
   



    const ImagingCache& _cache;

    // ── FFTW 资源 (Abbe 用) ────────────────────────────────────
    fftw_complex* _fft_in;
    fftw_complex* _fft_out;
    fftw_plan     _plan_fwd;
    fftw_plan     _plan_inv;



    // ── 中间缓冲 ────────────────────────────────────────────────
    mutable std::vector<Eigen::MatrixXcd> _electric_field;
    mutable Eigen::MatrixXcd _MASK,_PSF,_TEMP, _E;
    mutable Eigen::MatrixXd  _I_s;
};

}  // namespace litho
