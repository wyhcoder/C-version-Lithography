#pragma once

#include <Eigen/Dense>

#include <vector>

#include "litho_prepare.h"

namespace litho {

struct Imaging_Result {
    Eigen::MatrixXd aerial_image;
    Eigen::MatrixXd wafer_image;
};

class Imaging {
public:
    explicit Imaging(const ImagingCache& cache);
    ~Imaging();

    Imaging(const Imaging&) = delete;
    Imaging& operator=(const Imaging&) = delete;

    // 光刻胶显影
    Imaging_Result compute(const Eigen::MatrixXd& mask,
                           double threshold = 0.25,
                           double alpha = 50.0);

    // ── 计算的中间变量 ──────────────────────────────────────────────
    // 返回 const 引用，避免每次 CTM 迭代复制全部 K 个复电场
    const std::vector<Eigen::MatrixXcd>& get_electric_field() const noexcept {
        return _electric_field;
    }

private:
    const ImagingCache& _cache;

    // FFTW 输入、输出缓冲区及正/逆变换计划。
    fftw_complex* _fft_in = nullptr;
    fftw_complex* _fft_out = nullptr;
    fftw_plan _plan_fwd = nullptr;
    fftw_plan _plan_inv = nullptr;

    // 最近一次成像计算的频域掩模和各核复电场。
    std::vector<Eigen::MatrixXcd> _electric_field;
    Eigen::MatrixXcd _mask_frequency;
};

}  // namespace litho
