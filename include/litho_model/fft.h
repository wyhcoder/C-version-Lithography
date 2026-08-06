#pragma once

#include <Eigen/Dense>
#include <fftw3.h>

namespace litho {

// Eigen + FFTW3 的轻量封装
// 约定：forward / inverse 的归一化行为与 numpy.fft 一致
struct FFT {
    // 2D 前向 FFT（实数输入）
    static Eigen::MatrixXcd fft2(const Eigen::MatrixXd& A);

    // 2D 前向 FFT（复数输入）
    static Eigen::MatrixXcd fft2(const Eigen::MatrixXcd& A);

    // 2D 逆 FFT：等价于 numpy.fft.ifft2（已除 N）
    static Eigen::MatrixXcd ifft2(const Eigen::MatrixXcd& Asp);

    // 中心化零频：numpy.fft.fftshift（返回新矩阵）
    static Eigen::MatrixXcd fftshift(const Eigen::MatrixXcd& X);

    // ── 与外部 FFTW plan 配合使用的轻量工具 ───────────────────────
    // Eigen 矩阵 <-> fftw_complex 数组互拷（行优先，假设 dst/src 容量 = rows*cols）
    static void to_fftw(const Eigen::MatrixXcd& src, fftw_complex* dst);
    static void from_fftw(const fftw_complex* src, Eigen::MatrixXcd& dst);

    // 原地 fftshift：把零频从角点搬到中心（np.fft.fftshift 语义，支持任意奇偶尺寸）
    static void fftshift_inplace(Eigen::MatrixXcd& X);

    // 原地 ifftshift：把零频从中心搬回角点（np.fft.ifftshift 语义，支持任意奇偶尺寸）
    // 注意：奇数尺寸时 ifftshift ≠ fftshift，不能混用
    static void ifftshift_inplace(Eigen::MatrixXcd& X);
};

// FFT 频域卷积（直接对标提问里的 Python 函数）
//  C = ifftshift( ifft2( fft2(A) .* fft2(B) ) )
Eigen::MatrixXcd fftconvolve(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B);

}  // namespace litho
