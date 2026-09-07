#include "fft.h"
#include <stdexcept>

namespace litho {

// ── helper: Eigen::MatrixXd → flat fftw_complex array (imag=0) ──────────
static void eigen_to_fftw(const Eigen::MatrixXd& A, fftw_complex* out) {
    int rows = static_cast<int>(A.rows());
    int cols = static_cast<int>(A.cols());
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            int idx = r * cols + c;
            out[idx][0] = A(r, c);   // real
            out[idx][1] = 0.0;       // imag
        }
}

// ── helper: flat fftw_complex array → Eigen::MatrixXcd ──────────────────
static Eigen::MatrixXcd fftw_to_eigen(const fftw_complex* in,
                                       int rows, int cols) {
    Eigen::MatrixXcd out(rows, cols);
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            int idx = r * cols + c;
            out(r, c) = std::complex<double>(in[idx][0], in[idx][1]);
        }
    return out;
}

// ── 2D 前向 FFT ─────────────────────────────────────────────────────────
Eigen::MatrixXcd FFT::fft2(const Eigen::MatrixXd& A) {
    int rows = static_cast<int>(A.rows());
    int cols = static_cast<int>(A.cols());

    fftw_complex *in  = fftw_alloc_complex(rows * cols);
    fftw_complex *out = fftw_alloc_complex(rows * cols);

    eigen_to_fftw(A, in);

    fftw_plan p = fftw_plan_dft_2d(rows, cols, in, out,
                                   FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    Eigen::MatrixXcd result = fftw_to_eigen(out, rows, cols);

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    return result;
}

// ── 2D 前向 FFT（复数输入）───────────────────────────────────────────
Eigen::MatrixXcd FFT::fft2(const Eigen::MatrixXcd& A) {
    int rows = static_cast<int>(A.rows());
    int cols = static_cast<int>(A.cols());

    fftw_complex *in  = fftw_alloc_complex(rows * cols);
    fftw_complex *out = fftw_alloc_complex(rows * cols);

    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            int idx = r * cols + c;
            in[idx][0] = A(r, c).real();
            in[idx][1] = A(r, c).imag();
        }

    fftw_plan p = fftw_plan_dft_2d(rows, cols, in, out,
                                   FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    Eigen::MatrixXcd result = fftw_to_eigen(out, rows, cols);

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    return result;
}
Eigen::MatrixXcd FFT::ifft2(const Eigen::MatrixXcd& Asp) {
    int rows = static_cast<int>(Asp.rows());
    int cols = static_cast<int>(Asp.cols());
    double N = rows * cols;

    fftw_complex *in  = fftw_alloc_complex(rows * cols);
    fftw_complex *out = fftw_alloc_complex(rows * cols);

    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            int idx = r * cols + c;
            in[idx][0] = Asp(r, c).real();
            in[idx][1] = Asp(r, c).imag();
        }

    fftw_plan p = fftw_plan_dft_2d(rows, cols, in, out,
                                   FFTW_BACKWARD, FFTW_ESTIMATE);
    fftw_execute(p);

    Eigen::MatrixXcd result = fftw_to_eigen(out, rows, cols);
    result /= N;

    fftw_destroy_plan(p);
    fftw_free(in);
    fftw_free(out);
    return result;
}

// ── roll 辅助：等价 np.roll(x, shift, axis)，Y(i) = X((i - shift) mod n) ──
static Eigen::MatrixXcd roll_rows(const Eigen::MatrixXcd& X, int shift) {
    const int R = static_cast<int>(X.rows());
    shift = ((shift % R) + R) % R; // 将 shift 映射到 [0, R)
    if (shift == 0) return X;   
    Eigen::MatrixXcd Y(X.rows(), X.cols());
    Y.topRows(shift)        = X.bottomRows(shift); // 移动方向是向下的
    Y.bottomRows(R - shift) = X.topRows(R - shift);
    return Y;
}

static Eigen::MatrixXcd roll_cols(const Eigen::MatrixXcd& X, int shift) {
    const int C = static_cast<int>(X.cols());
    shift = ((shift % C) + C) % C;
    if (shift == 0) return X;
    Eigen::MatrixXcd Y(X.rows(), X.cols());
    Y.leftCols(shift)        = X.rightCols(shift);
    Y.rightCols(C - shift)   = X.leftCols(C - shift);
    return Y;
}

// ── fftshift：零频从角点搬到中心（np.fft.fftshift，np是用ceil向上取整）──
Eigen::MatrixXcd FFT::fftshift(const Eigen::MatrixXcd& X) {
    const int r2 = static_cast<int>(X.rows() + 1) / 2;
    const int c2 = static_cast<int>(X.cols() + 1) / 2;
    return roll_cols(roll_rows(X, r2), c2);
}

// ── 与外部 FFTW plan 配合的工具 ─────────────────────────────────────
void FFT::to_fftw(const Eigen::MatrixXcd& src, fftw_complex* dst) {
    int r = static_cast<int>(src.rows()), c = static_cast<int>(src.cols());
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j) {
            dst[i * c + j][0] = src(i, j).real();
            dst[i * c + j][1] = src(i, j).imag();
        }
}

void FFT::from_fftw(const fftw_complex* src, Eigen::MatrixXcd& dst) {
    int r = static_cast<int>(dst.rows()), c = static_cast<int>(dst.cols());
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
            dst(i, j) = std::complex<double>(src[i * c + j][0],
                                             src[i * c + j][1]);
}

void FFT::fftshift_inplace(Eigen::MatrixXcd& X) {
    // numpy的fftshift 是向下取整
    const int r2 = static_cast<int>(X.rows()  / 2);
    const int c2 = static_cast<int>(X.cols()  / 2);
    X = roll_cols(roll_rows(X, r2), c2);
}

void FFT::ifftshift_inplace(Eigen::MatrixXcd& X) {
    // numpy的ifftshift 是向上取整，和fftshift形成一个互逆操作
    const int r2 = static_cast<int>(X.rows()+1)  / 2;
    const int c2 = static_cast<int>(X.cols()+1)  / 2;
    X = roll_cols(roll_rows(X, r2), c2);
}


// ── fftconvolve：频域卷积 = ifftshift(ifft2(fft2(A) .* fft2(B))) ────
Eigen::MatrixXcd fftconvolve(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols())
        throw std::invalid_argument("fftconvolve: A 和 B 尺寸必须一致");

    Eigen::MatrixXcd Asp = FFT::fft2(A);
    Eigen::MatrixXcd Bsp = FFT::fft2(B);

    // 频域逐元素相乘
    Eigen::MatrixXcd C = Asp.array() * Bsp.array();

    Eigen::MatrixXcd c = FFT::ifft2(C);
    return FFT::fftshift(c);
}

}  // namespace litho
