#include "full_chip_imaging_compare.h"
#include "fft.h"
#include "lithography_simulator.h"
#include "loss.h"

#include <cmath>
#include <iostream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace litho;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

SimulationParameters make_params() {
    SimulationParameters params{};
    params.system = {193.0, 8.0, 128};
    params.optics.na = 1.35;
    params.optics.refractive_index = 1.44;
    params.optics.aberrations = {{4, 0.07}, {5, 0.03}, {7, 0.02}};
    params.source = {"annular", 0.6, 0.9};
    params.resist = {"sigmoid", 0.25, 12};
    return params;
}

Eigen::MatrixXd make_mask(int n) {
    Eigen::MatrixXd mask = Eigen::MatrixXd::Constant(n, n, 0.15);
    mask.block(n / 4, n / 5, n / 3, n / 2).setConstant(0.85);
    mask.block(n / 2, n / 3, n / 8, n / 9).setConstant(0.35);
    // 靠近周期边界的灰度条带检验核补零后频率位置是否正确。
    mask.block(0, n / 2, 3, n / 5).setConstant(0.6);
    return mask;
}

// 检查奇偶、非 2 的幂及小尺寸：像素中心关于几何中心对称，频率轴与 FFT 移位的零频位置一致。
void check_grid_sizes() {
    for (int n : {1, 4, 5, 62, 63, 256, 257}) {
        constexpr double dx = 7.5; // nm；非整数像素尺寸也应保持精确的空间/频率间隔。
        const Grid grid(n, dx, 1.35, 193.0);
        const auto& coords = grid.grid_coords();
        require(grid.size() == n && coords.x.rows() == n && coords.Fx_2d.cols() == n, "exact grid rounded the matrix dimensions");
        const double df = 193.0 / (n * dx * 1.35);
        require(coords.Fx_1d(n / 2) == 0.0, "DC does not lie at the centered FFT index");
        require(std::abs(coords.Fx_1d(0) + n / 2 * df) < 1e-12, "wrong negative FFT endpoint");
        require(std::abs(coords.Fx_1d(n - 1) - (n - 1 - n / 2) * df) < 1e-12, "wrong positive FFT endpoint");
        require(std::abs(coords.x(0, 0) + (n - 1) * dx / 2) < 1e-12, "wrong left pixel center");
        require(std::abs(coords.x(0, n - 1) - (n - 1) * dx / 2) < 1e-12, "wrong right pixel center");
        require(coords.x(0, n / 2) == (n % 2 == 0 ? dx / 2 : 0.0), "wrong spatial origin convention");
        for (int i = 0; i < n; ++i) {
            require(coords.x(0, i) == -coords.x(0, n - 1 - i), "pixel centers are not geometrically symmetric");
            require(coords.y(i, 0) == coords.x(0, i), "x/y spatial axes disagree");
            require(coords.Fx_2d(n / 2, i) == coords.Fx_1d(i), "frequency x axis does not follow columns");
            require(coords.Fy_2d(i, n / 2) == coords.Fy_1d(i), "frequency y axis does not follow rows");
            if (i > 0) {
                require(coords.x(0, i) - coords.x(0, i - 1) == dx, "wrong pixel spacing");
                require(std::abs(coords.Fx_1d(i) - coords.Fx_1d(i - 1) - df) < 1e-12, "wrong frequency spacing");
            }
        }

        // 角点的单位脉冲代表零频位置，检验两种 fftshift 接口和反向移位都与频率轴一致。
        Eigen::MatrixXcd corner = Eigen::MatrixXcd::Zero(n, n);
        corner(0, 0) = 1.0;
        const auto returned = FFT::fftshift(corner);
        auto inplace = corner;
        FFT::fftshift_inplace(inplace);
        require(returned(n / 2, n / 2) == 1.0, "fftshift did not move DC to the frequency origin");
        require((returned - inplace).norm() == 0.0, "fftshift interfaces disagree for this parity");
        FFT::ifftshift_inplace(inplace);
        require((inplace - corner).norm() == 0.0, "FFT shifts are not inverse operations");
    }
}

void check_compact_socs() {
    const auto params = make_params();
    const Grid grid(128, 8.0, 1.35, 193.0);
    const auto& coords = grid.grid_coords();
    const Pupil pupil(LithographySimulator::_make_pupil_params(params), coords.Fx_2d, coords.Fy_2d);
    const Source source(LithographySimulator::_make_source_params(params, coords.Fx_1d));
    const LithoPrepare dense(grid, pupil, source, true);
    const LithoPrepare compact(grid, pupil, source, true, true);
    require(dense.cache().N == compact.cache().N, "compact path changed FFT dimensions");
    for (int k = 0; k < dense.cache().socs_eigenvalues.size(); ++k) {
        require(std::abs(dense.cache().socs_eigenvalues[k] - compact.cache().socs_eigenvalues[k]) < 1e-9,
                "compact path changed SOCS eigenvalues");
    }
    Imaging dense_imaging(dense.cache()), compact_imaging(compact.cache());
    const auto mask = make_mask(128);
    const auto a = dense_imaging.compute(mask, 0.25, 12.0);
    const auto b = compact_imaging.compute(mask, 0.25, 12.0);
    // SVD 相位可不同，所以比较成像可观测量，而不直接比较各个复数特征核。
    require((a.aerial_image - b.aerial_image).cwiseAbs().maxCoeff() < 1e-10, "compact path changed aerial image");
    require((a.wafer_image - b.wafer_image).cwiseAbs().maxCoeff() < 1e-10, "compact path changed wafer image");
}

void check_full_frame_comparison(int n) {
    const auto params = make_params();
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(n, n);
    target.block(12, 9, 32, 28).setOnes();
    const auto gray = make_mask(n);
    const auto comparison = compare_full_chip_imaging(params, target, gray, n);
    require(comparison.grid_size == n, "optical grid must match full target dimensions");
    require((comparison.target_mask - target).norm() == 0.0, "target mask was changed");
    require((comparison.optimized_mask - gray).norm() == 0.0, "optimized mask was binarized or otherwise changed");

    // 用独立构造的完整 SVD 全图模型校验两个输入的成像，避免只检查导出的 mask 而漏掉成像前的转换。
    const Grid grid(n, 8.0, 1.35, 193.0);
    const auto& coords = grid.grid_coords();
    const Pupil pupil(LithographySimulator::_make_pupil_params(params), coords.Fx_2d, coords.Fy_2d);
    const Source source(LithographySimulator::_make_source_params(params, coords.Fx_1d));
    const LithoPrepare prepare(grid, pupil, source, true);
    Imaging imaging(prepare.cache());
    const auto a = imaging.compute(target, params.resist.threshold, params.resist.alpha);
    const auto b = imaging.compute(gray, params.resist.threshold, params.resist.alpha);
    require((a.aerial_image - comparison.target_image.aerial_image).cwiseAbs().maxCoeff() < 1e-10, "wrong target AI");
    require((a.wafer_image - comparison.target_image.wafer_image).cwiseAbs().maxCoeff() < 1e-10, "wrong target WI");
    require((b.aerial_image - comparison.optimized_image.aerial_image).cwiseAbs().maxCoeff() < 1e-10, "wrong gray AI");
    require((b.wafer_image - comparison.optimized_image.wafer_image).cwiseAbs().maxCoeff() < 1e-10, "wrong gray WI");
    require(std::abs(comparison.target_pe - Loss::pe_loss(a.wafer_image, target)) < 1e-9, "wrong target PE");
    require(std::abs(comparison.optimized_pe - Loss::pe_loss(b.wafer_image, target)) < 1e-9, "wrong optimized PE");
}

int main() {
#ifdef _OPENMP
    omp_set_num_threads(2);
#endif
    check_grid_sizes();
    check_compact_socs();
    check_full_frame_comparison(63);
    check_full_frame_comparison(64);
    std::cout << "Compact SOCS equivalence and continuous-mask imaging comparison passed.\n";
}
