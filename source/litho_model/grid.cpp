#include "grid.h"

namespace litho {
Grid::Grid(int grid_size_pixel, double pixel_size_nm, double NA, double wavelength)
    : _pixel_size_nm(pixel_size_nm), _grid_size(grid_size_pixel) {
    const int N = _grid_size;
    const double dx = _pixel_size_nm;

    // 像素中心 = 左边界 -N*dx/2 + (i+0.5)*dx，与 CUDA 的几何中心约定一致。
    // i 为局部像素索引 [0,N)；偶数 N 无零坐标采样点，奇数 N 的中心像素坐标为零。
    Eigen::VectorXd coord(N);
    for (int i = 0; i < N; ++i)
        coord(i) = (static_cast<double>(i) - (N - 1) * 0.5) * dx;

    _grid_coords.x.resize(N, N);
    _grid_coords.y.resize(N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            _grid_coords.x(i, j) = coord(j);   // x 沿列
            _grid_coords.y(i, j) = coord(i);   // y 沿行
        }
    }

    // FFT 周期长度为 N*dx；奇偶尺寸都按实际 N 生成频率，随后归一化到 NA/λ。
    // 空间原点位于像素间不影响零频存在；偶数频率轴的端点应保留 FFT 的非对称排列。
    _grid_coords.Fx_1d = fftF().array() / (NA / wavelength);
    _grid_coords.Fy_1d = _grid_coords.Fx_1d;
    _grid_coords.Fx_2d.resize(N, N);
    _grid_coords.Fy_2d.resize(N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            _grid_coords.Fx_2d(i, j) = _grid_coords.Fx_1d(j);
            _grid_coords.Fy_2d(i, j) = _grid_coords.Fy_1d(i);
        }
    }
}

Eigen::VectorXd Grid::fftF() const {
    const int N = _grid_size;
    const double dx = _pixel_size_nm;
    Eigen::VectorXd freq(N);
    const double df = 1.0 / (N * dx);
    // 整数频率编号 i-N/2：偶数 N 为 [-N/2,N/2)，奇数 N 为 [-(N-1)/2,(N-1)/2]。
    for (int i = 0; i < N; ++i) {
        freq(i) = (i - N / 2) * df;
    }
    return freq;
}

}  // namespace litho
