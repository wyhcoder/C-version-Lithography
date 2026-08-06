#include "grid.h"

namespace litho {


// ---- 构造函数：初始化后直接算好 _grid_coords ----
Grid::Grid(int row_mask_size, double pixel_size_nm, double NA, double wavelength)
    : _pixel_size_nm(pixel_size_nm)
    , _pitch_base_size()
{
    _pitch_base_size  = _get_bit_ceil(row_mask_size); // 2^k
    _pitch = _pitch_base_size * _pixel_size_nm; // 基础pitch，单位nm
    _grid_size = _pitch_base_size + 1; // grid size奇数
    
    int N = _grid_size;  
    double dx = _pixel_size_nm;

    // 空间坐标
    Eigen::VectorXd coord(N);
    for (int i = 0; i < N; ++i)
        coord(i) = (static_cast<double>(i) - N * 0.5) * dx;

    _grid_coords.x.resize(N, N);
    _grid_coords.y.resize(N, N);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            _grid_coords.x(i, j) = coord(j);   // x 沿列
            _grid_coords.y(i, j) = coord(i);   // y 沿行
        }
    }

    // 频率轴

    _norm_pitch = _pitch * NA / wavelength;
    double specimen = 1 / _norm_pitch;

    // 这个是已经归一化的频率轴
    _grid_coords.Fx_1d = (Eigen::VectorXd::LinSpaced(N, -(N-1)/2, (N-1)/2)).array() * specimen ;
    _grid_coords.Fy_1d = (Eigen::VectorXd::LinSpaced(N, -(N-1)/2, (N-1)/2)).array() * specimen ;


    // _grid_coords.Fx_1d = fftF();
    // _grid_coords.Fy_1d = fftF();
    // double f_max = NA / wavelength;
    // _grid_coords.Fx_1d = _grid_coords.Fx_1d.array() / f_max;
    // _grid_coords.Fy_1d = _grid_coords.Fy_1d.array() / f_max;



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
    int    N  = _grid_size;
    double dx = _pixel_size_nm;
    Eigen::VectorXd freq(N);
    const double df = 1.0 / (N * dx);
    for (int i = 0; i < N; ++i) {
        freq(i) = (static_cast<double>(i) - N * 0.5) * df;
    }
    return freq;
}

int Grid::_get_bit_ceil(int size){
    if (size <= 1) return 1;
    size--;
    size |= size >> 1;
    size |= size >> 2;
    size |= size >> 4;
    size |= size >> 8;
    size |= size >> 16;
    return size + 1;
}


}  // namespace litho
