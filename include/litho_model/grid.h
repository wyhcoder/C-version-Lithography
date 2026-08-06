#pragma once

#include <Eigen/Dense>

namespace litho {

struct GridResult {
    Eigen::MatrixXd x, y;         // 空间域 2D
    Eigen::VectorXd Fx_1d, Fy_1d; // 频率域 1D
    Eigen::MatrixXd Fx_2d, Fy_2d; // 频率域 2D
};

class Grid {
public:
    Grid(int row_mask_size, double pixel_size_nm, double NA, double wavelength);

    Eigen::VectorXd fftF() const;
    GridResult      get_grid() const { return _grid_coords; }
    const GridResult& grid_coords() const { return _grid_coords; }

    int    size()       const { return _grid_size; }
    double pixel_size() const { return _pixel_size_nm; }

private:
    int _get_bit_ceil(int size);
    double     _pixel_size_nm;
    int        _pitch_base_size;
    int        _grid_size;
    int        _pitch;
    double     _norm_pitch;
    GridResult _grid_coords;
};

}  // namespace litho
