#pragma once

#include <Eigen/Dense>

namespace litho {

struct GridResult {
    Eigen::MatrixXd x, y;         // 空间域 2D
    Eigen::VectorXd Fx_1d, Fy_1d; // 频率域 1D，已经是归一化的了
    Eigen::MatrixXd Fx_2d, Fy_2d; // 频率域 2D，已经是归一化的了
};

class Grid {
public:

    // 构造函数
    Grid(int row_mask_size, double pixel_size_nm, double NA, double wavelength);
    // 做一维傅立叶变换
    Eigen::VectorXd fftF() const;
    // 得到空间域坐标轴以及频域的坐标轴
    GridResult      get_grid() const { return _grid_coords; }
    const GridResult& grid_coords() const { return _grid_coords; }
    // 得到网格的尺寸（像素）
    int    size()       const { return _grid_size; }
    // 得到每个像素的物理尺寸（nm）
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
