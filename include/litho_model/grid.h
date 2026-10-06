#pragma once

#include <Eigen/Dense>

namespace litho {

struct GridResult {
    Eigen::MatrixXd x, y;         // 像素中心相对网格几何中心的空间坐标，单位 nm；行对应 y，列对应 x。
    Eigen::VectorXd Fx_1d, Fy_1d; // 中心化 FFT 频率除以 NA/λ 后的无量纲坐标；零频索引为 N/2。
    Eigen::MatrixXd Fx_2d, Fy_2d; // 同一频率轴展开的二维坐标；行对应 fy，列对应 fx。
};

class Grid {
public:
    // 构建严格为 grid_size_pixel × grid_size_pixel 的网格，自动处理奇偶尺寸，不扩成 2^k 或额外加一。
    // pixel_size_nm/wavelength 单位为 nm，NA 无量纲；空间原点位于几何中心，偶数时落在像素之间。
    Grid(int grid_size_pixel, double pixel_size_nm, double NA, double wavelength);
    // 返回中心化的一维 FFT 物理频率轴，单位 nm⁻¹，间隔 1/(N*pixel_size_nm)；此函数不执行 FFT。
    Eigen::VectorXd fftF() const;
    // 得到空间域坐标轴以及频域的坐标轴
    GridResult      get_grid() const { return _grid_coords; }
    const GridResult& grid_coords() const { return _grid_coords; }
    // 得到网格的尺寸（像素）
    int    size()       const { return _grid_size; }
    // 得到每个像素的物理尺寸（nm）
    double pixel_size() const { return _pixel_size_nm; }

private:
    double     _pixel_size_nm;
    int        _grid_size;
    GridResult _grid_coords;
};

}  // namespace litho
