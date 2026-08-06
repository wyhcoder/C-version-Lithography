#pragma once

#include <Eigen/Dense>
#include <unordered_map>


namespace litho {

class ZernikeGenerator {
public:
    ZernikeGenerator(const Eigen::MatrixXd& f, const Eigen::MatrixXd& g);

    // Fringe 第 n 项（n 从 1 开始），结果带缓存
    const Eigen::MatrixXd& get_zernike(int n);

    // 波前像差：W = Σ c_i * Z_i ；coeffs 是 {Fringe索引 -> 系数} 的稀疏映射
    Eigen::MatrixXd wavefront(const std::unordered_map<int, double>& coeffs); 

    // 用 wavefront 转换为 Pupil 相位:  φ = 2π/λ * W
    Eigen::MatrixXd wavefront_pupil_phase(const std::unordered_map<int, double>& coeffs,
                                          double wavelength_nm);
    // 计算 defocus 相位
    double compute_defocus_Z4_coefficient(const double defocus_nm, const double wavelength_nm, const double NA, const double n);

private:
    Eigen::MatrixXd _f, _g, _rho2;
    std::unordered_map<int, Eigen::MatrixXd> _cache;
};

}  // namespace litho
