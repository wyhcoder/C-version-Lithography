#pragma once

#include <Eigen/Dense>
#include <unordered_map>
#include <optional>
#include "zernike.h"

namespace litho {

struct Params {
    double NA         = 1.35;
    double wavelength = 193.0;    // [nm]
    double n          = 1.44;     // immersion 折射率
    std::unordered_map<int, double> zernike_coeffs;
    double defocus_nm = 0.0;
};

class Pupil {
public:
    // 构造函数，直接计算出光瞳函数
    Pupil(const Params& params, const Eigen::MatrixXd& fx, const Eigen::MatrixXd& fy);
    // 计算光瞳函数
    void compute_pupil();

    const Eigen::MatrixXd&  get_amplitude() const { return _amplitude; }
    const Eigen::MatrixXd&  get_phase()     const { return _phase; }
    const Eigen::MatrixXcd& get_pupil()     const { return _pupil; }
    const Eigen::MatrixXd&  fx()            const { return _fx; }
    const Eigen::MatrixXd&  fy()            const { return _fy; }
    double                  NA()            const { return _params.NA; }
    double                  wavelength()    const { return _params.wavelength; }

private:
    void _init_frequency();
    void _wavefront(const std::unordered_map<int, double>& coeffs);
    void _defocus_phase(double defocus_nm);

    Params           _params;
    Eigen::MatrixXd  _fx, _fy;           // 原始频率 [1/nm],已经归一化了
    Eigen::MatrixXd  _fx_norm, _fy_norm;  // 归一化频率 f,g ∈ [-1,1]
    Eigen::MatrixXd  _rho2_norm;          // 归一化半径²，判断光瞳边界用
    Eigen::MatrixXd  _amplitude;
    Eigen::MatrixXd  _w_static;
    Eigen::MatrixXd  _defocus_phase_mat;
    Eigen::MatrixXd  _phase;
    Eigen::MatrixXcd _pupil;

    // ZernikeGenerator 在 _init_frequency 后才能构造，用 optional 延迟初始化
    std::optional<ZernikeGenerator> _zernike;
};

}  // namespace litho
