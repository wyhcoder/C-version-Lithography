#include "zernike.h"
#include <cmath>
#include <unordered_map>

namespace litho {

ZernikeGenerator::ZernikeGenerator(const Eigen::MatrixXd& f, const Eigen::MatrixXd& g)
    : _f(f), _g(g), _rho2(f.array().square() + g.array().square())
{}

const Eigen::MatrixXd& ZernikeGenerator::get_zernike(int n) {
    auto it = _cache.find(n);
    if (it != _cache.end()) return it->second;

    const double s3 = std::sqrt(3.0);
    const double s5 = std::sqrt(5.0);
    const double s6 = std::sqrt(6.0);
    const double s8 = std::sqrt(8.0);

    // 一律用 .array() 做逐元素运算
    auto f  = _f.array();
    auto g  = _g.array();
    auto r2 = _rho2.array();

    Eigen::MatrixXd Z;
    switch (n) {
    case 1:  Z = Eigen::MatrixXd::Ones(_f.rows(), _f.cols());     break; // Piston
    case 2:  Z = 2.0 * _f;                                         break; // Tilt X
    case 3:  Z = 2.0 * _g;                                         break; // Tilt Y
    case 4:  Z = s3 * (2.0 * r2 - 1.0);                            break; // Defocus
    case 5:  Z = s6 * (2.0 * f * g);                               break; // Astig 45°
    case 6:  Z = s6 * (f.square() - g.square());                   break; // Astig 0°
    case 7:  Z = s8 * (3.0 * r2 - 2.0) * g;                        break; // Coma Y
    case 8:  Z = s8 * (3.0 * r2 - 2.0) * f;                        break; // Coma X
    case 9:  Z = s8 * (3.0 * f.square() * g - g.cube());           break; // Trefoil Y
    case 10: Z = s8 * (f.cube() - 3.0 * f * g.square());           break; // Trefoil X
    case 11: Z = s5 * (6.0 * r2.square() - 6.0 * r2 + 1.0);        break; // Spherical
    default: Z = Eigen::MatrixXd::Zero(_f.rows(), _f.cols());      break;
    }

    _cache[n] = std::move(Z);
    return _cache[n];
}

// coeffs: { Fringe索引 -> 系数 }, 例如 {{4, 0.05}, {9, -0.02}}
Eigen::MatrixXd ZernikeGenerator::wavefront(const std::unordered_map<int, double>& coeffs) {
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(_f.rows(), _f.cols());
    for (const auto& [index, c] : coeffs) {     // C++17 结构化绑定
        if (c == 0.0) continue;
        W += c * get_zernike(index);
    }
    return W;
}

// Eigen::MatrixXd ZernikeGenerator::compute_defocus_Z4_coefficient(const double defocus_nm, const double wavelength_nm,const double NA, const double n){
//     Eigen::MatrixXd term = (n * n - NA * NA * _rho2.array()).max(1e-6);
//     return (defocus_nm / wavelength_nm) * (n - term.array().sqrt());
// }

double ZernikeGenerator::compute_defocus_Z4_coefficient(const double defocus_nm, const double wavelength_nm,const double NA, const double n){
    double z4_coeff = (NA * NA * defocus_nm) / (4 * n * wavelength_nm);
    return z4_coeff;
}


Eigen::MatrixXd ZernikeGenerator::wavefront_pupil_phase(
    const std::unordered_map<int, double>& coeffs, double wavelength_nm)
{
    return (2.0 * M_PI / wavelength_nm) * wavefront(coeffs);
}

}  // namespace litho
