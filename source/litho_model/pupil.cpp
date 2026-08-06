#include "pupil.h"

namespace litho {

Pupil::Pupil(const Params& params, const Eigen::MatrixXd& fx, const Eigen::MatrixXd& fy)
    : _params(params), _fx(fx), _fy(fy)        // ① _params 先初始化
{
    _init_frequency();                           // 归一化频率 + amplitude
    _wavefront(params.zernike_coeffs);           // ② 字段名用 zernike_coeffs
    // _defocus_phase(params.defocus_nm);
    compute_pupil();
}

void Pupil::_init_frequency() {
    // double f_max  = _params.NA / _params.wavelength;
    // _fx_norm      = _fx.array() / f_max;
    // _fy_norm      = _fy.array() / f_max;
    _fx_norm = _fx; // ① _fx, _fy 已经是归一化坐标
    _fy_norm = _fy;
    // ③ 用归一化坐标算半径²，圆内判断才正确
    _rho2_norm    = _fx_norm.array().square() + _fy_norm.array().square();
    _amplitude    = (_rho2_norm.array() <= 1.0).cast<double>();
}

void Pupil::_wavefront(const std::unordered_map<int, double>& coeffs) {
    // ④ _zernike 在归一化坐标就绪后才构造（optional 延迟初始化）
    _zernike.emplace(_fx_norm, _fy_norm);
    _w_static = _zernike->wavefront(coeffs);
}

// void Pupil::_defocus_phase(double defocus_nm) {
//     // φ_defocus = (2π/λ) * Δz * [n·sqrt(1 - (NA/n)²·ρ²) - n]
//     // ⑤ 用归一化 _rho2_norm，(NA/n)²·ρ² = (NA·rho_norm/n)²... 化简后：
//     // term = n² - NA²·ρ_norm²  (ρ_norm = rho in normalized coords)
//     Eigen::MatrixXd term = (_params.n * _params.n
//                           - _params.NA * _params.NA * _rho2_norm.array()).max(1e-12);
//     _defocus_phase_mat = (2.0 * M_PI * defocus_nm / _params.wavelength)
//                        * (_params.n - term.array().sqrt());
// }

void Pupil::compute_pupil() {
    _phase = _w_static ;
    Eigen::MatrixXcd exp_i_phase =
        (_phase.array() * std::complex<double>(0.0, 2.0 * M_PI)).exp();
    _pupil = _amplitude.array().cast<std::complex<double>>() * exp_i_phase.array();
}

}  // namespace litho
