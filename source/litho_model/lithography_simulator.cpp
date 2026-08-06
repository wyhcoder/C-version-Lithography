#include "lithography_simulator.h"

namespace litho {

// ── 静态辅助：从 SimulationParameters 造 Pupil::Params ────────────────
Params LithographySimulator::_make_pupil_params(const SimulationParameters& p) {
    Params pp;
    pp.NA            = p.optics.na;
    pp.wavelength    = p.system.wavelength_nm;
    pp.n             = p.optics.refractive_index;
    pp.defocus_nm    = 0.0;
    pp.zernike_coeffs = p.optics.aberrations;
    // TODO: 若需要 Zernike 像差，把 p.optics.aberrations ("z4"->double) 转成
    //       unordered_map<int,double> 塞进 pp.zernike_coeffs
    return pp;
}

// ── 静态辅助：从 SimulationParameters 造 SourceParams ─────────────────
SourceParams LithographySimulator::_make_source_params(const SimulationParameters& p, const Eigen::VectorXd& fx) {
    SourceParams sp;
    sp.wavelength_nm = p.system.wavelength_nm;
    sp.NA            = p.optics.na;
    sp.n             = p.optics.refractive_index;
    sp.sigma_in      = p.source.sigma_in;
    sp.sigma_out     = p.source.sigma_out;
    sp.frequence_coords = fx;
    sp.upsample      = 10;
    sp.smoothing     = 0;
    return sp;
}

// ── 拼接掩模完整路径 ──────────────────────────────────────────────────
// 约定：yaml 里 image_name 只写"图案名"（不含目录、不含扩展名），
//       统一放在 target_pattern/ 下，扩展名 .bmp。
// 例如: image_name = "matrix_0525" → target_pattern/matrix_0525.bmp
static std::string _mask_path(const std::string& image_name) {
    return "target_pattern/" + image_name + ".bmp";
}

// ── 构造函数：全部走成员初始化列表 ────────────────────────────────────
LithographySimulator::LithographySimulator(const SimulationParameters& params)
    : _params(params),
      _grid  (params.mask.row_grid_size, params.system.pixel_size_nm,params.optics.na, params.system.wavelength_nm),
      _pupil (_make_pupil_params(params),
              _grid.grid_coords().Fx_2d,
              _grid.grid_coords().Fy_2d),
      _source(_make_source_params(params, _grid.grid_coords().Fx_1d)),
      _mask  (_mask_path(params.mask.image_name),
              params.system.grid_size_pixel,
              /*normalize=*/true),
      _ep_select(_mask.data(),4,1)
{
    // Source 需要在拿到 Grid 的频率轴后再计算权重图
    // _source.compute_source_map(_grid.grid_coords().Fx_1d,
    //                            _grid.grid_coords().Fy_1d);
}

}  // namespace litho
