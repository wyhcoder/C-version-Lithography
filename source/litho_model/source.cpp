#include "source.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace litho {

// ─── 构造函数 ──────────────────────────────────────────────────────────────
Source::Source(const SourceParams& params)
    : _params(params)
{
    compute_source_map();
}

// ─── 静态工具：cell-center → edges + widths ───────────────────────────────
AxisEdges Source::axis_edges_and_widths(const Eigen::VectorXd& axis) {
    if (axis.size() < 2)
        throw std::invalid_argument("axis 长度必须至少为 2");

    Eigen::VectorXd diffs = axis.tail(axis.size()-1) - axis.head(axis.size()-1);
    if ((diffs.array() <= 0.0).any())
        throw std::invalid_argument("axis 必须严格递增");

    Eigen::VectorXd inner = 0.5 * (axis.head(axis.size()-1) + axis.tail(axis.size()-1));

    Eigen::VectorXd edges(axis.size() + 1);
    edges(0)                       = axis(0)             - 0.5 * diffs(0);
    edges.segment(1, inner.size()) = inner;
    edges(edges.size()-1)          = axis(axis.size()-1) + 0.5 * diffs(diffs.size()-1);

    return {edges, edges.tail(edges.size()-1) - edges.head(edges.size()-1)};
}

// ─── 静态工具：每个 cell 细分为 upsample 份 ──────────────────────────────
Eigen::VectorXd Source::subdivide_axis_from_edges(const Eigen::VectorXd& edges,
                                                   int upsample) {
    int n = static_cast<int>(edges.size()) - 1;
    Eigen::VectorXd left  = edges.head(n);
    Eigen::VectorXd width = edges.tail(n) - left;

    Eigen::VectorXd result(n * upsample);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < upsample; ++k)
            result(i * upsample + k) = left(i) + width(i) * ((k + 0.5) / upsample);
    return result;
}

// ─── 静态工具：avg_pool2d（对应 F.avg_pool2d stride=kernel） ─────────────
Eigen::MatrixXd Source::_avg_pool2d(const Eigen::MatrixXd& src, int kernel) {
    int rows_out = src.rows() / kernel;
    int cols_out = src.cols() / kernel;
    Eigen::MatrixXd out(rows_out, cols_out);
    for (int i = 0; i < rows_out; ++i)
        for (int j = 0; j < cols_out; ++j)
            out(i, j) = src.block(i*kernel, j*kernel, kernel, kernel).mean();
    return out;
}

// ─── 计算 extent（光源的频率覆盖范围）────────────────────────────────────
std::pair<double, double> Source::_compute_extent(double step_x, double step_y) const {
    double mx = std::max({2.0 * step_x, 3.0 * _params.smoothing, 0.05});
    double my = std::max({2.0 * step_y, 3.0 * _params.smoothing, 0.05});
    return {_params.sigma_out + mx, _params.sigma_out + my};
}

// ─── 截取紧凑光源轴（归一化到 NA/λ=1）────────────────────────────────────
std::pair<Eigen::VectorXd, Eigen::VectorXd>
Source::_extract_compact_axes(const Eigen::VectorXd& fx,
                               const Eigen::VectorXd& fy) const {
    double f_max = _params.NA / _params.wavelength_nm;
    Eigen::VectorXd fx_n = fx.array() / f_max;
    Eigen::VectorXd fy_n = fy.array() / f_max;

    double step_x = std::abs(fx_n(1) - fx_n(0));
    double step_y = std::abs(fy_n(1) - fy_n(0));
    auto [ext_x, ext_y] = _compute_extent(step_x, step_y);

    // 只保留 |f| <= extent 的轴点（严格单调，所以可以直接筛选后复制）
    auto mask_x = (fx_n.array().abs() <= ext_x);
    auto mask_y = (fy_n.array().abs() <= ext_y);

    int nx = mask_x.count();
    int ny = mask_y.count();
    Eigen::VectorXd sfx(nx), sfy(ny);
    for (int i = 0, k = 0; i < fx_n.size(); ++i) if (mask_x(i)) sfx(k++) = fx_n(i);
    for (int i = 0, k = 0; i < fy_n.size(); ++i) if (mask_y(i)) sfy(k++) = fy_n(i);

    return {sfx, sfy};
}

// ─── 光源图案（环形照明示例；可扩展为偶极/四极等）────────────────────────
Eigen::MatrixXd Source::_generate_source_pattern(const Eigen::MatrixXd& fxx,
                                                  const Eigen::MatrixXd& fyy) const {
    Eigen::MatrixXd rho = (fxx.array().square() + fyy.array().square()).sqrt().matrix();

    // 环形：sigma_in < rho < sigma_out，边缘用 smooth_step 平滑
    Eigen::MatrixXd outer = _smooth_step(_params.smoothing,
                                         Eigen::MatrixXd((_params.sigma_out - rho.array()).matrix()));
    Eigen::MatrixXd inner = _smooth_step(_params.smoothing,
                                         Eigen::MatrixXd((rho.array() - _params.sigma_in).matrix()));
    return (outer.array() * inner.array()).matrix();
}

// ─── 平滑阶跃函数 ─────────────────────────────────────────────────────────
Eigen::MatrixXd Source::_smooth_step(double width, const Eigen::MatrixXd& x) const {
    if (width == 0.0)
        return (x.array() > 0.0).cast<double>();
    return 0.5 * (1.0 + x.unaryExpr([width](double v) {
        return std::erf(v / width);
    }).array());
}

// ─── 主函数：compute_source_map ───────────────────────────────────────────
const SourceMap& Source::compute_source_map(){
    // 1. 筛选频率坐标
    std::vector<double> ff_source_vec;
    for (int i = 0; i < _params.frequence_coords.size(); i++){
        if ( _params.frequence_coords(i) >= -2.0 && _params.frequence_coords(i) <= 2.0){
            ff_source_vec.push_back(_params.frequence_coords(i));
        }
    }

    int num = (int)ff_source_vec.size();
    Eigen::VectorXd ff_source(num);
    for (int i = 0; i < num; i++){
        ff_source(i) = ff_source_vec[i];
    }
    // 2. 高分辨率轴（upsample > 1 时细分）
    int high_num = num * _params.upsample;
    double start = ff_source(0);
    double end = ff_source(num - 1);
    Eigen::VectorXd temp_ffsource =
        Eigen::VectorXd::LinSpaced(high_num, start, end);
    Eigen::MatrixXd ffx(high_num, high_num);
    Eigen::MatrixXd ffy(high_num, high_num);
    for (int r = 0; r < high_num; r++){
        for (int c = 0; c < high_num; c++){
            ffx(r, c) = temp_ffsource(c);
            ffy(r, c) = temp_ffsource(r);
        }
    }
    // 3. 生成高分辨率光源图案
    Eigen::MatrixXd source_hi = _generate_source_pattern(ffx, ffy);
    Eigen::MatrixXd source_map = (_params.upsample > 1)
        ? _avg_pool2d(source_hi, _params.upsample)
        : source_hi;
    // 4. clamp 并检查非零
    source_map = source_map.array().max(0.0).matrix();
    if (source_map.size() == 0 || source_map.maxCoeff() <= 0.0)
        throw std::runtime_error(
            "生成的 source_map 全为 0，请检查 sigma/smoothing 与 grid 范围");
    double total = source_map.sum();
    if (!std::isfinite(total) || total <= 0.0)
        throw std::runtime_error("source_weight_map 总权重无效");
    // source_map /= total;
    _source_map = {ff_source, source_map};
    return _source_map;
}



// const SourceMap& Source::compute_source_map(const Eigen::VectorXd& fx,
//                                              const Eigen::VectorXd& fy) {
//     // 1. 截取紧凑轴（归一化坐标）
//     auto [sfx, sfy] = _extract_compact_axes(fx, fy);

//     // 2. 计算 edges + widths
//     auto [edges_x, widths_x] = axis_edges_and_widths(sfx);
//     auto [edges_y, widths_y] = axis_edges_and_widths(sfy);

//     // 3. 高分辨率轴（upsample > 1 时细分）
//     Eigen::VectorXd hi_fx = (_params.upsample > 1)
//         ? subdivide_axis_from_edges(edges_x, _params.upsample) : sfx;
//     Eigen::VectorXd hi_fy = (_params.upsample > 1)
//         ? subdivide_axis_from_edges(edges_y, _params.upsample) : sfy;

//     // 4. meshgrid（indexing="ij"：行→fy，列→fx）
//     int ny = hi_fy.size(), nx = hi_fx.size();
//     Eigen::MatrixXd fxx(ny, nx), fyy(ny, nx);
//     for (int r = 0; r < ny; ++r)
//         for (int c = 0; c < nx; ++c) {
//             fxx(r, c) = hi_fx(c);
//             fyy(r, c) = hi_fy(r);
//         }

//     // 5. 生成高分辨率光源图案
//     Eigen::MatrixXd source_hi = _generate_source_pattern(fxx, fyy);

//     // 6. avg_pool2d 下采样（upsample > 1）
//     Eigen::MatrixXd source_map = (_params.upsample > 1)
//         ? _avg_pool2d(source_hi, _params.upsample)
//         : source_hi;

//     // 7. clamp 并检查非零
//     source_map = source_map.array().max(0.0).matrix();
//     if (source_map.size() == 0 || source_map.maxCoeff() <= 0.0)
//         throw std::runtime_error(
//             "生成的 source_map 全为 0，请检查 sigma/smoothing 与 grid 范围");

//     // 8. 面积加权归一化（总和=1）
//     // widths_y[:, None] * widths_x[None, :]
//     Eigen::MatrixXd cell_area = widths_y * widths_x.transpose();   // [ny x nx]
//     Eigen::MatrixXd weight_map = source_map.array() * cell_area.array();
//     double total = weight_map.sum();
//     if (!std::isfinite(total) || total <= 0.0)
//         throw std::runtime_error("source_weight_map 总权重无效");
//     weight_map /= total;

//     // 9. 存储结果
//     _source_map = {sfx, sfy, edges_x, edges_y, widths_x, widths_y,
//                    source_map, weight_map};

//     return _source_map;
// }

}  // namespace litho
