#pragma once
#include <Eigen/Dense>
#include <utility>

namespace litho {

// struct SourceParams {
//     double wavelength_nm;
//     double NA;
//     double n;
//     double sigma_in;
//     double sigma_out;
//     int    upsample            = 10;
//     double smoothing           = 0.0;
//     double lobe_half_width_deg = 0.0;
// };

struct SourceParams {
    double wavelength_nm;
    double NA;
    double n;
    double sigma_in;
    double sigma_out;
    Eigen::MatrixXd frequence_coords;
    int    upsample            = 10;
    double smoothing           = 0.0;
    double lobe_half_width_deg = 0.0;
};






struct AxisEdges {
    Eigen::VectorXd edges;
    Eigen::VectorXd widths;
};

// compute_source_map 的完整结果
// struct SourceMap {
//     Eigen::VectorXd fx_norm;         // 光源归一化 fx 轴
//     Eigen::VectorXd fy_norm;         // 光源归一化 fy 轴
//     Eigen::VectorXd edges_x;
//     Eigen::VectorXd edges_y;
//     Eigen::VectorXd widths_x;
//     Eigen::VectorXd widths_y;
//     Eigen::MatrixXd source_map;       // 原始强度（未归一化）
//     Eigen::MatrixXd source_weight_map; // 面积加权并归一化（总和=1）
// };
struct SourceMap {
    Eigen::VectorXd source_coords;
    Eigen::MatrixXd source_weight_map;
};


class Source {
public:
    explicit Source(const SourceParams& params);

    // 主入口：生成光源权重图，结果缓存在成员变量
    // const SourceMap& compute_source_map(const Eigen::VectorXd& fx,
    //                                     const Eigen::VectorXd& fy);
    const SourceMap& compute_source_map();

    const SourceMap& source_map() const { return _source_map; }
    Eigen::MatrixXd get_source_map() { return _source_map.source_weight_map; }

    // ── 静态工具 ──────────────────────────────────────────
    static AxisEdges        axis_edges_and_widths(const Eigen::VectorXd& axis);
    static Eigen::VectorXd  subdivide_axis_from_edges(const Eigen::VectorXd& edges,
                                                      int upsample);

private:
    // 将全局频率轴归一化并截取到光源范围内
    std::pair<Eigen::VectorXd, Eigen::VectorXd>
        _extract_compact_axes(const Eigen::VectorXd& fx,
                              const Eigen::VectorXd& fy) const;

    // 计算光源范围（sigma_out + margin）
    std::pair<double, double> _compute_extent(double step_x, double step_y) const;

    // 由归一化 2D 网格生成光源强度分布（环形 / 偶极 / 四极等）
    Eigen::MatrixXd _generate_source_pattern(const Eigen::MatrixXd& fxx,
                                             const Eigen::MatrixXd& fyy) const;

    // 平滑阶跃函数（erf 平滑）
    Eigen::MatrixXd _smooth_step(double width, const Eigen::MatrixXd& x) const;

    // avg_pool2d 等价：对 (U×U) 块取均值做下采样
    static Eigen::MatrixXd _avg_pool2d(const Eigen::MatrixXd& src, int kernel);

    SourceParams _params;
    SourceMap    _source_map;
};

}  // namespace litho
