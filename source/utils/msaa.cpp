#include "msaa.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace litho {

namespace {

double tent_kernel(double r) {
    const double value = 1.0 - std::abs(r);
    return value > 0.0 ? value : 0.0;
}

// 对单条闭合轮廓复现论文公式 (6)-(8)。输入坐标顺序为 (y,x)。
Eigen::MatrixXd rasterize_one_dirac_indicator(
    const Polygon& polygon,
    int height,
    int width,
    double h,
    double segment_fraction)
{
    const int point_count = static_cast<int>(polygon.rows());
    if (point_count < 3) {
        return Eigen::MatrixXd::Zero(height, width);
    }
    if (polygon.cols() < 2 || !polygon.allFinite()) {
        throw std::invalid_argument(
            "rasterize_dirac_indicator: polygon must be a finite Nx2 matrix");
    }

    const double min_y = polygon.col(0).minCoeff();
    const double max_y = polygon.col(0).maxCoeff();
    const double min_x = polygon.col(1).minCoeff();
    const double max_x = polygon.col(1).maxCoeff();
    const double max_domain_y = static_cast<double>(height - 1) * h;
    const double max_domain_x = static_cast<double>(width - 1) * h;
    if (min_x <= 0.0 || min_y <= 0.0 ||
        max_x >= max_domain_x || max_y >= max_domain_y) {
        throw std::invalid_argument(
            "rasterize_dirac_indicator: contour must lie strictly inside the domain");
    }

    // 用标准 (x,y) 有向面积判定方向；首尾无需显式重复，最后一条边自动闭合。
    double twice_signed_area = 0.0;
    for (int i = 0; i < point_count; ++i) {
        const int next = (i + 1) % point_count;
        const double x0 = polygon(i, 1);
        const double y0 = polygon(i, 0);
        const double x1 = polygon(next, 1);
        const double y1 = polygon(next, 0);
        twice_signed_area += x0 * y1 - x1 * y0;
    }
    if (std::abs(twice_signed_area) < 1e-12) {
        throw std::invalid_argument(
            "rasterize_dirac_indicator: contour area is zero");
    }
    const bool counter_clockwise = twice_signed_area > 0.0;

    Eigen::MatrixXd gx = Eigen::MatrixXd::Zero(height, width);
    Eigen::MatrixXd gy = Eigen::MatrixXd::Zero(height, width);
    const double target_segment_length = segment_fraction * h;

    for (int edge = 0; edge < point_count; ++edge) {
        const int next = (edge + 1) % point_count;
        const double x0 = polygon(edge, 1);
        const double y0 = polygon(edge, 0);
        const double dx = polygon(next, 1) - x0;
        const double dy = polygon(next, 0) - y0;
        const double edge_length = std::hypot(dx, dy);
        if (edge_length <= 1e-14) continue;

        const int segment_count = std::max(
            1, static_cast<int>(std::ceil(edge_length / target_segment_length)));
        const double segment_length = edge_length / segment_count;
        const double tangent_x = dx / edge_length;
        const double tangent_y = dy / edge_length;

        // 逆时针轮廓内部在切线左侧，因此外法向为 (ty,-tx)。
        const double normal_x = counter_clockwise ? tangent_y : -tangent_y;
        const double normal_y = counter_clockwise ? -tangent_x : tangent_x;
        const double scale = -segment_length / (h * h);

        for (int segment = 0; segment < segment_count; ++segment) {
            const double fraction =
                (static_cast<double>(segment) + 0.5) / segment_count;
            const double sample_x = x0 + fraction * dx;
            const double sample_y = y0 + fraction * dy;
            const double ux = sample_x / h;
            const double uy = sample_y / h;
            const int ix0 = static_cast<int>(std::floor(ux));
            const int iy0 = static_cast<int>(std::floor(uy));

            // d(r)=1-|r| 的支撑为 [-1,1]，每个边界点最多影响四个网格点。
            for (int row : {iy0, iy0 + 1}) {
                if (row < 0 || row >= height) continue;
                const double wy = tent_kernel(uy - row);
                if (wy == 0.0) continue;
                for (int col : {ix0, ix0 + 1}) {
                    if (col < 0 || col >= width) continue;
                    const double weight = tent_kernel(ux - col) * wy;
                    gx(row, col) += scale * normal_x * weight;
                    gy(row, col) += scale * normal_y * weight;
                }
            }
        }
    }

    // x/y 累积方向的起始边界均令 chi=0。分别积分 Gx/Gy 后取平均。
    Eigen::MatrixXd chi_x = Eigen::MatrixXd::Zero(height, width);
    Eigen::MatrixXd chi_y = Eigen::MatrixXd::Zero(height, width);
    for (int row = 0; row < height; ++row) {
        double accumulated = 0.0;
        for (int col = 0; col < width; ++col) {
            accumulated += h * gx(row, col);
            chi_x(row, col) = accumulated;
        }
    }
    for (int col = 0; col < width; ++col) {
        double accumulated = 0.0;
        for (int row = 0; row < height; ++row) {
            accumulated += h * gy(row, col);
            chi_y(row, col) = accumulated;
        }
    }

    return (0.5 * (chi_x + chi_y)).cwiseMin(1.0).cwiseMax(0.0);
}

}  // namespace

// ── 采样点预设 ─────────────────────────────────────────────────────────────
Eigen::MatrixXd AntiAliasRenderer::_make_offsets_4x() {
    Eigen::MatrixXd o(4, 2);
    o << -0.25, -0.25,
         -0.25,  0.25,
          0.25,  0.25,
          0.25, -0.25;
    return o;
}

Eigen::MatrixXd AntiAliasRenderer::_make_offsets_16x() {
    Eigen::MatrixXd o(16, 2);
    // 行(dy), 列(dx)，4×4 均匀网格
    int k = 0;
    for (double dy : {-0.375, -0.125, 0.125, 0.375})
        for (double dx : {-0.375, -0.125, 0.125, 0.375})
            o.row(k++) << dy, dx;
    return o;
}

Eigen::MatrixXd AntiAliasRenderer::_make_offsets_8x8_uniform() {
    const int G = 8;
    const double base_scale = 0.4375;
    Eigen::MatrixXd o(G * G, 2);
    int k = 0;
    for (int r = 0; r < G; ++r) {
        double y = (0.5 / (G + 1) + r * (1.0 - 1.0 / (G + 1)) / (G - 1));
        for (int c = 0; c < G; ++c) {
            double x = (0.5 / (G + 1) + c * (1.0 - 1.0 / (G + 1)) / (G - 1));
            o(k, 0) = (y - 0.5) * base_scale * 2;
            o(k, 1) = (x - 0.5) * base_scale * 2;
            ++k;
        }
    }
    return o;
}

AntiAliasRenderer::AntiAliasRenderer(int msaa_level) {
    switch (msaa_level) {
        case 4:  _offsets = _make_offsets_4x();       break;
        case 16: _offsets = _make_offsets_16x();      break;
        case 64: _offsets = _make_offsets_8x8_uniform(); break;
        default:
            throw std::invalid_argument(
                "msaa_level 支持 4 / 16 / 64，当前值: " +
                std::to_string(msaa_level));
    }
}

// ── Ray-Casting 多边形内点判断 ────────────────────────────────────────────
Eigen::VectorXi AntiAliasRenderer::is_inside(
    const Polygon& polygon, const Eigen::MatrixXd& pts)
{
    int n  = static_cast<int>(polygon.rows());
    int np = static_cast<int>(pts.rows());
    Eigen::VectorXi result = Eigen::VectorXi::Zero(np);

    if (n < 3) return result;
    bool enable_inner_parallel = true;
#ifdef _OPENMP
    enable_inner_parallel = (omp_in_parallel() == 0);
#endif
    #pragma omp parallel for if(enable_inner_parallel) schedule(static)
    for (int pi = 0; pi < np; ++pi) {
        double py = pts(pi, 0);
        double px = pts(pi, 1);
        int cnt = 0;

        for (int i = 0; i < n; ++i) {
            double y1 = polygon(i, 0),       x1 = polygon(i, 1);
            double y2 = polygon((i+1)%n, 0), x2 = polygon((i+1)%n, 1);

            bool y_straddle = (y1 > py) != (y2 > py); // y 轴跨过
            if (!y_straddle) continue;

            // t ∈ [0,1]，避免除零
            double denom = y2 - y1;
            if (std::abs(denom) < 1e-12) continue;
            double t = (py - y1) / denom;
            double x_intersect = x1 + t * (x2 - x1);
            if (x_intersect > px) ++cnt;
        }
        result(pi) = cnt % 2;
    }
    return result;
}

// ── 单多边形光栅化 ────────────────────────────────────────────────────────
Eigen::MatrixXd AntiAliasRenderer::rasterize_polygon(
    const Polygon& polygon, int height, int width,
    const std::string& type) const
{
    int S = static_cast<int>(_offsets.rows());

    if (polygon.rows() < 3)
        return Eigen::MatrixXd::Zero(height, width);

    // 包围盒
    int ymin = std::max(0, (int)std::floor(polygon.col(0).minCoeff()));
    int xmin = std::max(0, (int)std::floor(polygon.col(1).minCoeff()));
    int ymax = std::min(height, (int)std::ceil(polygon.col(0).maxCoeff()) + 1);
    int xmax = std::min(width,  (int)std::ceil(polygon.col(1).maxCoeff()) + 1);

    int rows = ymax - ymin;
    int cols = xmax - xmin;
    if (rows <= 0 || cols <= 0)
        return Eigen::MatrixXd::Zero(height, width);

    // 生成所有采样点：每个像素 S 个子采样
    int np = rows * cols * S;
    Eigen::MatrixXd samples(np, 2);
    int idx = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            for (int s = 0; s < S; ++s) {
                samples(idx, 0) = (ymin + r) + _offsets(s, 0);  // y
                samples(idx, 1) = (xmin + c) + _offsets(s, 1);  // x
                ++idx;
            }
        }
    }

    // 内点判断
    Eigen::VectorXi inside = is_inside(polygon, samples);

    // 计算覆盖率
    Eigen::MatrixXd coverage_block(rows, cols);
    bool enable_coverage_parallel = true;
#ifdef _OPENMP
    enable_coverage_parallel = (omp_in_parallel() == 0);
#endif
    #pragma omp parallel for if(enable_coverage_parallel) collapse(2) schedule(static)
    for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; ++c) {
            int base = (r * cols + c) * S;
            double sum = 0;
            for (int s = 0; s < S; ++s) sum += inside(base + s);
            coverage_block(r, c) = sum / S;
        }

    Eigen::MatrixXd full = Eigen::MatrixXd::Zero(height, width);
    full.block(ymin, xmin, rows, cols) = coverage_block;

    if (type == "binary") {
        return (full.array() >= 0.5).cast<double>();
    }
    return full;
}

// ── MSAA 主入口 ────────────────────────────────────────────────────────────
Eigen::MatrixXd AntiAliasRenderer::MSAA(
    const Polygons& polygons,
    const Eigen::MatrixXd& mask_template,
    const std::string& type) const
{
    int H = static_cast<int>(mask_template.rows());
    int W = static_cast<int>(mask_template.cols());
    int P = static_cast<int>(polygons.size());

    bool already_parallel = false;
#ifdef _OPENMP
    already_parallel = (omp_in_parallel() != 0);
#endif

    // MEEF 已在控制点扰动层并行。此时直接串行聚合当前多边形，避免
    // 嵌套 parallel + critical 让不同扰动任务在全局临界区排队。
    if (already_parallel) {
        if (type == "binary") {
            Eigen::MatrixXi accum = Eigen::MatrixXi::Zero(H, W);
            for (int p = 0; p < P; ++p) {
                Eigen::MatrixXd cov = rasterize_polygon(
                    polygons[p], H, W, "gray");
                for (int r = 0; r < H; ++r)
                    for (int c = 0; c < W; ++c)
                        if (cov(r, c) >= 0.5) accum(r, c) ^= 1;
            }
            return accum.cast<double>();
        }

        Eigen::MatrixXd final_cov = Eigen::MatrixXd::Zero(H, W);
        for (int p = 0; p < P; ++p) {
            Eigen::MatrixXd cov = rasterize_polygon(
                polygons[p], H, W, "gray");
            final_cov = final_cov.array().max(cov.array()).matrix();
        }
        return final_cov.cwiseMin(1.0).cwiseMax(0.0);
    }

    if (type == "binary") {
        Eigen::MatrixXi accum = Eigen::MatrixXi::Zero(H, W);
        #pragma omp parallel
        {
            // 奇偶填充：XOR 叠加每个多边形的 bool 掩膜（线程私有副本）
            Eigen::MatrixXi accum_priv = Eigen::MatrixXi::Zero(H, W);

            #pragma omp for schedule(dynamic)
            for (int p = 0; p < P; ++p) {
                auto cov = rasterize_polygon(polygons[p], H, W, "gray");
                // 覆盖率 >= 0.5 算作内部
                for (int r = 0; r < H; ++r)
                    for (int c = 0; c < W; ++c)
                        if (cov(r, c) >= 0.5) accum_priv(r, c) ^= 1;
            }

            #pragma omp critical
            {
                for (int r = 0; r < H; ++r)
                    for (int c = 0; c < W; ++c)
                        accum(r, c) ^= accum_priv(r, c);
            }
        }
        return accum.cast<double>();

    } else {
        // gray：取每个多边形的最大覆盖率
        Eigen::MatrixXd final_cov = Eigen::MatrixXd::Zero(H, W);

        #pragma omp parallel
        {
            Eigen::MatrixXd accum_priv = Eigen::MatrixXd::Zero(H, W);

            #pragma omp for schedule(dynamic)
            for (int p = 0; p < P; ++p) {
                auto cov = rasterize_polygon(polygons[p], H, W, "gray");
                accum_priv = accum_priv.array().max(cov.array()).matrix();
            }

            #pragma omp critical
            {
                final_cov = final_cov.array().max(accum_priv.array()).matrix();
                final_cov = final_cov.cwiseMin(1.0).cwiseMax(0.0);
            }
        }
        return final_cov;
    }
}

Eigen::MatrixXd AntiAliasRenderer::rasterize_dirac_indicator(
    const Polygons& polygons,
    const Eigen::MatrixXd& mask_template,
    const std::string& type,
    double grid_spacing,
    double segment_fraction) const
{
    if (mask_template.rows() < 3 || mask_template.cols() < 3) {
        throw std::invalid_argument(
            "rasterize_dirac_indicator: mask template must be at least 3x3");
    }
    if (!std::isfinite(grid_spacing) || grid_spacing <= 0.0) {
        throw std::invalid_argument(
            "rasterize_dirac_indicator: grid_spacing must be positive");
    }
    if (!std::isfinite(segment_fraction) || segment_fraction <= 0.0) {
        throw std::invalid_argument(
            "rasterize_dirac_indicator: segment_fraction must be positive");
    }
    if (type != "gray" && type != "binary") {
        throw std::invalid_argument(
            "rasterize_dirac_indicator: type must be gray or binary");
    }

    const int height = static_cast<int>(mask_template.rows());
    const int width = static_cast<int>(mask_template.cols());

    if (type == "binary") {
        // 与现有 MSAA binary 语义一致：每条轮廓先重构，再按奇偶规则合并。
        Eigen::MatrixXi parity = Eigen::MatrixXi::Zero(height, width);
        for (const auto& polygon : polygons) {
            const Eigen::MatrixXd indicator = rasterize_one_dirac_indicator(
                polygon, height, width, grid_spacing, segment_fraction);
            for (int row = 0; row < height; ++row) {
                for (int col = 0; col < width; ++col) {
                    if (indicator(row, col) >= 0.5) parity(row, col) ^= 1;
                }
            }
        }
        return parity.cast<double>();
    }

    Eigen::MatrixXd combined = Eigen::MatrixXd::Zero(height, width);

    for (const auto& polygon : polygons) {
        Eigen::MatrixXd indicator = rasterize_one_dirac_indicator(
            polygon, height, width, grid_spacing, segment_fraction);
        combined = combined.array().max(indicator.array()).matrix();
    }
    combined = combined.cwiseMin(1.0).cwiseMax(0.0);
    return combined;
}

}  // namespace litho
