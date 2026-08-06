#pragma once

#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>
#include <string>
#include "imaging.h"         // Imaging, ImagingCache
#include "ep_select.h"       // ControlPoints, EpsResult
#include "msaa.h"            // Polygons
#include "parametric.h"      // ParametricDemo
#include "loss.h"            // Loss::evaluate_epe, Loss::weighted_epe

namespace litho {

// ── 控制点 + 优化方向 ──────────────────────────────────────────────────
struct CPWithDir {
    Eigen::Vector2d point;    // (y, x)
    Eigen::Vector2d dir;      // 单位角平分线方向 (向内)
};

// ── 优化记录 ────────────────────────────────────────────────────────────
struct OptimRecord {
    std::vector<double> epe_history;
    std::vector<double> wepe_history;
    std::vector<double> pe_history;
    std::vector<double> time_history;

    // 最优结果
    double best_wepe = 1e12;
    int    best_wepe_iter = 0;
    double best_epe  = 1e12;
    int    best_epe_iter = 0;

    Eigen::MatrixXd best_wepe_mask;
    Eigen::MatrixXd best_epe_mask;
    Eigen::MatrixXd best_wepe_cps;
    Eigen::MatrixXd best_epe_cps;
};

class MEEFOptimizer {
public:
    // imaging:      已预计算好的成像对象
    // parametric:   曲线拟合器 ("BS"/"OA"/"BZ")
    // renderer:     MSAA 渲染器
    // target_mask:  目标二值掩模 [N x N]
    // sraf_mask:    SRAF 常量 mask [N x N]
    // initial_cps:  初始控制点
    // eps_result:   EP 点 + 权重
    // curve_type:   "OA"/"BS"/"BZ"
    // delta:        MEEF 差分扰动量
    // threshold:    光刻胶阈值
    MEEFOptimizer(const Imaging&            imaging,
                  const ParametricDemo&     parametric,
                  const AntiAliasRenderer&  renderer,
                  const Eigen::MatrixXd&    target_mask,
                  const Eigen::MatrixXd&    sraf_mask,
                  const ControlPoints&      initial_cps,
                  const EpsResult&           eps_result,
                  const std::string&         curve_type,
                  double                     delta,
                  double                     threshold);

    // 运行优化
    OptimRecord run(int iterations);

private:
    // ── CP 方向向量 ─────────────────────────────────────────────────
    static std::vector<CPWithDir> get_cp_vectors(const Eigen::MatrixXd& contour);

    // ── 扰动单个 CP ─────────────────────────────────────────────────
    static Eigen::Vector2d move_cp(const Eigen::Vector2d& pt,
                                    const Eigen::Vector2d& dir,
                                    double delta);

    // ── 从 CP 生成 mask (CP → 曲线 → MSAA → +sraf) ──────────────────
    Eigen::MatrixXd cp_to_mask(const ControlPoints& cps) const;

    // ── mask → aerial → EPE vector ──────────────────────────────────
    Eigen::RowVectorXd compute_epe_vector(const Eigen::MatrixXd& mask) const;

    // ── 构建 MEEF 矩阵 ───────────────────────────────────────────────
    Eigen::MatrixXd build_meef_matrix(
        const ControlPoints&              cps,
        const std::vector<std::vector<CPWithDir>>& vecs,
        int num_cps, int num_eps);

    // ── L-curve 最优 lambda ──────────────────────────────────────────
    static double find_optimal_lambda(const Eigen::MatrixXd& M,
                                       const Eigen::RowVectorXd& e0);

    // ── 截断 SVD 求解 ────────────────────────────────────────────────
    static Eigen::VectorXd truncated_svd_solve(
        const Eigen::MatrixXd& M,
        const Eigen::RowVectorXd& e,
        double lambda,
        double energy_threshold = 0.96);

    // ── 更新控制点 ───────────────────────────────────────────────────
    static ControlPoints update_cps(
        const ControlPoints& old_cps,
        const std::vector<std::vector<CPWithDir>>& vecs,
        const Eigen::VectorXd& delta_d);

    // ── 成员 ─────────────────────────────────────────────────────────
    const Imaging&           _imaging;
    const ParametricDemo&    _parametric;
    const AntiAliasRenderer& _renderer;

    Eigen::MatrixXd _target;      // [N x N]
    Eigen::MatrixXd _sraf;        // [N x N]
    ControlPoints   _init_cps;
    EpsResult       _eps;

    std::string _curve_type;
    double      _delta;
    double      _threshold;

    int _num_eps;
    int _num_cps;
};

}  // namespace litho
