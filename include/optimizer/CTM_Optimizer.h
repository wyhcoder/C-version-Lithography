#pragma once

#include <Eigen/Dense>
#include <string>
#include <vector>

#include "lithography_simulator.h"

#include "litho_prepare.h"
#include "imaging.h"

#include "gradient.h"

namespace litho {

class CTM_Optimizer {
public:
    CTM_Optimizer(LithographySimulator& simulator,
                  ImagingCache&         cache,
                  int                   iterations,
                  double                learning_rate,
                  std::string           optimizer_method = "gradient_descent",
                  int                   lbfgs_history_size = 10,
                  double                gradient_tolerance = 0.0);
    ~CTM_Optimizer() = default;

    // verbose=false、save_intermediate=false 用于性能基准，避免终端和磁盘 I/O 干扰计时。
    Eigen::MatrixXd optimize(bool verbose = true, bool save_intermediate = true);
    Eigen::MatrixXd optimize_lbfgs(bool verbose = true);
    const std::vector<double>& error_history() const { return _error_history; }

private:
    double _evaluate_theta_objective(const Eigen::VectorXd& x, Eigen::VectorXd& gradient);
    Eigen::MatrixXd _optimize_gradient_descent(bool verbose, bool save_intermediate);
    Eigen::MatrixXd _initialize_theta();

    // ── 依赖 ──────────────────────────────────────────────────
    LithographySimulator _litho_simulator;   // 引用外部，不拷贝
    ImagingCache        _imaging_cache;     // 引用外部，不拷贝
    Imaging               _imaging;           // 一次性建好（含 FFTW plan）
    Gradient              _gradient;          // 一次性建好（含 FFTW plan）

    int    _iterations;
    double _learning_rate;
    std::string _optimizer_method;
    int _lbfgs_history_size;
    double _gradient_tolerance;

    // ── 状态矩阵（形状固定，复用内存） ─────────────────────────
    Eigen::MatrixXd _target_image;
    Eigen::MatrixXd _theta;
    Eigen::MatrixXd _optimizing_mask;
    Eigen::MatrixXd _best_mask;

    // ── 循环缓冲（避免每次迭代重新分配） ───────────────────────
    Imaging_Result _imaging_result;
    Eigen::MatrixXd _Gm;
    Eigen::MatrixXd _Gt;

    std::vector<double> _error_history;
};

}  // namespace litho
