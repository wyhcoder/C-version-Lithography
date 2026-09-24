#include "CTM_Optimizer.h"
#include "loss.h"
#include "save_txt.h"

#include <LBFGS.h>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace litho {

CTM_Optimizer::CTM_Optimizer(LithographySimulator& simulator,
                             ImagingCache&         cache,
                             int                   iterations,
                             double                learning_rate,
                             std::string           optimizer_method,
                             int                   lbfgs_history_size,
                             double                gradient_tolerance)
    : _litho_simulator(simulator),
      _imaging_cache  (cache),
      _imaging        (_imaging_cache),
      _gradient       (_imaging_cache,
                       simulator._params.resist.threshold,
                       simulator._params.resist.alpha),
      _iterations     (iterations),
      _learning_rate  (learning_rate),
      _optimizer_method(std::move(optimizer_method)),
      _lbfgs_history_size(lbfgs_history_size),
      _gradient_tolerance(gradient_tolerance)
{
    if (_optimizer_method != "gradient_descent" && _optimizer_method != "lbfgs") {
        throw std::invalid_argument("CTM optimizer must be gradient_descent or lbfgs");
    }
    if (_iterations <= 0) throw std::invalid_argument("CTM max_iteration must be positive");
    if (!std::isfinite(_learning_rate) || _learning_rate <= 0.0) {
        throw std::invalid_argument("CTM learning_rate must be finite and positive");
    }
    if (_lbfgs_history_size <= 0) throw std::invalid_argument("CTM lbfgs_history_size must be positive");
    if (!std::isfinite(_gradient_tolerance) || _gradient_tolerance < 0.0) {
        throw std::invalid_argument("CTM gradient_tolerance must be finite and non-negative");
    }
    const int N     = _litho_simulator._grid.size();
    _target_image   = _litho_simulator._mask.data();
    _theta           = Eigen::MatrixXd::Zero(N, N);
    _optimizing_mask = Eigen::MatrixXd::Zero(N, N);
    _best_mask       = Eigen::MatrixXd::Zero(N, N);
    _imaging_result.aerial_image     = Eigen::MatrixXd::Zero(N, N);
    _imaging_result.wafer_image      = Eigen::MatrixXd::Zero(N, N);
    _Gm              = Eigen::MatrixXd::Zero(N, N);
    _Gt              = Eigen::MatrixXd::Zero(N, N);
}

Eigen::MatrixXd CTM_Optimizer::_initialize_theta() {
    constexpr double PI    = 3.14159265358979323846;
    constexpr double tbias = 0.3;
    return (_target_image.array() > 0.0)
        .select(
            Eigen::MatrixXd::Constant(_target_image.rows(), _target_image.cols(),
                                       tbias * PI),
            Eigen::MatrixXd::Constant(_target_image.rows(), _target_image.cols(),
                                       (1.0 - tbias) * PI));
}

Eigen::MatrixXd CTM_Optimizer::optimize(bool verbose, bool save_intermediate) {
    if (_optimizer_method == "lbfgs") return optimize_lbfgs(verbose);
    return _optimize_gradient_descent(verbose, save_intermediate);
}

Eigen::MatrixXd CTM_Optimizer::_optimize_gradient_descent(bool verbose, bool save_intermediate) {
    const double threshold = _litho_simulator._params.resist.threshold;
    const double alpha     = _litho_simulator._params.resist.alpha;

    // 支持同一个优化器重复执行公平基准。
    _error_history.clear();

    // 初始化：θ → 灰度 mask
    _theta           = _initialize_theta();
    _optimizing_mask = 0.5 * (1.0 + _theta.array().cos()).matrix();

    double min_pe = std::numeric_limits<double>::max();
    _best_mask    = _optimizing_mask;

    for (int i = 0; i < _iterations; ++i) {
        // ① compute_wafer 内部会调 compute_aerial，
        //    并把每个 kernel 的电场缓存到 Imaging::_electric_field
        _imaging_result = _imaging.compute(_optimizing_mask, threshold, alpha);

        // ② 用当前电场求梯度 Gm = ∂PE/∂mask
        _Gm = _gradient.pe_gradient(_imaging_result.wafer_image, _target_image,
                                    _imaging.get_electric_field());

        if (save_intermediate) {
            SaveTxt::save_mat(_Gm, "Gm_CTM.txt");
        }

        // std::string base = "/Users/wyh/Desktop/学校/Litho_cpp";
        //     std::string cmd = "python3 " + base + "/scripts/show_multi.py "
        //         "Gm_CTM.txt viridis "
        //         " --title_prefix 'iter " + std::to_string(i+1) + "'";
        //     std::system(cmd.c_str());

        // ③ PE 损失（无状态调用，不 new 对象）
        const double pe = Loss::pe_loss(_imaging_result.wafer_image, _target_image);
        _error_history.push_back(pe);
        if (pe < min_pe) {
            min_pe     = pe;
            _best_mask = _optimizing_mask;
        }

        // ④ θ 域梯度: Gt = -0.5 · Gm · sin(θ)
        _Gt = (-0.5 * _Gm.array() * _theta.array().sin()).matrix();
        const double gradient_norm = _Gt.norm();
        if (verbose) {
            std::cout << "Iter " << (i + 1) << "  PE=" << pe << "  ||grad||=" << gradient_norm << std::endl;
        }
        if (gradient_norm <= _gradient_tolerance) {
            if (verbose) std::cout << "Stop: gradient norm <= " << _gradient_tolerance << std::endl;
            break;
        }

        // ⑤ 更新 θ 和灰度 mask
        //   注意：不要写 `_optimizing_mask.array() = expr`，.array() 视图不允许 resize，
        //         若形状暂时不匹配就会触发 Eigen assert。直接赋给矩阵本体即可。
        _theta           -= _learning_rate * _Gt;
        _optimizing_mask  = 0.5 * (1.0 + _theta.array().cos()).matrix();
    }

    if (verbose) {
        std::cout << "Best PE loss: " << min_pe << "\n"
                  << "Optimization complete!" << std::endl;
    }
    return _best_mask;
}

double CTM_Optimizer::_evaluate_theta_objective(const Eigen::VectorXd& x, Eigen::VectorXd& gradient) {
    const int rows = _target_image.rows();
    const int cols = _target_image.cols();
    const double threshold = _litho_simulator._params.resist.threshold;
    const double alpha = _litho_simulator._params.resist.alpha;
    Eigen::Map<const Eigen::MatrixXd> theta(x.data(), rows, cols);
    _optimizing_mask = 0.5 * (1.0 + theta.array().cos()).matrix();
    _imaging_result = _imaging.compute(_optimizing_mask, threshold, alpha);
    _Gm = _gradient.pe_gradient(_imaging_result.wafer_image, _target_image, _imaging.get_electric_field());
    _Gt = (-0.5 * _Gm.array() * theta.array().sin()).matrix();
    gradient = Eigen::Map<const Eigen::VectorXd>(_Gt.data(), _Gt.size());
    return Loss::pe_loss(_imaging_result.wafer_image, _target_image);
}

Eigen::MatrixXd CTM_Optimizer::optimize_lbfgs(bool verbose) {
    _error_history.clear();
    _theta = _initialize_theta();
    _optimizing_mask = 0.5 * (1.0 + _theta.array().cos()).matrix();

    LBFGSpp::LBFGSParam<double> param;
    param.max_iterations = _iterations;
    param.epsilon = _gradient_tolerance;
    param.epsilon_rel = 0.0;
    param.m = _lbfgs_history_size;
    LBFGSpp::LBFGSSolver<double> solver(param);

    Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(_theta.data(), _theta.size());
    auto objective = [this, verbose](const Eigen::VectorXd& variables, Eigen::VectorXd& gradient) {
        const double pe = _evaluate_theta_objective(variables, gradient);
        _error_history.push_back(pe); // 包含线搜索试探点；不是 L-BFGS 迭代次数。
        if (verbose) {
            std::cout << "L-BFGS eval " << _error_history.size() << "  PE=" << pe << "  ||grad||=" << gradient.norm() << std::endl;
        }
        return pe;
    };

    double final_pe = 0.0;
    const int iterations = solver.minimize(objective, x, final_pe);
    _theta = Eigen::Map<const Eigen::MatrixXd>(x.data(), _target_image.rows(), _target_image.cols());
    _optimizing_mask = 0.5 * (1.0 + _theta.array().cos()).matrix();
    _best_mask = _optimizing_mask;

    const double final_gradient_norm = solver.final_grad_norm();
    if (verbose) {
        std::cout << "L-BFGS iterations: " << iterations << '\n'
                  << "Function evaluations: " << _error_history.size() << '\n'
                  << "Final PE: " << final_pe << '\n'
                  << "Final gradient norm: " << final_gradient_norm << '\n'
                  << "Stop reason: " << (final_gradient_norm <= _gradient_tolerance ? "gradient tolerance" : "maximum iterations")
                  << std::endl;
    }
    return _best_mask;
}

}  // namespace litho
