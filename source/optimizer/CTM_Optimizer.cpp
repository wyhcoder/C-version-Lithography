#include "CTM_Optimizer.h"
#include "loss.h"
#include <iostream>
#include <limits>
#include <save_txt.h>

namespace litho {

CTM_Optimizer::CTM_Optimizer(LithographySimulator& simulator,
                             ImagingCache&         cache,
                             int                   iterations,
                             double                learning_rate)
    : _litho_simulator(simulator),
      _imaging_cache  (cache),
      _imaging        (_imaging_cache),
      _gradient       (_imaging_cache,
                       simulator._params.resist.threshold,
                       simulator._params.resist.alpha),
      _iterations     (iterations),
      _learning_rate  (learning_rate)
{
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
        if (verbose) {
            std::cout << "Iter " << (i + 1) << "  PE=" << pe << std::endl;
        }

        if (pe < min_pe) {
            min_pe     = pe;
            _best_mask = _optimizing_mask;
        }

        // ④ θ 域梯度: Gt = -0.5 · Gm · sin(θ)
        _Gt = (-0.5 * _Gm.array() * _theta.array().sin()).matrix();

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

}  // namespace litho
