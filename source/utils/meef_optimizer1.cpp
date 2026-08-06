// #include "meef_optimizer.h"
// #include <cmath>
// #include <iostream>
// #include <algorithm>

// namespace litho {

// MEEFOptimizer::MEEFOptimizer(
//     const Imaging&            imaging,
//     const ParametricDemo&     parametric,
//     const AntiAliasRenderer&  renderer,
//     const Eigen::MatrixXd&    target_mask,
//     const Eigen::MatrixXd&    sraf_mask,
//     const ControlPoints&      initial_cps,
//     const EpsResult&          eps_result,
//     const std::string&        curve_type,
//     double                    delta,
//     double                    threshold)
//     : _imaging(imaging),
//       _parametric(parametric),
//       _renderer(renderer),
//       _target(target_mask),
//       _sraf(sraf_mask),
//       _init_cps(initial_cps),
//       _eps(eps_result),
//       _curve_type(curve_type),
//       _delta(delta),
//       _threshold(threshold)
// {
//     _num_eps = static_cast<int>(eps_result.eps.rows());
//     _num_cps = 0;
//     for (auto& c : initial_cps) _num_cps += static_cast<int>(c.rows());
// }

// // ── get_cp_vectors: 每个控制点的角平分线方向 ────────────────────────────
// std::vector<CPWithDir> MEEFOptimizer::get_cp_vectors(const Eigen::MatrixXd& ctr) {
//     int n = static_cast<int>(ctr.rows());
//     std::vector<CPWithDir> res(n);

//     for (int i = 0; i < n; ++i) {
//         Eigen::Vector2d Qp = ctr.row((i-1+n)%n).transpose();
//         Eigen::Vector2d Qi = ctr.row(i).transpose();
//         Eigen::Vector2d Qn = ctr.row((i+1)%n).transpose();

//         Eigen::Vector2d v1 = Qp - Qi;
//         Eigen::Vector2d v2 = Qn - Qi;
//         v1 /= v1.norm() + 1e-8;
//         v2 /= v2.norm() + 1e-8;

//         Eigen::Vector2d bisector = v1 + v2;
//         double nb = bisector.norm();

//         if (nb < 1e-6) {
//             bisector = {-v2(1), v2(0)};
//         } else {
//             bisector /= nb;
//         }

//         // 外法线方向（逆时针轮廓的外法线）
//         Eigen::Vector2d edge = Qn - Qi;
//         Eigen::Vector2d normal = {-edge(1), edge(0)};

//         // 角平分线朝内 → 翻转
//         if (bisector.dot(normal) > 0)
//             bisector = -bisector;

//         res[i].point = Qi;
//         res[i].dir   = bisector;
//     }
//     return res;
// }

// // ── move_cp ──────────────────────────────────────────────────────────────
// Eigen::Vector2d MEEFOptimizer::move_cp(const Eigen::Vector2d& pt,
//                                         const Eigen::Vector2d& dir,
//                                         double delta) {
//     Eigen::Vector2d new_pt = pt + delta * dir;
//     // 四舍五入到 4 位小数
//     new_pt(0) = std::round(new_pt(0) * 1e4) / 1e4;
//     new_pt(1) = std::round(new_pt(1) * 1e4) / 1e4;
//     return new_pt;
// }

// // ── cp_to_mask: CPs → 曲线 → MSAA → + sraf ───────────────────────────────
// Eigen::MatrixXd MEEFOptimizer::cp_to_mask(const ControlPoints& cps) const {
//     auto curves = _parametric.get_curve_points(cps, 300);
//     Polygons polys;
//     for (auto& c : curves) polys.push_back(c);
//     auto main_mask = _renderer.MSAA(polys, _target, "gray");
//     return main_mask + _sraf;
// }

// // ── compute_epe_vector ───────────────────────────────────────────────────
// Eigen::RowVectorXd MEEFOptimizer::compute_epe_vector(
//     const Eigen::MatrixXd& mask) const
// {
//     auto aerial = _imaging.compute_aerial(mask);
//     auto [total, epe_vec] = caculate_epe(
//         aerial, _threshold, _eps.eps, _eps.weight_meef, false);
//     return epe_vec;
// }

// // ── build_meef_matrix: 中心差分（并行 mask 生成 + 串行 litho sim）───────
// Eigen::MatrixXd MEEFOptimizer::build_meef_matrix(
//     const ControlPoints& cps,
//     const std::vector<std::vector<CPWithDir>>& vecs,
//     int num_cps, int num_eps)
// {
//     // ── 阶段 1: 并行生成所有扰动后的 mask ─────────────────────────────
//     // 预分配: 2 * num_cps 个 mask（+δ 和 -δ 各一半）
//     std::vector<Eigen::MatrixXd> masks_plus(num_cps);
//     std::vector<Eigen::MatrixXd> masks_minus(num_cps);

// #pragma omp parallel for schedule(dynamic)
//     for (int col = 0; col < num_cps; ++col) {
//         // 反查 col → (contour_idx, cp_idx)
//         int ci = 0, i = col;
//         for (; ci < (int)cps.size(); ++ci) {
//             if (i < cps[ci].rows()) break;
//             i -= cps[ci].rows();
//         }
//         const auto& v = vecs[ci][i];

//         ControlPoints local_plus  = cps;
//         local_plus[ci].row(i)  = move_cp(v.point, v.dir, +_delta).transpose();
//         masks_plus[col] = cp_to_mask(local_plus);

//         ControlPoints local_minus = cps;
//         local_minus[ci].row(i) = move_cp(v.point, v.dir, -_delta).transpose();
//         masks_minus[col] = cp_to_mask(local_minus);
//     }

//     // ── 阶段 2: 串行 litho sim + 计算 EPE ─────────────────────────────
//     Eigen::MatrixXd M = Eigen::MatrixXd::Zero(num_eps, num_cps);

//     for (int col = 0; col < num_cps; ++col) {
//         auto epe_plus  = compute_epe_vector(masks_plus[col]);
//         auto epe_minus = compute_epe_vector(masks_minus[col]);

//         Eigen::RowVectorXd grad = (epe_plus - epe_minus) / (2.0 * _delta);
//         for (int j = 0; j < num_eps; ++j)
//             M(j, col) = std::round(grad(j) * 1e6) / 1e6;
//     }
//     return M;
// }

// // ── find_optimal_lambda: L-curve Tikhonov ───────────────────────────────
// double MEEFOptimizer::find_optimal_lambda(
//     const Eigen::MatrixXd& M,
//     const Eigen::RowVectorXd& e0)
// {
//     int m = static_cast<int>(M.rows());
//     int n = static_cast<int>(M.cols());
//     if (n == 0) return 1e-3;

//     Eigen::VectorXd b = -e0.transpose();   // [m x 1]
//     Eigen::MatrixXd MtM = M.transpose() * M;  // [n x n]
//     Eigen::VectorXd Mtb = M.transpose() * b;  // [n x 1]

//     // 扫描 100 个 lambda（对数均匀）
//     const int Nl = 100;
//     Eigen::VectorXd lambdas(Nl);
//     for (int i = 0; i < Nl; ++i)
//         lambdas(i) = std::pow(10.0, -6.0 + 8.0 * i / (Nl - 1));

//     Eigen::VectorXd res_norms(Nl), x_norms(Nl);

//     for (int i = 0; i < Nl; ++i) {
//         double lam = lambdas(i);
//         Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
//         Eigen::VectorXd x = (MtM + lam * I).ldlt().solve(Mtb);

//         Eigen::VectorXd residual = M * x - b;
//         res_norms(i) = residual.norm();
//         x_norms(i)    = x.norm();
//     }

//     // L-curve 曲率最大值
//     Eigen::VectorXd log_res = res_norms.array().log10();
//     Eigen::VectorXd log_x   = x_norms.array().log10();

//     // 数值一阶导 d(log_x)/d(log_res)
//     Eigen::VectorXd d1(Nl), d2(Nl);
//     for (int i = 1; i < Nl - 1; ++i) {
//         double dr = log_res(i+1) - log_res(i-1);
//         double dx = log_x(i+1)   - log_x(i-1);
//         d1(i) = std::abs(dr) > 1e-12 ? dx / dr : 0.0;
//     }
//     d1(0) = d1(1);
//     d1(Nl-1) = d1(Nl-2);

//     for (int i = 1; i < Nl - 1; ++i) {
//         double dr = log_res(i+1) - log_res(i-1);
//         d2(i) = std::abs(dr) > 1e-12 ? (d1(i+1) - d1(i-1)) / dr : 0.0;
//     }
//     d2(0) = d2(1);
//     d2(Nl-1) = d2(Nl-2);

//     // curvature = |d2| / (1 + d1²)^{3/2}
//     int best = 1;
//     double max_c = 0;
//     for (int i = 1; i < Nl - 1; ++i) {
//         double c = std::abs(d2(i)) / std::pow(1.0 + d1(i)*d1(i), 1.5);
//         if (c > max_c) { max_c = c; best = i; }
//     }
//     return lambdas(best);
// }

// // ── truncated_svd_solve ──────────────────────────────────────────────────
// Eigen::VectorXd MEEFOptimizer::truncated_svd_solve(
//     const Eigen::MatrixXd& M,
//     const Eigen::RowVectorXd& e,
//     double lambda,
//     double energy_threshold)
// {
//     int m = static_cast<int>(M.rows());
//     int n = static_cast<int>(M.cols());
//     if (n == 0) return Eigen::VectorXd::Zero(0);

//     Eigen::JacobiSVD<Eigen::MatrixXd> svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
//     const auto& S = svd.singularValues();    // [r]
//     const auto& U = svd.matrixU();           // [m x r]
//     const auto& V = svd.matrixV();           // [n x r]

//     int r = static_cast<int>(S.size());

//     // 能量截断
//     double total_energy = S.squaredNorm();
//     double cum = 0;
//     int k = r;
//     for (int i = 0; i < r; ++i) {
//         cum += S(i) * S(i);
//         if (cum / total_energy >= energy_threshold) {
//             k = i + 1;
//             break;
//         }
//     }

//     // Tikhonov: Δx = V_k * diag(s_i/(s_i²+λ)) * U_k^T * (-e)
//     Eigen::VectorXd b = -e.transpose();   // [m x 1]

//     Eigen::VectorXd result = Eigen::VectorXd::Zero(n);
//     for (int i = 0; i < k; ++i) {
//         double si = S(i);
//         double scale = si / (si * si + lambda);
//         double uiTb = U.col(i).dot(b);
//         result += scale * uiTb * V.col(i);
//     }

//     // round to 6 decimals
//     for (int i = 0; i < n; ++i)
//         result(i) = std::round(result(i) * 1e6) / 1e6;

//     return result;
// }

// // ── update_cps ────────────────────────────────────────────────────────────
// ControlPoints MEEFOptimizer::update_cps(
//     const ControlPoints& old_cps,
//     const std::vector<std::vector<CPWithDir>>& vecs,
//     const Eigen::VectorXd& delta_d)
// {
//     ControlPoints new_cps = old_cps;
//     int col = 0;
//     for (int ci = 0; ci < static_cast<int>(new_cps.size()); ++ci) {
//         for (int i = 0; i < static_cast<int>(new_cps[ci].rows()); ++i) {
//             Eigen::Vector2d new_pt = old_cps[ci].row(i).transpose()
//                 + delta_d(col) * vecs[ci][i].dir;
//             new_pt(0) = std::round(new_pt(0) * 1e4) / 1e4;
//             new_pt(1) = std::round(new_pt(1) * 1e4) / 1e4;
//             new_cps[ci].row(i) = new_pt.transpose();
//             ++col;
//         }
//     }
//     return new_cps;
// }

// // ── run: 主优化循环 ──────────────────────────────────────────────────────
// OptimRecord MEEFOptimizer::run(int iterations) {
//     OptimRecord rec;

//     // 初始误差计算
//     auto initial_mask = cp_to_mask(_init_cps);
//     auto initial_epe_vec = compute_epe_vector(initial_mask);
//     double init_epe = initial_epe_vec.sum();
//     auto initial_wafer = _imaging.compute_wafer(
//         _imaging.compute_aerial(initial_mask), _threshold, 0.05);
//     double init_pe = (initial_wafer - _target).squaredNorm();

//     auto wepe_weight = _eps.weight_epe;  // w_epe 数组
//     double init_wepe = (wepe_weight.cwiseProduct(initial_epe_vec)).sum();

//     rec.epe_history.push_back(init_epe / _num_eps);
//     rec.wepe_history.push_back(init_wepe / std::max(1.0, static_cast<double>((wepe_weight.array() == 1.0).count())));
//     rec.pe_history.push_back(init_pe);
//     rec.time_history.push_back(0.0);

//     // 当前 CP + 方向
//     ControlPoints current_cps = _init_cps;
//     std::vector<std::vector<CPWithDir>> current_vecs;
//     for (auto& c : current_cps)
//         current_vecs.push_back(get_cp_vectors(c));

//     Eigen::RowVectorXd epe_vec = initial_epe_vec;

//     for (int iter = 1; iter <= iterations; ++iter) {
//         std::cout << "===== 迭代 " << iter << " =====" << std::endl;

//         // 1) 构建 MEEF 矩阵
//         Eigen::MatrixXd M = build_meef_matrix(
//             current_cps, current_vecs, _num_cps, _num_eps);

//         // 2) 加权
//         Eigen::RowVectorXd w_meef = _eps.weight_meef;      // [1 x M]
//         Eigen::RowVectorXd w_epe  = _eps.weight_epe;       // [1 x M]
//         for (int j = 0; j < _num_cps; ++j) {
//             for (int i = 0; i < _num_eps; ++i) {
//                 M(i, j) *= w_meef(i) * w_epe(i);
//             }
//         }

//         // 截断小值
//         for (int i = 0; i < M.rows(); ++i)
//             for (int j = 0; j < M.cols(); ++j)
//                 if (std::abs(M(i, j)) < 1e-3) M(i, j) = 0.0;

//         // 3) L-curve + 截断 SVD 求解
//         double lam = find_optimal_lambda(M, epe_vec);
//         Eigen::VectorXd delta_cp = truncated_svd_solve(M, epe_vec, lam, 0.96);

//         // 4) 更新 CP
//         ControlPoints new_cps = update_cps(current_cps, current_vecs, delta_cp);

//         // 5) 新的 mask + 仿真
//         Eigen::MatrixXd new_mask = cp_to_mask(new_cps);
//         auto new_epe_vec = compute_epe_vector(new_mask);
//         double new_epe = new_epe_vec.sum();
//         auto new_wafer = _imaging.compute_wafer(
//             _imaging.compute_aerial(new_mask), _threshold, 0.05);
//         double new_pe = (new_wafer - _target).squaredNorm();
//         double new_wepe = (wepe_weight.cwiseProduct(new_epe_vec)).sum();

//         double avg_epe  = new_epe / _num_eps;
//         double n_weps   = std::max(1.0, static_cast<double>((wepe_weight.array() == 1.0).count()));
//         double avg_wepe = new_wepe / n_weps;

//         std::cout << "  EPE="  << avg_epe
//                   << "  wEPE=" << avg_wepe
//                   << "  PE="   << new_pe << std::endl;

//         rec.epe_history.push_back(avg_epe);
//         rec.wepe_history.push_back(avg_wepe);
//         rec.pe_history.push_back(new_pe);
//         rec.time_history.push_back(0.0);

//         // 记录最优
//         if (new_wepe < rec.best_wepe) {
//             rec.best_wepe = new_wepe;
//             rec.best_wepe_iter = iter;
//             rec.best_wepe_mask = new_mask;
//             rec.best_wepe_cps  = Eigen::MatrixXd(0,2);  // TODO
//         }
//         if (new_epe < rec.best_epe) {
//             rec.best_epe = new_epe;
//             rec.best_epe_iter = iter;
//             rec.best_epe_mask = new_mask;
//         }

//         // 6) 准备下一轮
//         epe_vec = new_epe_vec;
//         current_cps = new_cps;
//         for (int ci = 0; ci < static_cast<int>(current_cps.size()); ++ci)
//             current_vecs[ci] = get_cp_vectors(current_cps[ci]);
//     }

//     return rec;
// }

// }  // namespace litho
