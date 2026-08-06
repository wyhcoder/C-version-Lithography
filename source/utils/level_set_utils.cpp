#include "level_set_utils.h"
#include <iostream>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
// OpenCV 5 moved DistanceTypes (and DIST_L2) into this header.  OpenCV 4
// exposes it through imgproc.hpp, so only include the new header when needed.
#if CV_VERSION_MAJOR >= 5
#include <opencv2/geometry/2d.hpp>
#endif

namespace litho {

    Eigen::MatrixXd LevelSetUtils::pad_edge(const Eigen::MatrixXd &data, int pad, int axis){
        int R = data.rows();
        int C = data.cols();
        if (axis == 0){
            Eigen::MatrixXd padded(R + 2 * pad, C);
            for (int i = 0; i < pad; ++i){                    // ← pad 次，不是 R 次
                padded.row(i) = data.row(0);
            }
            padded.middleRows(pad, R) = data;
            for (int i = 0; i < pad; ++i){                    // ← pad 次
                padded.row(R + pad + i) = data.row(R - 1);
            }
            return padded;
        }else {
            Eigen::MatrixXd padded(R, C + 2 * pad);
            for (int j = 0; j < pad; ++j){                    // ← pad 次，不是 C 次
                padded.col(j) = data.col(0);
            }
            padded.middleCols(pad, C) = data;
            for (int j = 0; j < pad; ++j){                    // ← pad 次
                padded.col(C + pad + j) = data.col(C - 1);
            }
            return padded;
        }
    };

    Eigen::VectorXd LevelSetUtils::weno5_1d(const Eigen::VectorXd &v, int N_orig, double dx, std::string dir){
        const int N_pad = (int)v.size();

        // 一阶差分：从整个 padded 向量计算
        Eigen::VectorXd D1(N_pad - 1);
        for (int i = 0; i < N_pad - 1; ++i){
            D1(i) = (v(i + 1) - v(i)) / dx;
        }

        Eigen::VectorXd out(N_orig);

        // WENO5-JS 系数
        const double c13 = 1.0 / 3.0,  c76 = 7.0 / 6.0,  c116 = 11.0 / 6.0;
        const double c16 = 1.0 / 6.0,  c56 = 5.0 / 6.0;
        const double cS1 = 13.0 / 12.0, cS2 = 0.25;
        double g1, g2, g3;
        if (dir == "minus") { g1 = 0.1; g2 = 0.6; g3 = 0.3; }   // minus
        else            { g1 = 0.3; g2 = 0.6; g3 = 0.1; }   // plus

        const double eps0 = 1e-6;
        const double eps_min = 1e-99;

        // 输出原始点数 N_orig
        // Python: v_slices[i] = D1[i : i+N_orig]，即第 i 个输出点用 D1(i)..D1(i+4)
        for (int i = 0; i < N_orig; ++i){
            double v1, v2, v3, v4, v5;
            if (dir == "plus"){
                v1 = D1(i + 4); v2 = D1(i + 3); v3 = D1(i + 2); v4 = D1(i + 1); v5 = D1(i);
            }else{
                v1 = D1(i); v2 = D1(i + 1); v3 = D1(i + 2); v4 = D1(i + 3); v5 = D1(i + 4);
            }
            // 三个子模版的线性组合
            double d1 = c13 * v1 - c76 * v2 + c116 * v3;
            double d2 = -c16 * v2 + c56 * v3 + c13 * v4;
            double d3 = c13 * v3 + c56 * v4 - c16 * v5;

            // 光滑性指示器
            double t1 = v1 - 2 * v2 + v3;
            double t2 = v2 - 2 * v3 + v4;
            double t3 = v3 - 2 * v4 + v5;

            double S1 = cS1 * t1 * t1 + cS2 * (v1 - 4 * v2 + 3 * v3) * (v1 - 4 * v2 + 3 * v3);
            double S2 = cS1 * t2 * t2 + cS2 * (v2 - v4) * (v2 - v4);
            double S3 = cS1 * t3 * t3 + cS2 * (3 * v3 - 4 * v4 + v5) * (3 * v3 - 4 * v4 + v5);
            // epsilon 自适应
            double vmax2 = std::max({v1 * v1, v2 * v2, v3 * v3, v4 * v4, v5 * v5});
            double eps = eps0 * vmax2 + eps_min;

            // 非线性权重
            double a1 = g1 / ((S1 + eps) * (S1 + eps));
            double a2 = g2 / ((S2 + eps) * (S2 + eps));
            double a3 = g3 / ((S3 + eps) * (S3 + eps));
            double w_sum = a1 + a2 + a3;

            out(i) = (a1 * d1 + a2 * d2 + a3 * d3) / w_sum;
        }
        return out;
    }

    Eigen::MatrixXd LevelSetUtils::der_weno5(const Eigen::MatrixXd &M, double dx, std::string direction, int axis){
        if (direction != "plus" && direction != "minus"){
            throw std::invalid_argument("direction must be plus or minus");
        }
        bool is_2d = (M.rows() > 1 && M.cols() > 1);
        int ax = (axis == -1)? (is_2d? 1 : 0) : axis;

        std::string dir = (direction == "plus")? "plus" : "minus";

        Eigen::MatrixXd padded = pad_edge(M, 3, ax);

        int R = M.rows(), C = M.cols();
        Eigen::MatrixXd result(R, C);
        if (ax == 0) {
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < C; ++j)
                result.col(j) = weno5_1d(padded.col(j), R, dx, dir);
        } else {
            #pragma omp parallel for schedule(static)                       
            for (int i = 0; i < R; ++i)
                result.row(i) = weno5_1d(padded.row(i).transpose(), C, dx, dir);
        }
        return result;





    }

    Eigen::MatrixXd LevelSetUtils::_select_upwind_deriv(const Eigen::MatrixXd &Vn, const Eigen::MatrixXd &der_minus, const Eigen::MatrixXd &der_plus){
        Eigen::MatrixXd result = Vn;
        for (int i = 0; i < Vn.rows(); ++i){
            for (int j = 0; j < Vn.cols(); ++j){
                if (Vn(i, j) == 0){
                    result(i, j) = 0;
                    continue;
                }
                else if (Vn(i, j) > 0){              // Python: Vn > 0 → der_minus
                    result(i, j) = der_minus(i, j);
                }else{                                 // Vn < 0 → der_plus
                    result(i, j) = der_plus(i, j);
                }
            }
        }
        return result;
    }

    Evolve_Params LevelSetUtils::evolve_normal_WENO_godunov(const Eigen::MatrixXd &phi, const Eigen::MatrixXd &Vn, double dy, double dx){

        // 与 Python 对齐: 外层 pad 3（两个轴），der_weno5 内部再 pad 3
        Eigen::MatrixXd data_ext = pad_edge(pad_edge(phi, 3, 0), 3, 1);  // (N+6, N+6)
        Eigen::MatrixXd Vn_ext   = pad_edge(pad_edge(Vn,  3, 0), 3, 1);  // (N+6, N+6)

        // der_weno5 内部 pad 3 → 输出 (N+6, N+6)
        Eigen::MatrixXd phi_x_minus = der_weno5(data_ext, dx, "minus", 1);
        Eigen::MatrixXd phi_x_plus  = der_weno5(data_ext, dx, "plus",  1);

        Eigen::MatrixXd phi_T = data_ext.transpose();
        Eigen::MatrixXd phi_y_minus_T = der_weno5(phi_T, dy, "minus", 1);
        Eigen::MatrixXd phi_y_plus_T  = der_weno5(phi_T, dy, "plus",  1);
        Eigen::MatrixXd phi_y_minus = phi_y_minus_T.transpose();
        Eigen::MatrixXd phi_y_plus  = phi_y_plus_T.transpose();

        Eigen::MatrixXd phi_x = _select_upwind_deriv(Vn_ext, phi_x_minus, phi_x_plus);
        Eigen::MatrixXd phi_y = _select_upwind_deriv(Vn_ext, phi_y_minus, phi_y_plus);

        Eigen::MatrixXd grad_mag = (phi_x.array().square() + phi_y.array().square()).sqrt();
        Eigen::MatrixXd delta_ext = Vn_ext.array() * grad_mag.array();

        Eigen::MatrixXd H1_abs_ext = (Vn_ext.array() * phi_x.array()).abs();
        Eigen::MatrixXd H2_abs_ext = (Vn_ext.array() * phi_y.array()).abs();

        int N = phi.rows();
        Eigen::MatrixXd delta  = delta_ext.block(3, 3, N, N);
        Eigen::MatrixXd H1_abs = H1_abs_ext.block(3, 3, N, N);
        Eigen::MatrixXd H2_abs = H2_abs_ext.block(3, 3, N, N);

        return Evolve_Params(delta, H1_abs, H2_abs);
    }


    Eigen::MatrixXd LevelSetUtils::_gradient_1d(const Eigen::MatrixXd &M, double d, int axis){
        int R = M.rows(), C = M.cols();
        Eigen::MatrixXd result(R, C);
        bool enable_inner_parallel = true;
#ifdef _OPENMP
        enable_inner_parallel = (omp_in_parallel() == 0);
#endif
        if ( axis == 0){
            #pragma omp parallel for if(enable_inner_parallel) schedule(static)
            for ( int j = 0; j < C; ++j){
                result(0,j) = ( -3 * M(0,j) + 4 * M(1,j) - M(2,j)) / (2 * d);
                result(R-1,j) = (3 * M(R-1,j) - 4 * M(R-2,j) + M(R-3,j)) / (2 * d);
                for ( int i = 1; i < R - 1; ++i){
                    result(i,j) = (M(i+1,j) - M(i-1,j)) / (2 * d);
                }
            }   
        }else{
            #pragma omp parallel for if(enable_inner_parallel) schedule(static)
            for ( int i = 0; i < R; ++i){
                result(i, 0) = (-3 * M(i,0) + 4 * M(i,1) - M(i,2)) / (2 * d);
                result(i, C-1) = (3 * M(i,C-1) - 4 * M(i,C-2) + M(i,C-3)) / (2 * d);
                for ( int j = 1; j < C - 1; ++j){
                    result(i,j) = (M(i,j+1) - M(i,j-1)) / (2 * d);
                }
            }
        }
        return result;
    }

    Eigen::MatrixXd LevelSetUtils::evolve_kappa(const Eigen::MatrixXd &phi, double dx, double dy, double b){
        
        Eigen::MatrixXd phi_x = _gradient_1d(phi, dx, 1);
        Eigen::MatrixXd phi_y = _gradient_1d(phi, dy, 0);

        Eigen::MatrixXd phi_xy = _gradient_1d(phi_x, dy, 0);
        Eigen::MatrixXd phi_xx = _gradient_1d(phi_x, dx, 1);
        Eigen::MatrixXd phi_yy = _gradient_1d(phi_y, dy, 0);

        Eigen::MatrixXd abs_grad_sq = phi_x.array().square() + phi_y.array().square();
        Eigen::MatrixXd numerator = phi_xx.array() * phi_y.array().square() 
                                    - 2.0 * phi_x.array() * phi_y.array() * phi_xy.array() 
                                    + phi_yy.array() * phi_x.array().square();
        double eps = std::numeric_limits<double>::epsilon();
        Eigen::MatrixXd kappa_abs_phi = Eigen::MatrixXd::Zero(phi.rows(), phi.cols());
        for (int i = 0; i < phi.size(); ++i) {
            if (abs_grad_sq(i) > eps)
                kappa_abs_phi(i) = numerator(i) / abs_grad_sq(i);
        }

        return b * kappa_abs_phi;
    }

    double LevelSetUtils::get_dt_normal_kappa(double alpha, double dx, double dy, const Eigen::MatrixXd &H1_abs, const Eigen::MatrixXd &H2_abs, double b){
           double normal_term = H1_abs.maxCoeff() / dx + H2_abs.maxCoeff() / dy;
           double kappa_term = b * (1.0 / (dx * dx) + 1.0 / (dy * dy)) * 2;
           double denominator = normal_term + kappa_term;
           if (denominator < 1e-9){
               return 1e-3;
           }
           return alpha / denominator;
    }

    Eigen::MatrixXd LevelSetUtils::reinit_sdf(const Eigen::MatrixXd &phi, double dx, double dy){
        int R = phi.rows(), C = phi.cols();
        cv::Mat mask_out(R, C, CV_8U);
        cv::Mat mask_in(R, C, CV_8U);
        for (int i = 0; i < R; ++i){
            for (int j = 0; j < C; ++j){
                bool in_side = (phi(i,j) > 0);
                mask_out.at<uchar>(i,j) = in_side? 0 : 255;
                mask_in.at<uchar>(i,j) = in_side? 255 : 0;

            }
        }
        cv::Mat dist_out_cv, dist_in_cv;
        cv::distanceTransform(mask_out, dist_out_cv, cv::DIST_L2, 5);
        cv::distanceTransform(mask_in,  dist_in_cv,  cv::DIST_L2, 5);

        // 乘以间距 (假设 dx == dy)
        double d = dx;   // 各向同性
        Eigen::MatrixXd sdf(R, C);
        for (int i = 0; i < R; ++i) {
            for (int j = 0; j < C; ++j) {
                // 内正外负（与 Python reinit_SD_FMM 对齐）
                // 图形内部: dist_in=0, dist_out>0 → sdf = dist_out - dist_in > 0
                // 背景:     dist_in>0, dist_out=0 → sdf = dist_out - dist_in < 0
                sdf(i, j) = dist_in_cv.at<float>(i, j) * d
                        - dist_out_cv.at<float>(i, j)  * d;
            }
        }
        return sdf;
    }


    void LevelSetUtils::_monitor_sdf_quality(const Eigen::MatrixXd &phi, double dx, double dy,  int iter){
        Eigen::MatrixXd phi_x = _gradient_1d(phi, dx, 1);
        Eigen::MatrixXd phi_y = _gradient_1d(phi, dy, 0);
        Eigen::MatrixXd grad_mag = (phi_x.array().square() + phi_y.array().square()).sqrt();
        double max_grad = grad_mag.maxCoeff();
        double min_grad = grad_mag.minCoeff();
        double avg_grad = grad_mag.mean();
        double max_deviation = ((grad_mag.array() - 1.0).abs()).maxCoeff();
        double std_norm = grad_mag.array().sqrt().mean();
        std::cout<< " SDF Quality: iter = " << iter << "\n";
        std::cout<< " max_grad = " << max_grad << "\n";
        std::cout<< " min_grad = " << min_grad << "\n";
        std::cout<< " avg_grad,ideal 1= " << avg_grad << "\n";
        std::cout<< " max_deviation = " << max_deviation << "\n";
        std::cout<< " std_norm = " << std_norm << "\n"; 

    }






}
