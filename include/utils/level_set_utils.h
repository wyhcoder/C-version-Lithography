#pragma once

#include <Eigen/Dense>
#include <string>

namespace litho {

    struct Evolve_Params {
        Eigen::MatrixXd delta;
        Eigen::MatrixXd H1_abs;
        Eigen::MatrixXd H2_abs;
    };

    class LevelSetUtils {
    public:
        static Eigen::MatrixXd der_weno5(const Eigen::MatrixXd& M, double dx, std::string direction, int axis = -1);

        static Eigen::VectorXd weno5_1d(const Eigen::VectorXd& v, int N_orig, double dx, std::string dir);

        static Eigen::MatrixXd pad_edge(const Eigen::MatrixXd& data, int pad, int axis);

        static Eigen::MatrixXd _select_upwind_deriv(const Eigen::MatrixXd& Vn, const Eigen::MatrixXd& der_minus, const Eigen::MatrixXd& der_plus);

        static Evolve_Params evolve_normal_WENO_godunov(const Eigen::MatrixXd& phi, const Eigen::MatrixXd& Vn, double dy, double dx);

        static Eigen::MatrixXd evolve_kappa(const Eigen::MatrixXd& phi, double dx, double dy, double b);

        static Eigen::MatrixXd _gradient_1d(const Eigen::MatrixXd& M, double d, int axis);

        static double get_dt_normal_kappa(double alpha, double dx, double dy, const Eigen::MatrixXd& H1_abs, const Eigen::MatrixXd& H2_abs, double b);

        static Eigen::MatrixXd reinit_sdf(const Eigen::MatrixXd& phi, double dx, double dy);

        static void _monitor_sdf_quality(const Eigen::MatrixXd& phi, double dx, double dy, int iter);
    };


}