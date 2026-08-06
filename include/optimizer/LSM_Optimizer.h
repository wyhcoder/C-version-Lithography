#pragma once

#include<Eigen/Dense>
#include <vector>
#include "lithography_simulator.h"
#include "litho_prepare.h"
#include "imaging.h"
#include "gradient.h"


namespace litho {

    struct Gradien_dt{
        Eigen::MatrixXd Normal;
        double dt;
    };

    struct Level_set_result{
        Eigen::MatrixXd sdf;
        Eigen::MatrixXd mask;
        Eigen::MatrixXd initial_sdf;
    };

    class LSM_Optimizer{

        public:
            LSM_Optimizer(LithographySimulator& simulator, ImagingCache& cache, double dx, double dy, double cfl, double b, int iter);
            ~LSM_Optimizer() = default;
            Level_set_result optimize(Eigen::MatrixXd &mask);

            // 设置非主图形区域显影惩罚
            //   penalty=0 表示不启用
            void set_penalty(double penalty, double threshold = 0.01) {
                _penalty = penalty;
                _penalty_threshold = threshold;
            }
            
            




        private:
            Gradien_dt _caculate_evolution_term(Eigen::MatrixXd &phi, Eigen::MatrixXd &Gm);

            // -依赖 成像
            LithographySimulator _litho_simulator;
            ImagingCache _imaging_cache;
            Imaging _imaging;
            Gradient _gradient;
           
            
            
            // -参数
            double _dx;
            double _dy;
            double _cfl;
            double _b;
            int _iter;
            Eigen::MatrixXd _target_mask;
            std::vector<double> _error_history;

            // -中间变量
            Eigen::MatrixXd _phi_n;
            Eigen::MatrixXd _best_mask;
            Eigen::MatrixXd _Gm;
            Imaging_Result _imaging_result;
            Eigen::MatrixXd _phi_1;
            Eigen::MatrixXd _phi_n_plus_1;
            Eigen::MatrixXd _phi_M;
            Eigen::MatrixXd _temp;
            Eigen::MatrixXd _best_sdf;
            Eigen::MatrixXd _initial_sdf;
            
            Gradien_dt _normal_dt;
            Gradien_dt _normal_dt_plus_1;
            double _pe;

            // 非主图形区域显影惩罚
            double _penalty = 0.0;          // 默认 0 = 不启用
            double _penalty_threshold = 0.01;

    };



}