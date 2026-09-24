#include "LSM_Optimizer.h"
#include "level_set_utils.h"
#include "litho_prepare.h"
#include "lithography_simulator.h"
#include "loss.h"
#include "simulation_parameters.h"
#include <iostream>
#include <cstdlib>


namespace litho {
    LSM_Optimizer::LSM_Optimizer(LithographySimulator& simulator, ImagingCache& cache,double dx, double dy, double cfl, double b, int iter)
        :  _litho_simulator(simulator),
            _imaging_cache(cache),
            _imaging(_imaging_cache),
            _gradient(_imaging_cache,
                    simulator._params.resist.threshold,
                    simulator._params.resist.alpha),
            _dx(dx),
            _dy(dy),
            _cfl(cfl),
            _b(b),
            _iter(iter) {

            int N = _litho_simulator._grid.size();
            _target_mask = _litho_simulator._mask.data();
            _phi_n = Eigen::MatrixXd::Zero(N, N);
            _best_mask = Eigen::MatrixXd::Zero(N, N);
            _Gm = Eigen::MatrixXd::Zero(N, N);
            _imaging_result.aerial_image = Eigen::MatrixXd::Zero(N, N);
            _imaging_result.wafer_image = Eigen::MatrixXd::Zero(N, N);
            _normal_dt.Normal = Eigen::MatrixXd::Zero(N, N);
            _phi_1 = Eigen::MatrixXd::Zero(N, N);
            _phi_n_plus_1 = Eigen::MatrixXd::Zero(N, N);
            _normal_dt_plus_1.Normal = Eigen::MatrixXd::Zero(N, N);
            _phi_M = Eigen::MatrixXd::Zero(N, N);
            _temp = Eigen::MatrixXd::Zero(N, N);
            _best_sdf = Eigen::MatrixXd::Zero(N, N);

            }
        




    
    Gradien_dt LSM_Optimizer::_caculate_evolution_term(Eigen::MatrixXd &phi, Eigen::MatrixXd &Gm){

        Evolve_Params params = LevelSetUtils::evolve_normal_WENO_godunov(phi,Gm,_dy,_dx); 

        Eigen::MatrixXd delta_kappa = LevelSetUtils::evolve_kappa(phi, _dx, _dy, _b);

        Eigen::MatrixXd Normal =  delta_kappa - params.delta;

        double dt = LevelSetUtils::get_dt_normal_kappa(_cfl, _dx, _dy, params.H1_abs, params.H2_abs, _b);
        
        return {Normal, dt};
    }

    Level_set_result LSM_Optimizer::optimize(Eigen::MatrixXd &mask){
        // initial of level set function
        _phi_n = LevelSetUtils::reinit_sdf((mask.array() - 0.5).matrix(), _dx, _dy);
        _initial_sdf = _phi_n;
        // SaveTxt::save_mat(_initial_sdf, "initial_sdf.txt");

        const double threshold = _litho_simulator._params.resist.threshold;
        const double alpha = _litho_simulator._params.resist.alpha;

        if (_penalty > 0.0) {
            std::cout << "Penalty enabled: penalty=" << _penalty
                      << " threshold=" << _penalty_threshold << std::endl;
        }

        double min_error = std::numeric_limits<double>::max();
        _best_mask = mask;
        _temp = mask;
        // optimization
        for(int i = 0; i < _iter; ++i){

            _imaging_result = _imaging.compute(_temp, threshold, alpha);

            // loss
            if (_penalty > 0.0) {
                _pe = Loss::pe_loss_with_penalty(
                    _imaging_result.wafer_image, _target_mask,
                    _penalty, _penalty_threshold);
            } else {
                // _pe = Loss::pe_loss(_imaging_result.wafer_image, _target_mask);
                _pe = Loss::pe_loss_with_penalty(_imaging_result.wafer_image, _target_mask, 1.0, 0.01);
            }
            std::cout << "--------iter: " << i << " --------pe: " << _pe << std::endl;
            if (_pe < min_error){
                min_error = _pe;
                _best_mask = _temp;
                _best_sdf = _phi_M;

            }

            _error_history.push_back(_pe);

            if (_penalty > 0.0) {
                _Gm = _gradient.pe_gradient_with_penalty(
                    _imaging_result.wafer_image, _target_mask,
                    _imaging.get_electric_field(),
                    _penalty, _penalty_threshold);
            } else {
                _Gm = _gradient.pe_gradient(_imaging_result.wafer_image, _target_mask, _imaging.get_electric_field());
            }
            // SaveTxt::save_mat(_Gm, "Gm.txt");

            // use WENO5 to calculate the evolution term to update the level set function
            _normal_dt = _caculate_evolution_term(_phi_n, _Gm);
            _phi_1 = _phi_n + _normal_dt.dt * _normal_dt.Normal;
            _normal_dt_plus_1 = _caculate_evolution_term(_phi_1, _Gm);
            _phi_n_plus_1 = 0.5 * _phi_n + 0.5 * (_phi_1 + _normal_dt.dt * _normal_dt_plus_1.Normal);
            
            // update the level set function
            _phi_M  = _phi_n_plus_1;
            
            if (i % 20 == 0 && i > 0){
                std::cout << "--- reinit SDF at iter " << i << " ---" << std::endl;
                _phi_M = LevelSetUtils::reinit_sdf(_phi_M, _dx, _dy);
            }

            _phi_n = _phi_M;
            
            // LevelSetUtils::_monitor_sdf_quality(_phi_M, _dx, _dy, i);
            _temp = (_phi_M.array() >= 0).cast<double>().matrix();
            // SaveTxt::save_mat(_phi_M,    "phi_M.txt");
            // SaveTxt::save_mat(_temp,    "temp.txt");

           
            // SaveTxt::save_mat(_imaging_result.wafer_image, "wafer_image.txt");
            // SaveTxt::save_mat(_imaging_result.aerial_image, "aerial_image.txt");
            // ── 弹窗可视化（关窗口后继续） ──
            // 一图三子图：phi_M / temp / wafer
            // std::string base = "/Users/wyh/Desktop/学校/Litho_cpp";
            // std::string cmd = "python3 " + base + "/scripts/show_multi.py "
            //     "phi_M.txt hot temp.txt viridis aerial_image.txt viridis wafer_image.txt viridis  Gm.txt viridis"
            //     " --title_prefix 'iter " + std::to_string(i+1) + "'";
            // std::system(cmd.c_str());

            

        }
        return {_best_sdf, _best_mask, _initial_sdf};


    }


}