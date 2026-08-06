#pragma once
#include <Eigen/Dense>
#include "litho_prepare.h"
#include "lithography_simulator.h"
#include "pupil.h"
#include <string>
#include <vector>
#include <fftw3.h>


namespace litho {

    struct Max_Min_wafer{
        Eigen::MatrixXd max_wafer;
        Eigen::MatrixXd min_wafer;
    };

    struct Cache_result{
        std::vector<Eigen::MatrixXcd> kernels;
        std::vector<double> kernels_values;
        double normal_kernel_value;
    };

    struct Pvband_result{
        Eigen::MatrixXd pv_map;
        double pv_loss;
    };

    class PvbandComputer {
        public:
            PvbandComputer(LithographySimulator& simulator, double dose_magrin, double defocus_range, int defocus_step, std::string mode);
            
            Eigen::MatrixXd compute_pvloss_gradient(Eigen::MatrixXd &mask);
            ~PvbandComputer();

            Max_Min_wafer compute_pvband();

            Eigen::MatrixXd pv_loss_computer(LithographySimulator &simulator, double dose_magrin, double defocus_range, int defocus_step,std::string mode);

            Pvband_result compute_pvband(Eigen::MatrixXd &mask);
        private:
            Eigen::MatrixXd _sigmoid_resist(const Eigen::MatrixXd& aerial_image,
                                             double threshold) const;

            void _update_global_extrema(const Eigen::MatrixXd& current_wafer,
                                        Eigen::MatrixXd& max_wafer,
                                        Eigen::MatrixXd& min_wafer,
                                        bool& initialize);

            double _compute_z4_coefficient(double defocus);

            void _recompute_cache_at_defocus(double defocus);

            Eigen::MatrixXd _sigmoid_derivative(const Eigen::MatrixXd& wafer) const;

            // 将 dL/dI 通过当前 defocus 对应的光学核反传到 mask。
            Eigen::MatrixXd _backpropagate(
                const Eigen::MatrixXd& dLdI,
                const std::vector<Eigen::MatrixXcd>& electric_field,
                const ImagingCache& cache);

            

            LithographySimulator _litho_simulator;
            double _dose_magrin;
            double _defocus_range;
            int _defocus_step;
            std::string _mode;
            double _threshold;
            double _alpha;
            double _na;
            double _n;
            double _wavelength;
            double _threshold_high;
            double _threshold_low;
            double _pv_loss;
            std::vector<double> _defocus_list;
            Eigen::MatrixXd _current_wafer;
            Eigen::MatrixXd _pv_map;
            Max_Min_wafer _max_min_wafer;
            Params _pupil_params;
            ImagingCache _temp_cache;

            fftw_complex *_fft_in;
            fftw_complex *_fft_out;
            fftw_plan     _plan_fwd;
            fftw_plan     _plan_inv;


            

    };



}