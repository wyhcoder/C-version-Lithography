#include "pv_band_computer.h"
#include <fftw3.h>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "litho_prepare.h"
#include "lithography_simulator.h"
#include "pupil.h"
#include "imaging.h"
#include "fft.h"
namespace litho {
    PvbandComputer::PvbandComputer(LithographySimulator& simulator, double dose_magrin, double defocus_range, int defocus_step,std::string mode):
        _litho_simulator(simulator),
        _dose_magrin(dose_magrin),
        _defocus_range(defocus_range),
        _defocus_step(defocus_step),
        _mode(mode){
            _threshold = _litho_simulator._params.resist.threshold;
            _alpha = _litho_simulator._params.resist.alpha;
            _na = _litho_simulator._params.optics.na;
            _n = _litho_simulator._params.optics.refractive_index;
            _wavelength = _litho_simulator._params.system.wavelength_nm;
            _threshold_high = (1 + dose_magrin) * _threshold;
            _threshold_low = (1 - dose_magrin) * _threshold;

            if (_mode != "dose_only" && _mode != "defocus_only" && _mode != "full") {
                throw std::invalid_argument("PvbandComputer: unsupported mode '" + _mode + "'");
            }
            if (_dose_magrin < 0.0 || _defocus_range < 0.0 || _defocus_step < 1) {
                throw std::invalid_argument("PvbandComputer: invalid dose/defocus sampling parameters");
            }

            // dose_only 不需要离焦采样，只预计算 nominal focus 的 cache。
            if (_mode == "dose_only" ||
                _defocus_range == 0.0 || _defocus_step == 1) {
                _defocus_list = {0.0};
            } else {
                // range 表示总焦深范围：[-range/2, +range/2]。
                const double half_range = _defocus_range / 2.0;
                const Eigen::VectorXd defocus = Eigen::VectorXd::LinSpaced(
                    _defocus_step, -half_range, half_range);
                _defocus_list.resize(_defocus_step);
                for (int i = 0; i < _defocus_step; ++i) {
                    _defocus_list[i] = defocus(i);
                }
            }
        int N = _litho_simulator._grid.size();
        int sz = N * N;
        _current_wafer.resize(N, N);
        _max_min_wafer.max_wafer.resize(N, N);
        _max_min_wafer.min_wafer.resize(N, N);
        _pupil_params.n = _n;
        _pupil_params.wavelength = _wavelength;
        _pupil_params.NA = _na;
        _pupil_params.defocus_nm = 0.0;
        _pv_map.resize(N, N);
        _fft_in   = fftw_alloc_complex(sz);
        _fft_out  = fftw_alloc_complex(sz);
        _plan_fwd = fftw_plan_dft_2d(N, N, _fft_in, _fft_out,
                                 FFTW_FORWARD,  FFTW_MEASURE); //FFTW_PATIENT FFTW_MEASURE
        _plan_inv = fftw_plan_dft_2d(N, N, _fft_in, _fft_out,
                                 FFTW_BACKWARD, FFTW_MEASURE);
        (void)sz;  // sz 保留以防将来使用
        _build_defocus_caches();
        std::cout<<"pvband computer init"<<std::endl;
        };

    PvbandComputer::~PvbandComputer() {
        fftw_destroy_plan(_plan_fwd);
        fftw_destroy_plan(_plan_inv);
        fftw_free(_fft_in);
        fftw_free(_fft_out);
    }

    Eigen::MatrixXd PvbandComputer::_sigmoid_resist(
        const Eigen::MatrixXd& aerial_image,
        double threshold) const
    {
        return (1.0 / (1.0 + (-_alpha * (aerial_image.array() - threshold)).exp())).matrix();
    }

    void PvbandComputer::_update_global_extrema(
        const Eigen::MatrixXd& wafer,
        Eigen::MatrixXd& max_wafer,
        Eigen::MatrixXd& min_wafer,
        bool& initialize)
    {
        if (!initialize) {
            max_wafer = wafer;
            min_wafer = wafer;
            initialize = true;
        } else {
            max_wafer = max_wafer.cwiseMax(wafer);
            min_wafer = min_wafer.cwiseMin(wafer);
        }
    }

    double PvbandComputer::_compute_z4_coefficient(double defocus){
        double z4_coeff = (_na * _na * defocus) / (4 * _n * _wavelength);
        return z4_coeff;
    }

    void PvbandComputer::_build_defocus_caches(){
        _defocus_caches.clear();
        _defocus_caches.reserve(_defocus_list.size());

        for (double defocus : _defocus_list) {
            std::unordered_map<int, double> temp_aberrations =
                _litho_simulator._params.optics.aberrations;
            temp_aberrations[4] =
                _compute_z4_coefficient(defocus) +
                _litho_simulator._params.optics.aberrations[4];

            _pupil_params.defocus_nm = defocus;
            _pupil_params.zernike_coeffs = temp_aberrations;
            Pupil temp_pupil(_pupil_params,
                             _litho_simulator._grid.grid_coords().Fx_2d,
                             _litho_simulator._grid.grid_coords().Fy_2d);

            LithoPrepare local_prepare(_litho_simulator._grid, temp_pupil,
                                       _litho_simulator._source, true);
            _defocus_caches.push_back(local_prepare.cache());
        }
    };
    
    Pvband_result PvbandComputer::compute_pvband(Eigen::MatrixXd& mask) {
        std::vector<double> thresholds;

        if (_mode == "dose_only") {
            thresholds = {_threshold_low, _threshold_high};
        } else if (_mode == "defocus_only") {
            thresholds = {_threshold};
        } else {  // full
            thresholds = {_threshold_low, _threshold, _threshold_high};
        }

        bool initialized = false;
        for (const ImagingCache& cache : _defocus_caches) {
            Imaging imaging(cache);
            const Eigen::MatrixXd aerial_image =
                imaging.compute(mask, _threshold, _alpha).aerial_image;

            for (double threshold : thresholds) {
                const Eigen::MatrixXd wafer =
                    _sigmoid_resist(aerial_image, threshold);
                _update_global_extrema(wafer,
                                       _max_min_wafer.max_wafer,
                                       _max_min_wafer.min_wafer,
                                       initialized);
            }
        }

        if (!initialized) {
            throw std::runtime_error("PvbandComputer: no process corners were evaluated");
        }

        _pv_map = (_max_min_wafer.max_wafer.array() -
                   _max_min_wafer.min_wafer.array()).matrix();
        _pv_loss = _pv_map.sum();
        return {_pv_map, _pv_loss};
    }

    Eigen::MatrixXd PvbandComputer::_sigmoid_derivative(
        const Eigen::MatrixXd& wafer) const
    {
        return (_alpha * wafer.array() * (1.0 - wafer.array())).matrix();
    }

    Eigen::MatrixXd PvbandComputer::_backpropagate(
        const Eigen::MatrixXd& dLdI,
        const std::vector<Eigen::MatrixXcd>& electric_field,
        const ImagingCache& cache)
    {
        const int N = cache.N;
        const int K = static_cast<int>(electric_field.size());
        const double invN = 1.0 / (static_cast<double>(N) * N);
        const auto& Hf = cache.H_k_frequence;
        const auto& ws = cache.source_ws;
        const auto& sv = cache.socs_vals;
        const bool is_abbe = (K == static_cast<int>(ws.size()));

        if (dLdI.rows() != N || dLdI.cols() != N) {
            throw std::invalid_argument("PvbandComputer: dLdI dimensions do not match cache");
        }
        if (static_cast<int>(Hf.size()) < K ||
            (!is_abbe && static_cast<int>(sv.size()) < K)) {
            throw std::invalid_argument("PvbandComputer: optical kernels are incomplete");
        }

        // 每个 k 仅写自己的 gradient slot；所有线程完成后固定按 k 顺序归约。
        std::vector<Eigen::MatrixXd> kernel_gradients(K);

        #pragma omp parallel
        {
            // FFTW plan 只读共享，执行时的 in/out buffer 必须线程私有。
            fftw_complex* local_in = fftw_alloc_complex(N * N);
            fftw_complex* local_out = fftw_alloc_complex(N * N);
            Eigen::MatrixXcd local_buf(N, N);

            #pragma omp for schedule(static)
            for (int k = 0; k < K; ++k) {
                const double weight = is_abbe ? ws(k) : sv[k];
                local_buf = electric_field[k].conjugate().array() * dLdI.array();

                FFT::ifftshift_inplace(local_buf);
                FFT::to_fftw(local_buf, local_in);
                fftw_execute_dft(_plan_fwd, local_in, local_out);
                FFT::from_fftw(local_out, local_buf);

                local_buf.array() *= Hf[k].conjugate().array();

                FFT::to_fftw(local_buf, local_in);
                fftw_execute_dft(_plan_inv, local_in, local_out);
                FFT::from_fftw(local_out, local_buf);
                FFT::fftshift_inplace(local_buf);

                // I = Σ w_k |E_k|² / Σw，实数 mask 的 d|E|²/dM 含 2·Re{...}。
                kernel_gradients[k] =
                    (2.0 * weight * invN * local_buf.real().array()).matrix();
            }

            fftw_free(local_in);
            fftw_free(local_out);
        }

        Eigen::MatrixXd gradient = Eigen::MatrixXd::Zero(N, N);
        for (int k = 0; k < K; ++k) {
            gradient += kernel_gradients[k];
        }

        const double source_weight_sum = ws.sum();
        if (source_weight_sum == 0.0) {
            throw std::runtime_error("PvbandComputer: source weight sum is zero");
        }
        return gradient / source_weight_sum;
    }

    Eigen::MatrixXd PvbandComputer::compute_pvloss_gradient(Eigen::MatrixXd& mask) {
        std::vector<double> thresholds;

        if (_mode == "dose_only") {
            thresholds = {_threshold_low, _threshold_high};
        } else if (_mode == "defocus_only") {
            thresholds = {_threshold};
        } else {  // full
            thresholds = {_threshold_low, _threshold, _threshold_high};
        }

        const int N = _litho_simulator._grid.size();
        const int defocus_count = static_cast<int>(_defocus_caches.size());
        const int threshold_count = static_cast<int>(thresholds.size());
        if (defocus_count == 0 || threshold_count == 0) {
            throw std::runtime_error("PvbandComputer: no process corners for PV gradient");
        }

        // 前向阶段：逐像素记录 PV max/min 来自哪个 defocus 和 threshold corner。
        Eigen::MatrixXd max_wafer;
        Eigen::MatrixXd min_wafer;
        Eigen::MatrixXi max_defocus_winner;
        Eigen::MatrixXi max_threshold_winner;
        Eigen::MatrixXi min_defocus_winner;
        Eigen::MatrixXi min_threshold_winner;
        bool initialized = false;

        for (int di = 0; di < defocus_count; ++di) {
            const ImagingCache& cache = _defocus_caches[di];
            Imaging imaging(cache);
            const Eigen::MatrixXd aerial_image =
                imaging.compute(mask, _threshold, _alpha).aerial_image;

            for (int ti = 0; ti < threshold_count; ++ti) {
                const Eigen::MatrixXd wafer =
                    _sigmoid_resist(aerial_image, thresholds[ti]);

                if (!initialized) {
                    max_wafer = wafer;
                    min_wafer = wafer;
                    max_defocus_winner = Eigen::MatrixXi::Constant(N, N, di);
                    max_threshold_winner = Eigen::MatrixXi::Constant(N, N, ti);
                    min_defocus_winner = Eigen::MatrixXi::Constant(N, N, di);
                    min_threshold_winner = Eigen::MatrixXi::Constant(N, N, ti);
                    initialized = true;
                    continue;
                }

                // Tie 时保留先遇到的 corner，选择确定的 hard max/min 次梯度。
                for (Eigen::Index p = 0; p < wafer.size(); ++p) {
                    if (wafer(p) > max_wafer(p)) {
                        max_wafer(p) = wafer(p);
                        max_defocus_winner(p) = di;
                        max_threshold_winner(p) = ti;
                    }
                    if (wafer(p) < min_wafer(p)) {
                        min_wafer(p) = wafer(p);
                        min_defocus_winner(p) = di;
                        min_threshold_winner(p) = ti;
                    }
                }
            }
        }

        // 与 compute_pvband(mask) 使用相同的 hard max/min 前向定义。
        _max_min_wafer.max_wafer = max_wafer;
        _max_min_wafer.min_wafer = min_wafer;
        _pv_map = (max_wafer.array() - min_wafer.array()).matrix();
        _pv_loss = _pv_map.sum();

        // 反向阶段：逐 defocus 重建其 electric field，只反传该 corner 获胜的像素。
        Eigen::MatrixXd total_gradient = Eigen::MatrixXd::Zero(N, N);

        for (int di = 0; di < defocus_count; ++di) {
            const ImagingCache& cache = _defocus_caches[di];
            Imaging imaging(cache);
            const Eigen::MatrixXd aerial_image =
                imaging.compute(mask, _threshold, _alpha).aerial_image;
            Eigen::MatrixXd dLdI = Eigen::MatrixXd::Zero(N, N);
            bool has_contribution = false;

            for (int ti = 0; ti < threshold_count; ++ti) {
                const Eigen::MatrixXd wafer =
                    _sigmoid_resist(aerial_image, thresholds[ti]);
                const Eigen::MatrixXd sigmoid_derivative =
                    _sigmoid_derivative(wafer);

                for (Eigen::Index p = 0; p < wafer.size(); ++p) {
                    if (max_defocus_winner(p) == di &&
                        max_threshold_winner(p) == ti) {
                        dLdI(p) += sigmoid_derivative(p);
                        has_contribution = true;
                    }
                    if (min_defocus_winner(p) == di &&
                        min_threshold_winner(p) == ti) {
                        dLdI(p) -= sigmoid_derivative(p);
                        has_contribution = true;
                    }
                }
            }

            // 没有 winner，或 max/min 在该 defocus 完全抵消时，无需做昂贵的反传。
            if (has_contribution && dLdI.squaredNorm() > 0.0) {
                total_gradient += _backpropagate(
                    dLdI, imaging.get_electric_field(), cache);
            }
        }

        return total_gradient;
    }








    


}
