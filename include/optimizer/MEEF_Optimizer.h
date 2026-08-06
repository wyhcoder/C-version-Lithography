#pragma once

#include <Eigen/Dense>
#include <opencv2/core/mat.hpp>
#include <limits>
#include <string>
#include <vector>
#include "ep_select.h"
#include "imaging.h"
#include "lithography_simulator.h"
#include "litho_prepare.h"


namespace litho {

    struct MEEFPipelineConfig{
        std::string pattern_name;
        std::string ls_mask_path;
        std::string save_file_path;
        std::string move_strategy = "xy";
        int iter = 100;
        double step_tol = 0.0;  // 0 表示关闭位移提前终止
        int patience = 3;
        // 主图形取控制点间隔
        int main_cp_interval = 7;
        std::string main_symmetry = "none";
        // SRAF cp
        int sraf_cp_interval = 5;
        int sraf_min_cps = 8;
        int sraf_min_aera = 50;
        // MSAA
        int msaa_level = 16;
        // 曲线类型
        std::string curve_type = "BS";
        double delta = 0.15;
        // 膨胀半径
        int dilate_radius = 2;
        // 线段间隔
        int interval_line = 5;
        // 角度间隔
        int interval_corner = 2;
        // 中间点权重
        double mid_weight = 4.0;
        // 其他点权重
        double other_weight = 1.0;
        // true：MEEF 每一行再乘 weight_epe，只优化 WEPE 标记的关键 EP 点。
        // false：仅使用 weight_meef，所有 EP 点参与优化。
        bool optimize_wepe_only = false;
    };
    struct Main_SRAF{
        Eigen::MatrixXd main_mask;
        Eigen::MatrixXd sraf_mask;
    };

    // X/Y 解耦的 MEEF 矩阵；行对应 EP，列对应展平后的主图形控制点。
    struct MEEFMatrixXY {
        Eigen::MatrixXd mx;
        Eigen::MatrixXd my;
    };


    class MEEF_Optimizer{
        public:
            MEEF_Optimizer(const LithographySimulator& simulator,const ImagingCache& cache, const MEEFPipelineConfig& config);

            static Main_SRAF _split_main_and_sraf(
                const Eigen::MatrixXd& lsm_mask,
                const Eigen::MatrixXd& target_mask,
                int dilate_radius = 2);

            void optimize();

            const std::vector<double>& epe_history() const { return _epe_history; }
            const std::vector<double>& wepe_history() const { return _wepe_history; }
            const std::vector<double>& pe_history() const { return _pe_history; }
            const std::vector<double>& time_history() const { return _time_history; }
            const ControlPoints& final_control_points() const { return _main_control_points; }

            // 对主图形每个控制点分别沿真实 x/y 方向做 ±delta 中心差分。
            // OpenMP 并行执行 4*num_cps 个扰动任务，SRAF mask 在所有任务中固定。
            MEEFMatrixXY build_meef_matrix_xy(const ControlPoints& current_cps) const;
            MEEFMatrixXY build_meef_matrix_xy() const {
                return build_meef_matrix_xy(_main_control_points);
            }


        private:
            static cv::Mat _to_cv8u(const Eigen::MatrixXd& M);
            static IPoints _cv_to_ipoints(const std::vector<cv::Point>& c);
            static  ControlPoints _extract_mask_control_points(const Eigen::MatrixXd& mask, int k, const std::string& symmetry);
            static IPoints _sample_elements(const IPoints& points, int k, const std::string& method);
            static Contour _ipoints_to_contour(const IPoints& points);
            static ControlPoints _extract_SRAF_control_points(const Eigen::MatrixXd& staf_mask, int k, int min_area, int min_cps);
            static void _save_cp_history(const ControlPoints& cps, const std::string& path, int iteration_idx);
            void _save_curve_history(const ControlPoints& cps, int iteration_idx) const;
            void _save_iteration_results(
                const Eigen::MatrixXd& mask,
                const Imaging_Result& imaging_result,
                const Eigen::VectorXd& delta_d,
                const MEEFMatrixXY& meef_matrix) const;
            void _save_history() const;
            void _save_best_results() const;
            static Eigen::VectorXd _SVD_get_delta(
                const Eigen::MatrixXd& M,
                const Eigen::RowVectorXd& e,
                double lambda,
                double energy_threshold = 0.96);
            static Eigen::VectorXd _log_space(double start, double end, int num);
            static std::pair<Eigen::VectorXd, Eigen::VectorXd> _compute_L_curve(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& lambda);
            static double _find_optimal_lambda(const Eigen::VectorXd& lambda, const Eigen::VectorXd& res_norms, const Eigen::VectorXd& x_norms);
            static double _find_truelambadas(const Eigen::MatrixXd& M, const Eigen::RowVectorXd& e0);
            static Eigen::VectorXd gradient(const Eigen::VectorXd& y, const Eigen::VectorXd& x);
            static ControlPoints _update_control_points(
                const ControlPoints& cps,
                const Eigen::VectorXd& delta_x,
                const Eigen::VectorXd& delta_y);
            // 把 main_cps / sraf_cps / eps / sraf_mask 保存到 _config.save_file_path
            void _save_meta();
            
            Main_SRAF _main_sraf;
            EpsResult _eps_result;
            ControlPoints _main_control_points;
            ControlPoints _sraf_control_points;
            LithographySimulator _simulator;
            ImagingCache _cache;
            MEEFPipelineConfig _config;
            Eigen::MatrixXd _target_mask;
            Eigen::MatrixXd _lsm_mask;
            int _num_eps;
            int _num_main_cps;
            int _num_sraf_cps;
            int _num_weps;
            Eigen::MatrixXd _render_sraf_mask;
            Eigen::MatrixXd _render_main_mask;
            Eigen::MatrixXd _render_initial_mask;

            // 优化历史：成员保存，便于 optimize() 返回后读取和持久化。
            std::vector<int> _iteration_history;
            std::vector<double> _time_history;
            std::vector<double> _wepe_history;
            std::vector<double> _epe_history;
            std::vector<double> _pe_history;

            // 最优结果：需要跨迭代保存完整快照，因此属于类状态。
            double _best_wepe = std::numeric_limits<double>::max();
            double _best_wepe_epe = std::numeric_limits<double>::max();
            int _best_wepe_iteration = -1;
            ControlPoints _best_wepe_cps;
            Eigen::MatrixXd _best_wepe_mask;
            Imaging_Result _best_wepe_imaging;

            double _best_epe = std::numeric_limits<double>::max();
            double _best_epe_wepe = std::numeric_limits<double>::max();
            int _best_epe_iteration = -1;
            ControlPoints _best_epe_cps;
            Eigen::MatrixXd _best_epe_mask;
            Imaging_Result _best_epe_imaging;
    };

}  // namespace litho
