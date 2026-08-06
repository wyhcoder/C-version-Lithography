#include "MEEF_Optimizer.h"
#include "ep_select.h"
#include "imaging.h"
#include "loss.h"
#include "parametric.h"
#include "save_txt.h"
#include <Eigen/SVD>
#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace litho {

    MEEF_Optimizer::MEEF_Optimizer(const LithographySimulator& simulator,const ImagingCache& cache, const MEEFPipelineConfig& config):
        _simulator(simulator),
        _cache(cache),
        _config(config)
    {
        // 加载 ls_mask 并分离出主图形和SRAF
        SaveTxt::load_txt(_config.ls_mask_path , _lsm_mask);
        _target_mask = _simulator._mask.data();
        _main_sraf = _split_main_and_sraf(
            _lsm_mask, _target_mask, _config.dilate_radius);
        EpSelect ep_select(_target_mask, _config.mid_weight, _config.other_weight);
        _eps_result = ep_select.select_eps_others(_config.interval_line, _config.interval_corner);
        _main_control_points = _extract_mask_control_points(_target_mask, config.main_cp_interval, config.main_symmetry);
        _sraf_control_points = _extract_SRAF_control_points(_main_sraf.sraf_mask, config.sraf_cp_interval, config.sraf_min_aera, config.sraf_min_cps);
        _num_eps = static_cast<int>(_eps_result.eps.rows());
        _num_weps = static_cast<int>((_eps_result.weight_epe.array() == 1.).count());
        _num_main_cps = 0;
        _num_sraf_cps = 0;
        for (const auto& cp : _main_control_points) {
            _num_main_cps += static_cast<int>(cp.rows());
        }
        for (const auto& cp : _sraf_control_points) {
            _num_sraf_cps += static_cast<int>(cp.rows());
        }
        int N = _target_mask.rows();
        _render_sraf_mask.resize(N, N);
        _render_main_mask.resize(N, N);
        _render_initial_mask.resize(N, N);

        ParametricDemo parametric_sraf(
            config.curve_type, _target_mask, config.msaa_level);
        _render_sraf_mask = parametric_sraf.render_curve(_sraf_control_points);
        _render_main_mask = parametric_sraf.render_curve(_main_control_points);
        _render_initial_mask =
            _render_sraf_mask.array() + _render_main_mask.array();
        _save_meta();


        





    }
    // ── 转换：Eigen → cv::Mat ────────────────────────────────────────────────
    cv::Mat MEEF_Optimizer::_to_cv8u(const Eigen::MatrixXd& M){
        int rows = static_cast<int>(M.rows());
        int cols = static_cast<int>(M.cols());
        cv::Mat out(rows, cols, CV_8UC1);
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                out.at<uint8_t>(r, c) = static_cast<uint8_t>(
                    std::clamp(M(r, c) * 255.0, 0.0, 255.0));
        return out;
    }

    IPoints MEEF_Optimizer::_cv_to_ipoints(const std::vector<cv::Point>& c){
        IPoints out;
        out.reserve(c.size());
        for (const auto& p : c) out.push_back({p.y, p.x});
        return out;
    }

    Main_SRAF MEEF_Optimizer::_split_main_and_sraf(
        const Eigen::MatrixXd& lsm_mask,
        const Eigen::MatrixXd& target_mask,
        int dilate_radius){
        int H = static_cast<int>(lsm_mask.rows());
        int W = static_cast<int>(lsm_mask.cols());

        Main_SRAF result;
        result.main_mask = Eigen::MatrixXd::Zero(H, W);
        result.sraf_mask = Eigen::MatrixXd::Zero(H, W);
        if (H == 0 || W == 0) return result;

        if (dilate_radius < 0) {
            throw std::invalid_argument(
                "MEEF_Optimizer::_split_main_and_sraf: dilate_radius must be non-negative");
        }

        // 1. target_mask 二值化：>0 → 255，等价 Python (target_mask > 0).astype(uint8)
        cv::Mat origin_bin(H, W, CV_8UC1);
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c)
                origin_bin.at<uchar>(r, c) = (target_mask(r, c) > 0.0) ? 255 : 0;

        // 2. 圆形结构元膨胀（等价 scipy.disk(radius) + binary_dilation）
        //    MORPH_ELLIPSE 内切于 ksize×ksize 正方形时即为圆
        cv::Mat dilated_bin;
        if (dilate_radius == 0) {
            dilated_bin = origin_bin.clone();
        } else {
            int ksize = 2 * dilate_radius + 1;
            cv::Mat kernel = cv::getStructuringElement(
                cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));
            cv::dilate(origin_bin, dilated_bin, kernel);
        }

        // 3. 按膨胀掩膜拆分 lsm_mask —— 不改变像素灰度，仅按空间区域归类
        //    dilated 区域 → main_mask；其余 → sraf_mask
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                if (dilated_bin.at<uchar>(r, c) > 0) {
                    result.main_mask(r, c) = lsm_mask(r, c);
                } else {
                    result.sraf_mask(r, c) = lsm_mask(r, c);
                }
            }
        }
        return result;
    }
    IPoints MEEF_Optimizer::_sample_elements(const IPoints &pts, int k, const std::string& mode){
        int step = std::max(1, (mode == "skip" ? k+1 : k));
        IPoints out;
        for (int i = 0; i < pts.size(); i += step) {
            out.push_back(pts[i]);
        }
        return out;
    }

    Contour MEEF_Optimizer::_ipoints_to_contour(const IPoints &pts){
        int n = static_cast<int>(pts.size());
        Contour m(n,2);
        for(int i = 0; i < n; ++i){
            m(i, 0) = static_cast<double>(pts[i][0]);
            m(i, 1) = static_cast<double>(pts[i][1]);
        }
        return m;
    }


    ControlPoints MEEF_Optimizer::_extract_mask_control_points(const Eigen::MatrixXd &mask, int k, const std::string& symmetry){
        cv::Mat mask_cv = _to_cv8u(mask);
        std::vector<std::vector<cv::Point>> cv_contours;
        cv::findContours(mask_cv, cv_contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
        ControlPoints result;
        for(auto &c : cv_contours){
            result.push_back(_ipoints_to_contour(_sample_elements(_cv_to_ipoints(c), k, "skip")));
        }
        // 对称性检查以及对称性处理
        if (symmetry.empty() || result.size() < 2){
            return result;
        }
        int h = mask.rows(), w = mask.cols(), cy = h/2, cx = w/2;
        Contour sym(result[0].rows(), 2);
        for (int i = 0; i < (int)result[0].rows(); ++i) {
            double y=result[0](i,0), x=result[0](i,1);
            if      (symmetry=="center")     { sym(i,0)=2*cy-y; sym(i,1)=2*cx-x; }
            else if (symmetry=="left-right") { sym(i,0)=y;      sym(i,1)=2*cx-x; }
            else if (symmetry=="diagonal")   { sym(i,0)=h-1-x;  sym(i,1)=w-1-y;  }
            else throw std::invalid_argument("Unsupported symmetry: " + symmetry);
        }
        result[1] = sym;
        return result;

    }
    ControlPoints MEEF_Optimizer::_extract_SRAF_control_points(const Eigen::MatrixXd &sraf_mask, int k, int min_area, int min_cps){
        cv::Mat sraf_mask_cv = _to_cv8u(sraf_mask);

        // 1. 连通域分析 + 按面积过滤小连通块
        cv::Mat labels, stats, centroids;
        int n_labels = cv::connectedComponentsWithStats(
            sraf_mask_cv, labels, stats, centroids, 8, CV_32S);

        cv::Mat cleaned = sraf_mask_cv.clone();
        for (int lb = 1; lb < n_labels; ++lb) {
            if (stats.at<int>(lb, cv::CC_STAT_AREA) < min_area) {
                cleaned.setTo(0, labels == lb);
            }
        }

        // 2. 在清洗后的 mask 上提取外轮廓
        std::vector<std::vector<cv::Point>> cv_contours;
        cv::findContours(cleaned, cv_contours,
                         cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        // 3. 每条轮廓按间隔 k 采样；SRAF 块小，min_cps 保护：不足时缩小 step
        ControlPoints result;
        result.reserve(cv_contours.size());
        for (auto& c : cv_contours) {
            IPoints ip = _cv_to_ipoints(c);
            int N = (int)ip.size();
            if (N == 0) continue;

            int step = std::max(1, k + 1);                  // mode="skip"
            int n_by_step = (N + step - 1) / step;          // ceil(N / step)
            if (min_cps > 0 && n_by_step < min_cps) {
                step = std::max(1, N / min_cps);            // 缩小 step 保证 ≥ min_cps 个点
            }

            IPoints sampled;
            sampled.reserve((N + step - 1) / step);
            for (int i = 0; i < N; i += step) sampled.push_back(ip[i]);
            result.push_back(_ipoints_to_contour(sampled));
        }
        return result;
    }

    // ── 保存元数据（对应 Python _save_meta）──────────────────────────────
    // 输出文件（写入 _config.save_file_path）：
    //   main_cps.txt   —— 主图形控制点（# main contour i + y x）
    //   sraf_cps.txt   —— SRAF 控制点   （# sraf block i  + y x）
    //   eps.txt        —— EP 点（整数）
    //   sraf_mask.txt  —— SRAF mask（6 位小数，无 header，对齐 np.savetxt fmt="%.6f"）
    //   pipeline_config.json 暂未实现（Python 端 try/except 容错，可省略）
    void MEEF_Optimizer::_save_meta() {
        namespace fs = std::filesystem;
        fs::path out(_config.save_file_path);
        fs::create_directories(out);

        // 通用：写控制点文件（main contour / sraf block 两种 tag）
        auto write_cps = [&](const char* filename, const char* tag,
                             const ControlPoints& cps) {
            std::ofstream f(out / filename);
            f << std::fixed << std::setprecision(4);
            for (size_t i = 0; i < cps.size(); ++i) {
                f << "# " << tag << " " << i << "\n";
                for (int r = 0; r < cps[i].rows(); ++r) {
                    f << cps[i](r, 0) << " " << cps[i](r, 1) << "\n";
                }
            }
        };

        // 1. 主图形 CP
        write_cps("main_cps.txt", "main contour", _main_control_points);

        // 2. SRAF CP
        write_cps("sraf_cps.txt", "sraf block", _sraf_control_points);

        // 3. EP 点（整数，等价 np.savetxt(fmt="%d")）
        {
            std::ofstream f(out / "eps.txt");
            const Eigen::MatrixXd& eps = _eps_result.eps;
            for (int r = 0; r < eps.rows(); ++r) {
                for (int c = 0; c < eps.cols(); ++c) {
                    f << static_cast<int>(std::round(eps(r, c)));
                    if (c < eps.cols() - 1) f << " ";
                }
                f << "\n";
            }
        }

        // 通用：保存渲染后的 mask（6 位小数、无 header，兼容 np.loadtxt）
        auto write_mask = [&](const char* filename, const Eigen::MatrixXd& m) {
            std::ofstream f(out / filename);
            f << std::fixed << std::setprecision(6);
            for (int r = 0; r < m.rows(); ++r) {
                for (int c = 0; c < m.cols(); ++c) {
                    f << m(r, c);
                    if (c < m.cols() - 1) f << " ";
                }
                f << "\n";
            }
        };

        // 4. 渲染后的 SRAF / 主图形 mask
        write_mask("sraf_mask.txt", _render_sraf_mask);
        write_mask("main_mask.txt", _render_main_mask);
    }

    void MEEF_Optimizer::_save_cp_history(const ControlPoints& cps, const std::string& path, int iteration_idx){
        namespace fs = std::filesystem;
        fs::path cp_dir = fs::path(path)  / "curves"/ "cp_history";
        std::error_code ec;
        fs::create_directories(cp_dir, ec);
        char filename[64];
        std::snprintf(filename, sizeof(filename), "cp_iter_%03d.txt", iteration_idx);
        fs::path out_path = cp_dir / filename;
        std::ofstream f(out_path);
        if (!f.is_open()) {
        std::cerr << "  [warn] 保存 CP 历史失败 (iter " << iteration_idx
                  << "): 无法打开文件 " << out_path << "\n";
        return;
        }
        try{
            f << "# CP positions at iteration " << iteration_idx << "\n";
            f << "# num_contours = " << cps.size() << "\n";
            f << std::fixed << std::setprecision(6);
            for (int ci = 0; ci < (int)cps.size(); ++ci) {
                const Eigen::MatrixXd& cp = cps[ci];
                f << "# contour " << ci << " (num_points = " << cp.rows() << ")\n";
                for (int pi = 0; pi < cp.rows(); ++pi) {
                    f << cp(pi, 0) << " " << cp(pi, 1) << "\n";
                }
                f << "\n";
            }

        } catch (std::exception& e) {
            std::cerr << "  [warn] 保存 CP 历史失败 (iter " << iteration_idx
                      << "): " << e.what() << "\n";
        }
    


    }

    void MEEF_Optimizer::_save_curve_history(
        const ControlPoints& cps,
        int iteration_idx) const
    {
        namespace fs = std::filesystem;
        fs::path curve_dir = fs::path(_config.save_file_path) /
                             "curves" / "bspline_curves";
        fs::create_directories(curve_dir);

        char filename[64];
        std::snprintf(filename, sizeof(filename),
                      "bspline_iter_%03d.txt", iteration_idx);
        fs::path out_path = curve_dir / filename;
        std::ofstream f(out_path);
        if (!f) {
            throw std::runtime_error(
                "MEEF_Optimizer::_save_curve_history: cannot open " +
                out_path.string());
        }

        ParametricDemo parametric(
            _config.curve_type, _target_mask, _config.msaa_level);
        Polygons curves = parametric.get_curve_points(cps, 200);
        f << std::fixed << std::setprecision(6);
        f << "# Parametric curves at iteration " << iteration_idx << '\n';
        f << "# num_contours = " << curves.size() << '\n';
        for (size_t i = 0; i < curves.size(); ++i) {
            f << "# contour " << i
              << " (num_points = " << curves[i].rows() << ")\n";
            for (int r = 0; r < curves[i].rows(); ++r) {
                f << curves[i](r, 0) << ' ' << curves[i](r, 1) << '\n';
            }
            f << '\n';
        }
    }

    void MEEF_Optimizer::_save_iteration_results(
        const Eigen::MatrixXd& mask,
        const Imaging_Result& imaging_result,
        const Eigen::VectorXd& delta_d,
        const MEEFMatrixXY& meef_matrix) const
    {
        namespace fs = std::filesystem;
        fs::path iter_dir = fs::path(_config.save_file_path) / "iterations";
        fs::create_directories(iter_dir);

        auto write_matrix = [&](const fs::path& path,
                                const auto& matrix,
                                int precision = 6) {
            std::ofstream f(path);
            if (!f) {
                throw std::runtime_error(
                    "MEEF_Optimizer::_save_iteration_results: cannot open " +
                    path.string());
            }
            f << std::fixed << std::setprecision(precision);
            for (int r = 0; r < matrix.rows(); ++r) {
                for (int c = 0; c < matrix.cols(); ++c) {
                    f << matrix(r, c);
                    if (c + 1 < matrix.cols()) f << ' ';
                }
                f << '\n';
            }
        };

        write_matrix(iter_dir / "mask.txt", mask);
        write_matrix(iter_dir / "aerial.txt", imaging_result.aerial_image);
        write_matrix(iter_dir / "wafer.txt", imaging_result.wafer_image);
        write_matrix(iter_dir / "meef_mx.txt", meef_matrix.mx);
        write_matrix(iter_dir / "meef_my.txt", meef_matrix.my);
        write_matrix(iter_dir / "delta_d.txt", delta_d);
    }

    void MEEF_Optimizer::_save_history() const
    {
        namespace fs = std::filesystem;
        fs::path out = fs::path(_config.save_file_path) / "errors.csv";
        std::ofstream f(out);
        if (!f) {
            throw std::runtime_error(
                "MEEF_Optimizer::_save_history: cannot open " + out.string());
        }

        f << "iteration,pe,epe,wepe,time_seconds\n";
        f << std::fixed << std::setprecision(10);
        for (size_t i = 0; i < _iteration_history.size(); ++i) {
            f << _iteration_history[i] << ','
              << _pe_history[i] << ','
              << _epe_history[i] << ','
              << _wepe_history[i] << ','
              << _time_history[i] << '\n';
        }
    }

    void MEEF_Optimizer::_save_best_results() const
    {
        namespace fs = std::filesystem;
        fs::path out(_config.save_file_path);
        fs::create_directories(out);

        auto write_matrix = [](const fs::path& path,
                               const Eigen::MatrixXd& matrix,
                               int precision = 6) {
            std::ofstream f(path);
            if (!f) {
                throw std::runtime_error(
                    "MEEF_Optimizer::_save_best_results: cannot open " +
                    path.string());
            }
            f << std::fixed << std::setprecision(precision);
            for (int r = 0; r < matrix.rows(); ++r) {
                for (int c = 0; c < matrix.cols(); ++c) {
                    f << matrix(r, c);
                    if (c + 1 < matrix.cols()) f << ' ';
                }
                f << '\n';
            }
        };

        auto write_cps = [](const fs::path& path,
                            const ControlPoints& cps) {
            std::ofstream f(path);
            if (!f) {
                throw std::runtime_error(
                    "MEEF_Optimizer::_save_best_results: cannot open " +
                    path.string());
            }
            f << std::fixed << std::setprecision(6);
            for (size_t ci = 0; ci < cps.size(); ++ci) {
                f << "# contour " << ci
                  << " (num_points = " << cps[ci].rows() << ")\n";
                for (int r = 0; r < cps[ci].rows(); ++r) {
                    f << cps[ci](r, 0) << ' ' << cps[ci](r, 1) << '\n';
                }
                f << '\n';
            }
        };

        auto save_one = [&](const char* dirname,
                            int iteration,
                            double epe,
                            double wepe,
                            const ControlPoints& cps,
                            const Eigen::MatrixXd& mask,
                            const Imaging_Result& imaging) {
            if (iteration < 0) return;
            fs::path dir = out / dirname;
            fs::create_directories(dir);
            write_matrix(dir / "mask.txt", mask);
            write_matrix(dir / "aerial.txt", imaging.aerial_image);
            write_matrix(dir / "wafer.txt", imaging.wafer_image);
            write_cps(dir / "control_points.txt", cps);
            std::ofstream f(dir / "metrics.txt");
            f << std::setprecision(12)
              << "iteration " << iteration << '\n'
              << "epe_total " << epe << '\n'
              << "epe_mean " << epe / std::max(1, _num_eps) << '\n'
              << "wepe_total " << wepe << '\n'
              << "wepe_mean " << wepe / std::max(1, _num_weps) << '\n';
        };

        save_one("best_wepe", _best_wepe_iteration,
                 _best_wepe_epe, _best_wepe,
                 _best_wepe_cps, _best_wepe_mask, _best_wepe_imaging);
        save_one("best_epe", _best_epe_iteration,
                 _best_epe, _best_epe_wepe,
                 _best_epe_cps, _best_epe_mask, _best_epe_imaging);
    }


    MEEFMatrixXY MEEF_Optimizer::build_meef_matrix_xy(
        const ControlPoints& current_cps) const
    {
        const int num_eps = static_cast<int>(_eps_result.eps.rows());

        // 固定全局列顺序：contour-major，再按轮廓内控制点顺序。
        std::vector<std::pair<int, int>> cp_indices;
        for (int contour_idx = 0;
             contour_idx < static_cast<int>(current_cps.size());
             ++contour_idx) {
            for (int cp_idx = 0; cp_idx < current_cps[contour_idx].rows(); ++cp_idx) {
                cp_indices.emplace_back(contour_idx, cp_idx);
            }
        }
        const int num_cps = static_cast<int>(cp_indices.size());

        MEEFMatrixXY result{
            Eigen::MatrixXd::Zero(num_eps, num_cps),
            Eigen::MatrixXd::Zero(num_eps, num_cps)};
        if (num_eps == 0 || num_cps == 0) return result;

        const double delta = _config.delta;
        if (!std::isfinite(delta) || delta <= 0.0) {
            throw std::invalid_argument(
                "MEEF_Optimizer::build_meef_matrix_xy: delta must be positive");
        }
        if (_render_sraf_mask.rows() != _target_mask.rows() ||
            _render_sraf_mask.cols() != _target_mask.cols()) {
            throw std::runtime_error(
                "MEEF_Optimizer::build_meef_matrix_xy: SRAF/target dimensions mismatch");
        }

        // 每个控制点 4 个任务：x+/x-/y+/y-。行是 CP，列是 EP。
        Eigen::MatrixXd epe_x_plus  = Eigen::MatrixXd::Zero(num_cps, num_eps);
        Eigen::MatrixXd epe_x_minus = Eigen::MatrixXd::Zero(num_cps, num_eps);
        Eigen::MatrixXd epe_y_plus  = Eigen::MatrixXd::Zero(num_cps, num_eps);
        Eigen::MatrixXd epe_y_minus = Eigen::MatrixXd::Zero(num_cps, num_eps);

        const int task_count = 4 * num_cps;
        int worker_count = 1;
#ifdef _OPENMP
        worker_count = std::max(1, std::min(task_count, omp_get_max_threads()));
#endif

        // FFTW planner 不是线程安全的，因此按线程数在进入并行区前顺序构造
        // Imaging；执行阶段每个 OpenMP 线程独占一套 plan/buffer。
        std::vector<std::unique_ptr<Imaging>> imaging_workers;
        imaging_workers.reserve(worker_count);
        for (int i = 0; i < worker_count; ++i) {
            imaging_workers.push_back(std::make_unique<Imaging>(_cache));
        }

        std::atomic<bool> task_failed{false};
        std::mutex error_mutex;
        std::string first_error;

        #pragma omp parallel num_threads(worker_count)
        {
            int thread_id = 0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif
            Imaging& imaging = *imaging_workers[thread_id]; // 解引用得到 Imaging 对象
            ParametricDemo parametric(
                _config.curve_type, _target_mask, _config.msaa_level);

            #pragma omp for schedule(dynamic, 1)
            for (int task_idx = 0; task_idx < task_count; ++task_idx) {
                if (task_failed.load(std::memory_order_relaxed)) continue;

                try {
                    const int col_idx = task_idx / 4; // 列索引就是 CP 索引
                    const int variant = task_idx % 4; // 0: x+, 1: x-, 2: y+, 3: y-
                    const bool move_x = variant < 2;
                    const double signed_delta =
                        (variant % 2 == 0) ? delta : -delta;
                    const auto [contour_idx, cp_idx] = cp_indices[col_idx];

                    ControlPoints local_cps = current_cps;
                    // 工程内控制点格式是 [y, x]：真实 x 改第 1 列，真实 y 改第 0 列。
                    const int coordinate = move_x ? 1 : 0;
                    double moved = local_cps[contour_idx](cp_idx, coordinate) +
                                   signed_delta;
                    local_cps[contour_idx](cp_idx, coordinate) =
                        std::round(moved * 1e4) / 1e4;

                    Eigen::MatrixXd mask = parametric.render_curve(local_cps);
                    mask.array() += _render_sraf_mask.array();

                    Imaging_Result imaging_result = imaging.compute(
                        mask,
                        _simulator._params.resist.threshold,
                        _simulator._params.resist.alpha);
                    EpeEvaluation epe = Loss::evaluate_epe(
                        imaging_result.aerial_image,
                        _eps_result.eps,
                        _simulator._params.resist.threshold,
                        _simulator._params.system.pixel_size_nm);

                    if (move_x) {
                        if (signed_delta > 0.0) {
                            epe_x_plus.row(col_idx) = epe.epe_vector;
                        } else {
                            epe_x_minus.row(col_idx) = epe.epe_vector;
                        }
                    } else {
                        if (signed_delta > 0.0) {
                            epe_y_plus.row(col_idx) = epe.epe_vector;
                        } else {
                            epe_y_minus.row(col_idx) = epe.epe_vector;
                        }
                    }
                } catch (const std::exception& e) {
                    task_failed.store(true, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (first_error.empty()) {
                        first_error = "MEEF perturbation task " +
                                      std::to_string(task_idx) + ": " + e.what();
                    }
                } catch (...) {
                    task_failed.store(true, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (first_error.empty()) {
                        first_error = "MEEF perturbation task " +
                                      std::to_string(task_idx) +
                                      ": unknown exception";
                    }
                }
            }
        }

        if (task_failed.load(std::memory_order_relaxed)) {
            throw std::runtime_error(first_error.empty()
                                         ? "MEEF perturbation task failed"
                                         : first_error);
        }

        // 中心差分并对齐 Python 的 6 位小数结果。
        const double inv_two_delta = 1.0 / (2.0 * delta);
        auto round6 = [](double value) {
            return std::round(value * 1e6) / 1e6;
        };
        for (int col = 0; col < num_cps; ++col) {
            for (int ep = 0; ep < num_eps; ++ep) {
                result.mx(ep, col) = round6(
                    (epe_x_plus(col, ep) - epe_x_minus(col, ep)) *
                    inv_two_delta);
                result.my(ep, col) = round6(
                    (epe_y_plus(col, ep) - epe_y_minus(col, ep)) *
                    inv_two_delta);
            }
        }
        return result;
    }

    Eigen::VectorXd MEEF_Optimizer::_log_space(double start, double end, int num){
        Eigen::VectorXd out(num);
        double step = (end - start) / (num - 1);
        for (int i = 0; i < num; ++i){
            out(i) = std::pow(10.0, start + i * step);
        }
        return out;
    }

    std::pair<Eigen::VectorXd, Eigen::VectorXd> MEEF_Optimizer::_compute_L_curve(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, const Eigen::VectorXd& lambda){
        int n = A.cols();
        int m = lambda.size();
        Eigen::VectorXd  res_norms(m), x_norms(m);

        Eigen::MatrixXd ATA = A.transpose() * A;
        Eigen::VectorXd ATb = A.transpose() * b;
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(n, n);
        for( int i = 0; i < m; ++i){
            Eigen::VectorXd x = (ATA + lambda(i)*I).ldlt().solve(ATb);
            res_norms(i) = (A * x - b).norm();
            x_norms(i) = x.norm();
        }
        return std::make_pair(res_norms, x_norms);

    }
    
    Eigen::VectorXd MEEF_Optimizer::gradient(
        const Eigen::VectorXd& y,
        const Eigen::VectorXd& x) {
        int n = y.size();
        Eigen::VectorXd d(n);
        // 边界：一阶前向/后向差分
        d(0)     = (y(1) - y(0)) / (x(1) - x(0));
        d(n - 1) = (y(n - 1) - y(n - 2)) / (x(n - 1) - x(n - 2));
        // 内部：非均匀网格二阶中心差分（与 numpy.gradient 一致）
        for (int i = 1; i < n - 1; ++i) {
            double h_prev = x(i) - x(i - 1);
            double h_next = x(i + 1) - x(i);
            double a = -h_next / (h_prev * (h_prev + h_next));
            double b = (h_next - h_prev) / (h_prev * h_next);
            double c = h_prev / (h_next * (h_prev + h_next));
            d(i) = a * y(i - 1) + b * y(i) + c * y(i + 1);
        }
        return d;
    }
     
    double MEEF_Optimizer::_find_optimal_lambda(const Eigen::VectorXd& lambda, const Eigen::VectorXd& res_norms, const Eigen::VectorXd& x_norms){
        Eigen::VectorXd log_res = res_norms.array().log10();
        Eigen::VectorXd log_x   = x_norms.array().log10();

        Eigen::VectorXd d1 = gradient(log_x, log_res);
        Eigen::VectorXd d2 = gradient(d1, log_res);
        Eigen::VectorXd curvature = d2.array().abs()
                       / (1.0 + d1.array().square()).pow(1.5);

        int n = curvature.size();
        int max_idx = 1;
        for (int i = 2; i < n - 1; ++i)
            if (curvature(i) > curvature(max_idx)) max_idx = i;

        return lambda(max_idx);
    }

    double MEEF_Optimizer::_find_truelambadas(const Eigen::MatrixXd& M, const Eigen::RowVectorXd& e0){
        Eigen::VectorXd b = -e0.transpose();
        Eigen::VectorXd lambdas = _log_space(-6, 2, 200);
        auto [res_norms, x_norms] = _compute_L_curve(M, b, lambdas);
        return _find_optimal_lambda(lambdas, res_norms, x_norms);

        
    }

    Eigen::VectorXd MEEF_Optimizer::_SVD_get_delta(
        const Eigen::MatrixXd& M,
        const Eigen::RowVectorXd& e,
        double lambda,
        double energy_threshold)
    {
        const int num_cps = static_cast<int>(M.cols());
        if (M.rows() != e.size()) {
            throw std::invalid_argument(
                "MEEF_Optimizer::_SVD_get_delta: M.rows() must equal e.size()");
        }
        if (!std::isfinite(lambda) || lambda < 0.0) {
            throw std::invalid_argument(
                "MEEF_Optimizer::_SVD_get_delta: lambda must be finite and non-negative");
        }
        if (!std::isfinite(energy_threshold) ||
            energy_threshold <= 0.0 || energy_threshold > 1.0) {
            throw std::invalid_argument(
                "MEEF_Optimizer::_SVD_get_delta: energy_threshold must be in (0, 1]");
        }
        if (M.rows() == 0 || num_cps == 0) {
            return Eigen::VectorXd::Zero(num_cps);
        }

        // Thin SVD：M = U * diag(S) * V^T。
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            M, Eigen::ComputeThinU | Eigen::ComputeThinV);
        const Eigen::VectorXd& singular_values = svd.singularValues();
        const int rank_size = static_cast<int>(singular_values.size());
        if (rank_size == 0) return Eigen::VectorXd::Zero(num_cps);

        // 取使累计奇异值平方能量首次达到阈值的最小 k。
        const double total_energy = singular_values.squaredNorm();
        if (!std::isfinite(total_energy) || total_energy <= 1e-30) {
            return Eigen::VectorXd::Zero(num_cps);
        }

        int k = rank_size;
        double cumulative_energy = 0.0;
        for (int i = 0; i < rank_size; ++i) {
            cumulative_energy += singular_values(i) * singular_values(i);
            if (cumulative_energy / total_energy >= energy_threshold) {
                k = i + 1;
                break;
            }
        }

        // Tikhonov 截断 SVD：
        // delta = V_k * diag(s / (s^2 + lambda)) * U_k^T * (-e)。
        const Eigen::VectorXd rhs = -e.transpose();
        Eigen::VectorXd coefficients =
            svd.matrixU().leftCols(k).transpose() * rhs;
        for (int i = 0; i < k; ++i) {
            const double s = singular_values(i);
            const double denominator = s * s + lambda;
            coefficients(i) = denominator > 1e-30
                ? coefficients(i) * s / denominator
                : 0.0;
        }

        return svd.matrixV().leftCols(k) * coefficients;
    }

    ControlPoints MEEF_Optimizer::_update_control_points(
        const ControlPoints& cps,
        const Eigen::VectorXd& delta_x,
        const Eigen::VectorXd& delta_y){
        int num_cps = 0;
        for(const auto& contour : cps){
            num_cps += static_cast<int>(contour.rows());
        }
        if (delta_x.size() != num_cps || delta_y.size() != num_cps) {
        throw std::invalid_argument(
            "MEEF_Optimizer::_update_control_points: "
            "delta size must equal total control-point count");
        }
        ControlPoints updated = cps;
        int idx =0;
        for(auto& contour : updated){
            for (int point_idx = 0; point_idx < contour.rows(); ++point_idx){
                // 与 Python get_new_cps_xy 一致：控制点存储为 [y, x]，
                // delta_x/delta_y 分别写入第 1/0 列。
                contour(point_idx, 1) = std::round(
                    (contour(point_idx, 1) + delta_x(idx)) * 1e4) / 1e4;
                contour(point_idx, 0) = std::round(
                    (contour(point_idx, 0) + delta_y(idx)) * 1e4) / 1e4;
                ++idx;
            }
        }      
        return updated;

    }

    void MEEF_Optimizer::optimize(){
        namespace fs = std::filesystem;
        fs::create_directories(_config.save_file_path);

        const double threshold = _simulator._params.resist.threshold;
        const double alpha = _simulator._params.resist.alpha;
        const double pixel_size = _simulator._params.system.pixel_size_nm;
        const Eigen::RowVectorXd& wepe_weight = _eps_result.weight_epe;

        Imaging imaging(_cache);
        ParametricDemo parametric(
            _config.curve_type, _target_mask, _config.msaa_level);

        struct EvaluatedState {
            Eigen::MatrixXd mask;
            Imaging_Result imaging;
            double epe = 0.0;
            double wepe = 0.0;
            Eigen::RowVectorXd epe_vector;
            double pe = 0.0;
        };

        auto evaluate = [&](const Eigen::MatrixXd& mask) {
            EvaluatedState state;
            state.mask = mask;
            state.imaging = imaging.compute(mask, threshold, alpha);
            EpeEvaluation epe = Loss::evaluate_epe(
                state.imaging.aerial_image,
                _eps_result.eps,
                threshold,
                pixel_size);
            state.epe = epe.total_epe;
            state.epe_vector = std::move(epe.epe_vector);
            state.wepe = Loss::weighted_epe(
                wepe_weight, state.epe_vector);
            state.pe = Loss::pe_loss(
                state.imaging.wafer_image, _target_mask);
            return state;
        };

        auto save_matrix = [](const fs::path& path,
                              const Eigen::MatrixXd& matrix,
                              int precision = 6) {
            std::ofstream f(path);
            if (!f) {
                throw std::runtime_error(
                    "MEEF_Optimizer::optimize: cannot open " + path.string());
            }
            f << std::fixed << std::setprecision(precision);
            for (int r = 0; r < matrix.rows(); ++r) {
                for (int c = 0; c < matrix.cols(); ++c) {
                    f << matrix(r, c);
                    if (c + 1 < matrix.cols()) f << ' ';
                }
                f << '\n';
            }
        };

        // 每次 optimize() 都从干净状态开始，允许同一对象重复运行。
        _iteration_history.clear();
        _time_history.clear();
        _wepe_history.clear();
        _epe_history.clear();
        _pe_history.clear();
        _best_wepe = std::numeric_limits<double>::max();
        _best_wepe_epe = std::numeric_limits<double>::max();
        _best_wepe_iteration = -1;
        _best_wepe_cps.clear();
        _best_wepe_mask.resize(0, 0);
        _best_wepe_imaging = {};
        _best_epe = std::numeric_limits<double>::max();
        _best_epe_wepe = std::numeric_limits<double>::max();
        _best_epe_iteration = -1;
        _best_epe_cps.clear();
        _best_epe_mask.resize(0, 0);
        _best_epe_imaging = {};

        // 1. 保存并评估 LSM 基线。历史 iter=0 与 Python 一致记录 LSM。
        EvaluatedState current = evaluate(_lsm_mask);
        save_matrix(fs::path(_config.save_file_path) / "lsm_mask.txt",
                    _lsm_mask);
        save_matrix(fs::path(_config.save_file_path) / "lsm_aerial.txt",
                    current.imaging.aerial_image);
        save_matrix(fs::path(_config.save_file_path) / "lsm_wafer.txt",
                    current.imaging.wafer_image);
        save_matrix(fs::path(_config.save_file_path) / "target_mask.txt",
                    _target_mask, 0);

        _iteration_history.push_back(0);
        _time_history.push_back(0.0);
        _epe_history.push_back(current.epe / std::max(1, _num_eps));
        _wepe_history.push_back(current.wepe / std::max(1, _num_weps));
        _pe_history.push_back(current.pe);
        std::cout << "LSM: mean_wEPE=" << _wepe_history.back()
                  << " mean_EPE=" << _epe_history.back()
                  << " PE=" << current.pe << '\n';
        std::cout << "main_cps" << _num_main_cps <<  "sraf_cps" << _num_sraf_cps << '\n';
        // 2. 参数化初始状态作为第 1 轮求解的当前状态。
        ControlPoints current_cps = _main_control_points;
        _save_cp_history(current_cps, _config.save_file_path, 0);
        _save_curve_history(current_cps, 0);
        current = evaluate(_render_initial_mask);
        std::cout << "Parametric initial: mean_wEPE="
                  << current.wepe / std::max(1, _num_weps)
                  << " mean_EPE=" << current.epe / std::max(1, _num_eps)
                  << " PE=" << current.pe << '\n';

        const auto optimize_start = std::chrono::steady_clock::now();
        int below_tol_count = 0;
        int actual_iterations = 0;

        for (int iteration = 1; iteration <= _config.iter; ++iteration) {
            const auto iteration_start = std::chrono::steady_clock::now();
            std::cout << "===== iter " << iteration
                      << " move=" << _config.move_strategy << " =====\n";

            if (_config.move_strategy != "xy") {
                throw std::invalid_argument(
                    "MEEF_Optimizer::optimize currently supports move_strategy=xy only");
            }

            // 当前 CP + 当前 EPE 向量构建并求解 Mx/My。
            MEEFMatrixXY meef_matrix = build_meef_matrix_xy(current_cps);

            // MEEF 形状为 [num_eps, num_cps]：权重按 EP 行广播到每一列。
            // optimize_wepe_only=true 时，weight_epe=0 的非关键 EP 行整体归零，
            // 因而求解只响应 weight_epe=1 的 WEPE 关键点。
            auto apply_meef_weight = [&](Eigen::MatrixXd& matrix) {
                if (matrix.rows() != _eps_result.weight_meef.size() ||
                    matrix.rows() != _eps_result.weight_epe.size()) {
                    throw std::runtime_error(
                        "MEEF_Optimizer::optimize: MEEF row/EP weight size mismatch");
                }

                matrix.array().colwise() *=
                    _eps_result.weight_meef.transpose().array();
                if (_config.optimize_wepe_only) {
                    matrix.array().colwise() *=
                        _eps_result.weight_epe.transpose().array();
                }

                matrix = matrix.unaryExpr(
                    [](double value) {
                        return std::abs(value) < 1e-3 ? 0.0 : value;
                    });
            };
            apply_meef_weight(meef_matrix.mx);
            apply_meef_weight(meef_matrix.my);

            const Eigen::RowVectorXd e0 = current.epe_vector;
            const double lambda_x = _find_truelambadas(meef_matrix.mx, e0);
            Eigen::VectorXd delta_x = _SVD_get_delta(
                meef_matrix.mx, e0, lambda_x);
            const double lambda_y = _find_truelambadas(meef_matrix.my, e0);
            Eigen::VectorXd delta_y = _SVD_get_delta(
                meef_matrix.my, e0, lambda_y);
            delta_x = (delta_x.array() * 1e6).round() / 1e6;
            delta_y = (delta_y.array() * 1e6).round() / 1e6;
            Eigen::VectorXd delta_d =
                (delta_x.array().square() + delta_y.array().square()).sqrt();
            delta_d = (delta_d.array() * 1e6).round() / 1e6;

            // 先更新 CP，再渲染并评估新状态；禁止用旧 mask 评估新 CP。
            ControlPoints new_cps = _update_control_points(
                current_cps, delta_x, delta_y);
            Eigen::MatrixXd new_mask = parametric.render_curve(new_cps);
            new_mask.array() += _render_sraf_mask.array();
            EvaluatedState new_state = evaluate(new_mask);

            _save_cp_history(new_cps, _config.save_file_path, iteration);
            _save_curve_history(new_cps, iteration);
            _save_iteration_results(
                new_mask, new_state.imaging, delta_d, meef_matrix);

            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - optimize_start).count();
            _iteration_history.push_back(iteration);
            _time_history.push_back(elapsed);
            _epe_history.push_back(new_state.epe / std::max(1, _num_eps));
            _wepe_history.push_back(
                new_state.wepe / std::max(1, _num_weps));
            _pe_history.push_back(new_state.pe);

            // 最优快照必须保存 new_cps，而不是上轮 current_cps。
            if (new_state.wepe < _best_wepe) {
                _best_wepe = new_state.wepe;
                _best_wepe_epe = new_state.epe;
                _best_wepe_iteration = iteration;
                _best_wepe_cps = new_cps;
                _best_wepe_mask = new_mask;
                _best_wepe_imaging = new_state.imaging;
            }
            if (new_state.epe < _best_epe) {
                _best_epe = new_state.epe;
                _best_epe_wepe = new_state.wepe;
                _best_epe_iteration = iteration;
                _best_epe_cps = new_cps;
                _best_epe_mask = new_mask;
                _best_epe_imaging = new_state.imaging;
            }

            current_cps = std::move(new_cps);
            current = std::move(new_state);
            _main_control_points = current_cps;
            actual_iterations = iteration;

            const double step_max =
                delta_d.size() > 0 ? delta_d.cwiseAbs().maxCoeff() : 0.0;
            const double iter_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - iteration_start).count();
            std::cout << "iter " << iteration
                      << ": mean_wEPE=" << _wepe_history.back()
                      << " mean_EPE=" << _epe_history.back()
                      << " PE=" << current.pe
                      << " |dx|max=" << delta_x.cwiseAbs().maxCoeff()
                      << " |dy|max=" << delta_y.cwiseAbs().maxCoeff()
                      << " |d|max=" << step_max
                      << " time=" << iter_seconds << " s\n";

            if (_config.step_tol > 0.0) {
                if (step_max < _config.step_tol) {
                    ++below_tol_count;
                    if (below_tol_count >= std::max(1, _config.patience)) {
                        std::cout << "Converged after " << iteration
                                  << " iterations: |d|max < "
                                  << _config.step_tol << '\n';
                        break;
                    }
                } else {
                    below_tol_count = 0;
                }
            }
        }

        _save_history();
        _save_best_results();
        std::cout << "MEEF optimization finished: "
                  << actual_iterations << '/' << _config.iter
                  << " iterations, total "
                  << std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - optimize_start)
                         .count()
                  << " s\n";
    }




}