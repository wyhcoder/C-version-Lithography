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
#include <sstream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
namespace litho {

    // 功能：构造并初始化 MEEF 优化器。加载目标/LSM/SRAF，生成或导入控制点，
    //       选择 EP 点，渲染参数化初始掩模，并把初始化元数据保存到输出目录。
    MEEF_Optimizer::MEEF_Optimizer(const LithographySimulator& simulator,const ImagingCache& cache, const MEEFPipelineConfig& config):
        _simulator(simulator),
        _cache(cache),
        _config(config)
    {
        // 加载基础 LSM，并按 target 将主图形与 SRAF 分离。
        SaveTxt::load_txt(_config.ls_mask_path , _lsm_mask);
        _target_mask = _simulator._mask.data();
        if (_lsm_mask.rows() != _target_mask.rows() ||
            _lsm_mask.cols() != _target_mask.cols()) {
            throw std::invalid_argument(
                "MEEF_Optimizer: LSM mask dimensions must match target mask");
        }
        _main_sraf = _split_main_and_sraf(
            _lsm_mask, _target_mask, _config.dilate_radius);

        // Python set_meef.py 的 fitted SRAF 路径实际保存的是拟合后的完整 mask。
        // 因此这里也先相对 target 做拆分，再只保留 SRAF 灰度部分。
        if (_config.sraf_mask_mode == "fitted_txt") {
            Eigen::MatrixXd fitted_mask;
            SaveTxt::load_txt(_config.fitted_sraf_txt_path, fitted_mask);
            if (fitted_mask.rows() != _target_mask.rows() ||
                fitted_mask.cols() != _target_mask.cols()) {
                throw std::invalid_argument(
                    "MEEF_Optimizer: fitted SRAF mask dimensions must match target mask");
            }
            _main_sraf.sraf_mask = _split_main_and_sraf(
                fitted_mask, _target_mask, _config.dilate_radius).sraf_mask;
        } else if (_config.sraf_mask_mode != "lsm") {
            throw std::invalid_argument(
                "MEEF_Optimizer: unsupported sraf_mask_mode: " +
                _config.sraf_mask_mode);
        }

        EpSelect ep_select(_target_mask, _config.mid_weight, _config.other_weight);
        _eps_result = ep_select.select_eps_others(_config.interval_line, _config.interval_corner);
        if (_config.main_cp_mode == "target_interval") {
            _main_control_points = _extract_mask_control_points(
                _target_mask, config.main_cp_interval, config.main_symmetry);
        } else if (_config.main_cp_mode == "file") {
            _main_control_points = _load_control_points_txt(
                _config.main_cps_path,
                static_cast<int>(_target_mask.rows()),
                static_cast<int>(_target_mask.cols()));
        } else {
            throw std::invalid_argument(
                "MEEF_Optimizer: unsupported main_cp_mode: " +
                _config.main_cp_mode);
        }
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
        // fitted_txt 已经是拟合结果，直接固定使用，避免从轮廓再次拟合造成漂移。
        _render_sraf_mask = (_config.sraf_mask_mode == "fitted_txt")
            ? _main_sraf.sraf_mask
            : parametric_sraf.render_curve(_sraf_control_points);
        _render_main_mask = parametric_sraf.render_curve(_main_control_points);
        _render_initial_mask =
            _render_sraf_mask.array() + _render_main_mask.array();
        _save_meta();


        





    }
    // 功能：把归一化的 Eigen 灰度矩阵转换为 OpenCV 8 位单通道图像；
    //       像素值先乘 255，再限制到 [0, 255]，供轮廓和连通域算法使用。
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

    // 功能：把 OpenCV 的 (x, y) 整数点序列转换为工程统一使用的 (y, x) 点序列。
    IPoints MEEF_Optimizer::_cv_to_ipoints(const std::vector<cv::Point>& c){
        IPoints out;
        out.reserve(c.size());
        for (const auto& p : c) out.push_back({p.y, p.x});
        return out;
    }

    // 功能：将包含主图形和 SRAF 的 mask 拆成两个灰度矩阵。
    //       优先按连通域与 target 的真实重叠比例分类；完全不重叠时使用膨胀区域兜底。
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

        if (target_mask.rows() != H || target_mask.cols() != W) {
            throw std::invalid_argument(
                "MEEF_Optimizer::_split_main_and_sraf: mask dimensions mismatch");
        }

        // 与 Python extract_sraf 对齐：先按 8 邻接连通域及真实重叠比例分类。
        constexpr double fg_threshold = 1e-6;
        constexpr double overlap_ratio = 0.05;
        cv::Mat foreground(H, W, CV_8UC1);
        cv::Mat origin_bin(H, W, CV_8UC1);
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                foreground.at<uchar>(r, c) =
                    (lsm_mask(r, c) > fg_threshold) ? 255 : 0;
                origin_bin.at<uchar>(r, c) = (target_mask(r, c) > 0.0) ? 255 : 0;
            }
        }

        if (cv::countNonZero(foreground) == 0) return result;

        cv::Mat labels, stats, centroids;
        const int component_count = cv::connectedComponentsWithStats(
            foreground, labels, stats, centroids, 8, CV_32S);
        std::vector<int> overlaps(static_cast<size_t>(component_count), 0);
        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                const int label = labels.at<int>(r, c);
                if (label > 0 && origin_bin.at<uchar>(r, c) != 0) {
                    ++overlaps[static_cast<size_t>(label)];
                }
            }
        }

        std::vector<bool> is_main_component(
            static_cast<size_t>(component_count), false);
        bool matched_main = false;
        for (int label = 1; label < component_count; ++label) {
            const int area = stats.at<int>(label, cv::CC_STAT_AREA);
            const int overlap = overlaps[static_cast<size_t>(label)];
            const bool is_main = overlap > 0 &&
                overlap >= std::max(1, static_cast<int>(overlap_ratio * area));
            is_main_component[static_cast<size_t>(label)] = is_main;
            matched_main = matched_main || is_main;
        }

        for (int r = 0; r < H; ++r) {
            for (int c = 0; c < W; ++c) {
                const int label = labels.at<int>(r, c);
                if (label == 0) continue;
                if (is_main_component[static_cast<size_t>(label)]) {
                    result.main_mask(r, c) = lsm_mask(r, c);
                } else {
                    result.sraf_mask(r, c) = lsm_mask(r, c);
                }
            }
        }

        if (matched_main) return result;

        // 极端情况下 fitted mask 与 target 完全不重叠，回退到膨胀区域拆分。
        cv::Mat dilated_bin;
        if (dilate_radius == 0) {
            dilated_bin = origin_bin.clone();
        } else {
            int ksize = 2 * dilate_radius + 1;
            cv::Mat kernel = cv::Mat::zeros(ksize, ksize, CV_8UC1);
            for (int y = -dilate_radius; y <= dilate_radius; ++y) {
                for (int x = -dilate_radius; x <= dilate_radius; ++x) {
                    if (x * x + y * y <= dilate_radius * dilate_radius) {
                        kernel.at<uchar>(y + dilate_radius,
                                         x + dilate_radius) = 255;
                    }
                }
            }
            cv::dilate(origin_bin, dilated_bin, kernel);
        }

        result.main_mask.setZero();
        result.sraf_mask.setZero();
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

    // 功能：从文本文件读取多个 (y, x) 控制点轮廓，并检查格式、有限性和图像边界；
    //       空行或“# contour ...”注释用于分隔不同轮廓。
    ControlPoints MEEF_Optimizer::_load_control_points_txt(
        const std::string& path, int image_rows, int image_cols)
    {
        std::ifstream input(path);
        if (!input) {
            throw std::runtime_error(
                "MEEF_Optimizer: cannot open control-points file: " + path);
        }

        std::vector<std::vector<Eigen::Vector2d>> parsed;
        std::vector<Eigen::Vector2d> current;
        auto finish_contour = [&]() {
            if (!current.empty()) {
                parsed.push_back(std::move(current));
                current.clear();
            }
        };

        std::string line;
        int line_number = 0;
        while (std::getline(input, line)) {
            ++line_number;
            const auto first = line.find_first_not_of(" \t\r\n");
            if (first == std::string::npos) {
                finish_contour();
                continue;
            }
            if (line[first] == '#') {
                if (line.find("contour", first) != std::string::npos) {
                    finish_contour();
                }
                continue;
            }

            std::istringstream values(line);
            double y = 0.0;
            double x = 0.0;
            std::string extra;
            if (!(values >> y >> x) || (values >> extra)) {
                throw std::runtime_error(
                    "MEEF_Optimizer: expected exactly 'y x' at " + path +
                    ":" + std::to_string(line_number));
            }
            if (!std::isfinite(y) || !std::isfinite(x) ||
                y < 0.0 || y > image_rows - 1.0 ||
                x < 0.0 || x > image_cols - 1.0) {
                throw std::runtime_error(
                    "MEEF_Optimizer: invalid/out-of-bounds control point at " +
                    path + ":" + std::to_string(line_number));
            }
            current.emplace_back(y, x);
        }
        finish_contour();

        if (parsed.empty()) {
            throw std::runtime_error(
                "MEEF_Optimizer: no control-point contours in " + path);
        }

        ControlPoints result;
        result.reserve(parsed.size());
        for (size_t contour_index = 0; contour_index < parsed.size();
             ++contour_index) {
            const auto& points = parsed[contour_index];
            if (points.size() < 2) {
                throw std::runtime_error(
                    "MEEF_Optimizer: contour " +
                    std::to_string(contour_index) +
                    " has fewer than 2 points in " + path);
            }
            Contour contour(static_cast<Eigen::Index>(points.size()), 2);
            for (Eigen::Index row = 0; row < contour.rows(); ++row) {
                contour.row(row) = points[static_cast<size_t>(row)].transpose();
            }
            result.push_back(std::move(contour));
        }
        return result;
    }

    // 功能：按给定间隔从整数点序列中抽样；skip 模式使用 k+1 作为步长，
    //       其他模式直接使用 k，并保证实际步长至少为 1。
    IPoints MEEF_Optimizer::_sample_elements(const IPoints &pts, int k, const std::string& mode){
        int step = std::max(1, (mode == "skip" ? k+1 : k));
        IPoints out;
        for (int i = 0; i < pts.size(); i += step) {
            out.push_back(pts[i]);
        }
        return out;
    }

    // 功能：把 vector 形式的整数 (y, x) 点序列转换为 Eigen 的 N×2 浮点轮廓矩阵。
    Contour MEEF_Optimizer::_ipoints_to_contour(const IPoints &pts){
        int n = static_cast<int>(pts.size());
        Contour m(n,2);
        for(int i = 0; i < n; ++i){
            m(i, 0) = static_cast<double>(pts[i][0]);
            m(i, 1) = static_cast<double>(pts[i][1]);
        }
        return m;
    }

    // 功能：从二值/灰度主图形 mask 提取外轮廓并间隔采样控制点；
    //       配置对称模式时，用第一条轮廓生成对应的镜像轮廓。
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

    // 功能：从 SRAF mask 中过滤过小连通域、提取外轮廓并采样控制点；
    //       当采样点过少时自动减小步长，以尽量满足每块 SRAF 的最少控制点数。
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

    // 功能：把优化器初始化后的主图形控制点、SRAF 控制点、EP 点以及渲染后的
    //       主图形/SRAF mask 保存到配置的输出目录，便于检查和复现实验。
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

    // 功能：把指定迭代轮次的原始控制点保存到 curves/cp_history；
    //       写入失败时只输出警告，不中断正在进行的优化。
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

    // 功能：将指定轮次的控制点展开为实际参数化曲线采样点，保存到
    //       curves/bspline_curves，用于观察控制点更新后曲线如何变化。
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

    // 功能：保存当前迭代的 mask、空中像、晶圆像、控制点位移大小以及 X/Y MEEF 矩阵；
    //       后一轮会覆盖 iterations 目录中的这些“当前结果”文件。
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

    // 功能：把各轮的迭代编号、PE、EPE、加权 EPE 和累计时间汇总写入 errors.csv。
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

    // 功能：分别保存“加权 EPE 最小”和“普通 EPE 最小”时的控制点、mask、
    //       空中像、晶圆像及指标，形成 best_wepe 和 best_epe 两组快照。
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
    // 功能：对每个主图形控制点执行 x+/x-/y+/y- 四种扰动和成像，
    //       用中心差分构建行对应 EP、列对应控制点的 X/Y MEEF 矩阵。
    // 计算公式：
    //   Mx(ep, cp) = [EPE(x + delta) - EPE(x - delta)] / (2 * delta)
    //   My(ep, cp) = [EPE(y + delta) - EPE(y - delta)] / (2 * delta)
    // 每个控制点需要执行 x+、x-、y+、y- 四次独立成像，使用 OpenMP 并行计算。
    MEEFMatrixXY MEEF_Optimizer::build_meef_matrix_xy(
        const ControlPoints& current_cps) const
    {
        const int num_eps = static_cast<int>(_eps_result.eps.rows());

        // 将“多个轮廓中的控制点”展平为一维索引。
        // cp_indices[全局CP编号] = {轮廓编号, 轮廓内CP编号}。
        std::vector<std::pair<int, int>> cp_indices;
        for (int contour_idx = 0;
             contour_idx < static_cast<int>(current_cps.size());
             ++contour_idx) {
            for (int cp_idx = 0; cp_idx < current_cps[contour_idx].rows(); ++cp_idx) {
                cp_indices.emplace_back(contour_idx, cp_idx);
            }
        }
        const int num_cps = static_cast<int>(cp_indices.size());

        // 最终矩阵：行对应 EP，列对应控制点。
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

        // 暂存四种扰动产生的 EPE。这里为了便于按任务写入，行是 CP、列是 EP；
        // 最后计算 result 时再转换成“行是 EP、列是 CP”的 MEEF 排列。
        Eigen::MatrixXd epe_x_plus  = Eigen::MatrixXd::Zero(num_cps, num_eps);
        Eigen::MatrixXd epe_x_minus = Eigen::MatrixXd::Zero(num_cps, num_eps);
        Eigen::MatrixXd epe_y_plus  = Eigen::MatrixXd::Zero(num_cps, num_eps);
        Eigen::MatrixXd epe_y_minus = Eigen::MatrixXd::Zero(num_cps, num_eps);

        // 每个控制点拆成 4 个任务，因此总任务数为 4 * num_cps。
        int worker_count = 1;
        const int task_count = 4 * num_cps;
#ifdef _OPENMP
        // 线程数不超过任务数，也不超过 OpenMP 允许的最大线程数。
        worker_count = std::max(1, std::min(task_count, omp_get_max_threads()));
#endif

        // 每个线程准备一个独立 Imaging 对象。
        // FFTW 创建 plan 的过程不是线程安全的，所以必须在进入并行区之前串行构造；
        // 并行执行时，每个线程只使用自己的 plan 和缓冲区，避免数据竞争。
        std::vector<std::unique_ptr<Imaging>> imaging_workers;
        imaging_workers.reserve(worker_count);
        for (int i = 0; i < worker_count; ++i) {
            imaging_workers.push_back(std::make_unique<Imaging>(_cache));
        }

        // OpenMP 循环中的异常不能直接安全地传播到并行区外，因此：
        // 1. 原子变量通知其他线程停止新任务；
        // 2. 互斥锁保护 first_error，只保存第一个错误信息；
        // 3. 离开并行区后由主线程统一抛出异常。
        std::atomic<bool> task_failed{false};
        std::mutex error_mutex;
        std::string first_error;

        // 创建 worker_count 个线程，花括号内的代码每个线程都会执行一次。
        #pragma omp parallel num_threads(worker_count)
        {
            // 非 OpenMP 构建保持 thread_id=0，整个循环自动退化为单线程。
            int thread_id = 0;
#ifdef _OPENMP
            thread_id = omp_get_thread_num();
#endif

            // 线程私有资源：每个线程独占一个 Imaging 和一个曲线渲染器。
            // imaging_workers 本身是共享容器，但不同线程访问不同元素。
            Imaging& imaging = *imaging_workers[thread_id];
            ParametricDemo parametric(_config.curve_type, _target_mask, _config.msaa_level);

            // 动态调度：线程每次领取 1 个扰动任务；先完成的线程继续领取任务，
            // 可降低不同曲线渲染/成像耗时造成的线程等待。
            #pragma omp for schedule(dynamic, 1)
            for (int task_idx = 0; task_idx < task_count; ++task_idx) {
                // 某个线程失败后，其余线程跳过尚未开始的任务。
                if (task_failed.load(std::memory_order_relaxed)) continue;

                try {
                    // 将线性任务编号还原为“控制点编号 + 扰动方向”：
                    // task 0~3 属于 CP0，task 4~7 属于 CP1，以此类推。
                    const int col_idx = task_idx / 4;
                    const int variant = task_idx % 4;
                    // variant: 0=x+, 1=x-, 2=y+, 3=y-。
                    const bool move_x = variant < 2;
                    const double signed_delta =
                        (variant % 2 == 0) ? delta : -delta;
                    const auto [contour_idx, cp_idx] = cp_indices[col_idx];

                    // 每个任务复制一份控制点，保证线程之间不会修改同一份数据。
                    ControlPoints local_cps = current_cps;
                    // 工程内控制点格式是 [y, x]：真实 x 改第 1 列，真实 y 改第 0 列。
                    const int coordinate = move_x ? 1 : 0;
                    double moved = local_cps[contour_idx](cp_idx, coordinate) +
                                   signed_delta;
                    local_cps[contour_idx](cp_idx, coordinate) =
                        std::round(moved * 1e4) / 1e4;

                    // 用扰动后的控制点重新生成主图形，再叠加固定不动的 SRAF。
                    Eigen::MatrixXd mask = parametric.render_curve(local_cps);
                    mask.array() += _render_sraf_mask.array();

                    // 对扰动掩模执行成像，并得到所有 EP 的误差向量。
                    Imaging_Result imaging_result = imaging.compute(
                        mask,
                        _simulator._params.resist.threshold,
                        _simulator._params.resist.alpha);
                    EpeEvaluation epe = Loss::evaluate_epe(
                        imaging_result.aerial_image,
                        _eps_result.eps,
                        _simulator._params.resist.threshold,
                        _simulator._params.system.pixel_size_nm);

                    // 每个任务只写一个确定的矩阵行，其他线程写不同的行/方向，
                    // 因此不需要互斥锁。四个临时矩阵分别保存四种扰动结果。
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
                    // 通知其他线程停止，并在锁保护下记录第一个异常。
                    task_failed.store(true, std::memory_order_relaxed);
                    std::lock_guard<std::mutex> lock(error_mutex);
                    if (first_error.empty()) {
                        first_error = "MEEF perturbation task " +
                                      std::to_string(task_idx) + ": " + e.what();
                    }
                } catch (...) {
                    // 捕获非 std::exception 类型的未知异常。
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

        // 所有线程在并行区末尾同步；若有任务失败，现在由主线程抛出异常。
        if (task_failed.load(std::memory_order_relaxed)) {
            throw std::runtime_error(first_error.empty()
                                         ? "MEEF perturbation task failed"
                                         : first_error);
        }

        // 匿名函数（Lambda）：将结果四舍五入到 6 位小数，与 Python 结果对齐。
        const double inv_two_delta = 1.0 / (2.0 * delta);
        auto round6 = [](double value) {
            return std::round(value * 1e6) / 1e6;
        };

        // 用四组 EPE 结果计算中心差分，得到最终 Mx 和 My。
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

    // 功能：在 10^start 到 10^end 之间生成 num 个等对数间隔的正数，
    //       用作 L-Curve 扫描的正则化参数候选值。
    Eigen::VectorXd MEEF_Optimizer::_log_space(double start, double end, int num){
        Eigen::VectorXd out(num);
        double step = (end - start) / (num - 1);
        for (int i = 0; i < num; ++i){
            out(i) = std::pow(10.0, start + i * step);
        }
        return out;
    }

    // 功能：对每个候选 lambda 求解 Tikhonov 正则化方程，返回对应的
    //       残差范数 ||Ax-b|| 和解范数 ||x||，作为 L-Curve 的两条坐标。
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
    // 功能：计算 y 相对于非均匀坐标 x 的数值梯度；边界使用单边差分，
    //       内部使用与 numpy.gradient 对齐的非均匀网格中心差分。
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
    // 功能：在对数 L-Curve 上估算离散曲率，并返回内部曲率最大点对应的 lambda。
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

    // 功能：为当前 MEEF 线性系统生成 lambda 候选、计算 L-Curve，
    //       并自动选出用于位移求解的正则化参数。
    double MEEF_Optimizer::_find_truelambadas(const Eigen::MatrixXd& M, const Eigen::RowVectorXd& e0){
        Eigen::VectorXd b = -e0.transpose();
        Eigen::VectorXd lambdas = _log_space(-6, 2, 200);
        auto [res_norms, x_norms] = _compute_L_curve(M, b, lambdas);
        return _find_optimal_lambda(lambdas, res_norms, x_norms);

        
    }

    // 功能：对 MEEF 矩阵执行薄 SVD，按奇异值能量阈值截断不可辨识方向，
    //       再结合 Tikhonov 滤波求解使 EPE 减小的控制点位移向量。
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

    // 功能：把展平的 x/y 位移依次加到多轮廓控制点；保持 (y, x) 存储顺序，
    //       并将更新后的坐标舍入到小数点后 4 位。
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

    // 功能：执行完整 MEEF 优化流程。先评估 LSM 和参数化初始状态，再逐轮构建
    //       MEEF 矩阵、求控制点位移、重新成像和记录误差，最后保存历史及最优结果。
    void MEEF_Optimizer::optimize(){
        namespace fs = std::filesystem;
        // 确保结果保存目录存在。
        fs::create_directories(_config.save_file_path);

        std::cout << "\n"
                  << "==================== MEEF Optimization ====================\n"
                  << "  Output directory : " << _config.save_file_path << '\n'
                  << "  Move strategy    : " << _config.move_strategy << '\n'
                  << "  Max iterations   : " << _config.iter << '\n'
                  << "===========================================================\n";

        const double threshold = _simulator._params.resist.threshold;
        const double alpha = _simulator._params.resist.alpha;
        const double pixel_size = _simulator._params.system.pixel_size_nm;
        const Eigen::RowVectorXd& wepe_weight = _eps_result.weight_epe;

        Imaging imaging(_cache);
        ParametricDemo parametric(
            _config.curve_type, _target_mask, _config.msaa_level);

        // 保存一次掩模评估产生的图像和误差指标。
        struct EvaluatedState {
            Eigen::MatrixXd mask;
            Imaging_Result imaging;
            double epe = 0.0;
            double wepe = 0.0;
            Eigen::RowVectorXd epe_vector;
            double pe = 0.0;
        };

        // 匿名函数（Lambda）：对掩模成像并计算各项误差。
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

        // 匿名函数（Lambda）：将矩阵按文本格式保存到指定路径。
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

        // 匿名函数（Lambda）：按统一列宽输出一组误差指标。
        auto print_metrics = [](const char* label,
                                double mean_wepe,
                                double mean_epe,
                                double pe) {
            std::ostringstream line;
            line << "  " << std::left << std::setw(18) << label
                 << std::right << std::fixed << std::setprecision(6)
                 << " | mean_wEPE: " << std::setw(12) << mean_wepe
                 << " | mean_EPE: " << std::setw(12) << mean_epe
                 << " | PE: " << std::setw(12) << pe;
            std::cout << line.str() << '\n';
        };

        // 清空上一次运行产生的历史和最优结果。
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

        // 1. 评估并保存 LSM 基线结果。
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

        std::cout << "\n[Baseline]\n";
        print_metrics("LSM", _wepe_history.back(),
                      _epe_history.back(), current.pe);
        {
            std::ostringstream line;
            line << "  " << std::left << std::setw(18) << "Control points"
                 << std::right
                 << " | main: " << std::setw(8) << _num_main_cps
                 << " | SRAF: " << std::setw(8) << _num_sraf_cps;
            std::cout << line.str() << '\n';
        }

        // 2. 将参数化初始掩模作为迭代起点。
        ControlPoints current_cps = _main_control_points;
        _save_cp_history(current_cps, _config.save_file_path, 0);
        _save_curve_history(current_cps, 0);
        current = evaluate(_render_initial_mask);

        std::cout << "\n[Initial State]\n";
        print_metrics("Parametric",
                      current.wepe / std::max(1, _num_weps),
                      current.epe / std::max(1, _num_eps),
                      current.pe);

        const auto optimize_start = std::chrono::steady_clock::now();
        int below_tol_count = 0;
        int actual_iterations = 0;

        // 3. 迭代构建 MEEF 矩阵并更新控制点。
        for (int iteration = 1; iteration <= _config.iter; ++iteration) {
            const auto iteration_start = std::chrono::steady_clock::now();
            std::cout << "\n[Iteration " << iteration << '/' << _config.iter
                      << "] move=" << _config.move_strategy << '\n';

            if (_config.move_strategy != "xy") {
                throw std::invalid_argument(
                    "MEEF_Optimizer::optimize currently supports move_strategy=xy only");
            }

            // 根据当前控制点构建 x、y 方向的 MEEF 矩阵。
            MEEFMatrixXY meef_matrix = build_meef_matrix_xy(current_cps);

            // 匿名函数（Lambda）：按 EP 权重筛选并缩放 MEEF 矩阵。
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

                // 内层匿名函数：将绝对值过小的元素置零。
                matrix = matrix.unaryExpr(
                    [](double value) {
                        return std::abs(value) < 1e-3 ? 0.0 : value;
                    });
            };
            apply_meef_weight(meef_matrix.mx);
            apply_meef_weight(meef_matrix.my);

            // 使用 SVD 求解每个控制点的 x、y 位移。
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

            // 更新控制点，重新渲染掩模并评估新状态。
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

            // 分别记录加权 EPE 和普通 EPE 的最优结果。
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

            // 将本轮结果作为下一轮的输入。
            current_cps = std::move(new_cps);
            current = std::move(new_state);
            _main_control_points = current_cps;
            actual_iterations = iteration;

            const double step_max =
                delta_d.size() > 0 ? delta_d.cwiseAbs().maxCoeff() : 0.0;
            const double iter_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - iteration_start).count();

            print_metrics("Result", _wepe_history.back(),
                          _epe_history.back(), current.pe);
            {
                std::ostringstream line;
                line << std::fixed << std::setprecision(6)
                     << "  " << std::left << std::setw(18) << "Step"
                     << std::right
                     << " | |dx|max: " << std::setw(12)
                     << delta_x.cwiseAbs().maxCoeff()
                     << " | |dy|max: " << std::setw(12)
                     << delta_y.cwiseAbs().maxCoeff()
                     << " | |d|max: " << std::setw(12) << step_max;
                std::cout << line.str() << '\n';
            }
            {
                std::ostringstream line;
                line << std::fixed << std::setprecision(3)
                     << "  " << std::left << std::setw(18) << "Time"
                     << std::right
                     << " | iteration: " << std::setw(10) << iter_seconds << " s"
                     << " | total: " << std::setw(10) << elapsed << " s";
                std::cout << line.str() << '\n';
            }

            // 连续若干轮位移小于阈值时提前停止。
            if (_config.step_tol > 0.0) {
                if (step_max < _config.step_tol) {
                    ++below_tol_count;
                    if (below_tol_count >= std::max(1, _config.patience)) {
                        std::cout << "  Status             | converged: |d|max < "
                                  << _config.step_tol << " for "
                                  << below_tol_count << " iteration(s)\n";
                        break;
                    }
                } else {
                    below_tol_count = 0;
                }
            }
        }

        // 4. 保存误差历史和最优结果。
        _save_history();
        _save_best_results();
        const double total_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - optimize_start).count();
        std::ostringstream summary;
        summary << "\n"
                << "==================== Optimization Summary =================\n"
                << "  Iterations       : " << actual_iterations << '/'
                << _config.iter << '\n'
                << "  Total time       : " << std::fixed << std::setprecision(3)
                << total_seconds << " s\n"
                << "  Results saved to : " << _config.save_file_path << '\n'
                << "===========================================================\n";
        std::cout << summary.str();
    }




}
