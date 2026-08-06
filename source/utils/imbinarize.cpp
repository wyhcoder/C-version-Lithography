#include "imbinarize.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

namespace litho {

    Eigen::MatrixXd Imbinarize::binarize(Eigen::MatrixXd &gray_mask, int block_size, double C){
        cv::Mat gray, bw;
        cv::eigen2cv(gray_mask, gray);
        gray.convertTo(gray, CV_8U, 255.0);
        cv::adaptiveThreshold(gray, bw, 255, 
            cv::ADAPTIVE_THRESH_GAUSSIAN_C,
            cv::THRESH_BINARY, block_size, C);

        Eigen::MatrixXd result;
        cv::cv2eigen(bw, result);
        return result/255.0;



    }

    Eigen::MatrixXd Imbinarize::binarize_OTSU(Eigen::MatrixXd &gray_mask){
        cv::Mat gray, bw;
        cv::eigen2cv(gray_mask, gray);
        gray.convertTo(gray, CV_8U, 255.0);
        cv::threshold(gray, bw, 0, 255,
                  cv::THRESH_BINARY | cv::THRESH_OTSU);
        Eigen::MatrixXd result;
        cv::cv2eigen(bw, result);
        return result / 255.0;   // 归一化回 [0,1]
    }

    // ── helper：把 0/1 的 uint8 cv::Mat 转成 Eigen::MatrixXd ──
    static Eigen::MatrixXd _u8_to_eigen01(const cv::Mat& m) {
        Eigen::MatrixXd out;
        cv::Mat tmp;
        m.convertTo(tmp, CV_64F, 1.0);        // 0/1 -> 0.0/1.0
        cv::cv2eigen(tmp, out);
        return out;
    }

    // ── helper：np.quantile(vals, q)，线性插值版 ──
    static double _quantile(std::vector<double> vals, double q) {
        if (vals.empty()) return 0.0;
        q = std::clamp(q, 0.0, 1.0);
        const size_t n = vals.size();
        const double pos = q * (n - 1);
        const size_t lo  = static_cast<size_t>(std::floor(pos));
        const size_t hi  = static_cast<size_t>(std::ceil(pos));
        std::nth_element(vals.begin(), vals.begin() + lo, vals.end());
        double v_lo = vals[lo];
        if (lo == hi) return v_lo;
        std::nth_element(vals.begin() + lo + 1, vals.begin() + hi, vals.end());
        double v_hi = vals[hi];
        return v_lo + (v_hi - v_lo) * (pos - lo);
    }

    // ── helper：peak_local_max 的 OpenCV 版 ──
    //   等价 skimage.feature.peak_local_max(img, min_distance=1, threshold_abs=t)
    //   min_distance=1 → 3×3 邻域内是最大值即算峰
    static std::vector<double> _peak_values(const cv::Mat& img64,
                                            double threshold_abs) {
        cv::Mat dilated;
        cv::dilate(img64, dilated,
                   cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)));
        cv::Mat is_peak = (img64 >= dilated) & (img64 > threshold_abs);
        std::vector<double> vals;
        vals.reserve(1024);
        for (int r = 0; r < img64.rows; ++r) {
            const uchar*  m = is_peak.ptr<uchar>(r);
            const double* p = img64.ptr<double>(r);
            for (int c = 0; c < img64.cols; ++c)
                if (m[c]) vals.push_back(p[c]);
        }
        return vals;
    }

    BinarizeResult Imbinarize::binarize_advanced(
        const Eigen::MatrixXd& gray_mask,
        const Eigen::MatrixXd& target_pattern,
        double peak_min_intensity,
        double quantile_val,
        int    area_threshold,
        int    min_spacing_pixels)
    {
        std::cout << "\n--- Starting Mask Binarization Process ---\n";

        // Eigen → cv::Mat(CV_64F)
        cv::Mat gray64, tgt64;
        {
            Eigen::MatrixXd gm = gray_mask;         // eigen2cv 要非 const
            Eigen::MatrixXd tp = target_pattern;
            cv::eigen2cv(gm, gray64);
            cv::eigen2cv(tp, tgt64);
        }
        const int H = gray64.rows;
        const int W = gray64.cols;

        BinarizeResult R;
        auto empty01 = [&]{ return Eigen::MatrixXd::Zero(H, W); };

        // ── 阶段 1 & 2: 自适应二值化 + 伪影滤除 ────────────────
        std::cout << "Step 1 & 2: Adaptive Binarization and Artifact Filtering...\n";

        std::vector<double> peak_vals = _peak_values(gray64, peak_min_intensity);
        if (peak_vals.empty()) {
            std::cout << "Warning: No local peaks found. Returning empty masks.\n";
            R.initial_binary_mask  = empty01();
            R.filtered_binary_mask = empty01();
            R.sraf_mask_initial    = empty01();
            R.sraf_mask_final      = empty01();
            R.final_mask           = empty01();
            R.keep_out_zone        = empty01();
            return R;
        }

        const double adaptive_threshold = _quantile(peak_vals, quantile_val);

        // initial_binary_mask = gray >= adaptive_threshold  (uint8: 0/1)
        cv::Mat initial_bin;
        cv::compare(gray64, adaptive_threshold, initial_bin, cv::CMP_GE);
        initial_bin /= 255;                            // → 0/1

        // 连通域标记 + 按面积过滤
        cv::Mat labels, stats, centroids;
        int n_labels = cv::connectedComponentsWithStats(
            initial_bin, labels, stats, centroids, 8, CV_32S);

        cv::Mat filtered_bin = initial_bin.clone();
        for (int lb = 1; lb < n_labels; ++lb) {           // 0 是背景
            int area = stats.at<int>(lb, cv::CC_STAT_AREA);
            if (area < area_threshold) {
                filtered_bin.setTo(0, labels == lb);
            }
        }

        // ── 阶段 3: 初步 SRAF = filtered ∩ ¬target ─────────────
        cv::Mat tgt_bin;
        cv::compare(tgt64, 0.5, tgt_bin, cv::CMP_GT);     // uint8 0/255
        tgt_bin /= 255;                                    // → 0/1

        cv::Mat not_tgt = 1 - tgt_bin;
        cv::Mat sraf_initial = filtered_bin.mul(not_tgt); // 0/1

        // ── 阶段 4: 强制 SRAF 间距 ─────────────────────────────
        std::cout << "Step 3: Enforcing " << min_spacing_pixels
                  << " pixel minimum spacing for SRAFs...\n";

        cv::Mat sraf_final = sraf_initial.clone();
        cv::Mat keep_out   = cv::Mat::zeros(H, W, CV_8U);   // 默认全 0
        if (min_spacing_pixels > 0) {
            // disk(radius) 结构元
            int ksize = 2 * min_spacing_pixels + 1;
            cv::Mat kernel = cv::getStructuringElement(
                cv::MORPH_ELLIPSE, cv::Size(ksize, ksize));

            cv::dilate(tgt_bin, keep_out, kernel);        // 膨胀 target
            cv::Mat allow = 1 - keep_out;
            sraf_final = sraf_initial.mul(allow);
        }

        // final = target ∪ sraf_final
        cv::Mat final_bin;
        cv::bitwise_or(tgt_bin, sraf_final, final_bin);

        // ── 阶段 5: 对最终 mask 再做一次连通域过滤 ─────────────
        // 防止 target 区域或 SRAF 合并后产生的小孤立块
        cv::Mat final_labels, final_stats, final_centroids;
        int n_final = cv::connectedComponentsWithStats(
            final_bin, final_labels, final_stats, final_centroids, 8, CV_32S);
        for (int lb = 1; lb < n_final; ++lb) {
            int area = final_stats.at<int>(lb, cv::CC_STAT_AREA);
            if (area < area_threshold) {
                final_bin.setTo(0, final_labels == lb);
            }
        }

        // cv::Mat(0/1, uint8) → Eigen::MatrixXd
        R.initial_binary_mask  = _u8_to_eigen01(initial_bin);
        R.filtered_binary_mask = _u8_to_eigen01(filtered_bin);
        R.sraf_mask_initial    = _u8_to_eigen01(sraf_initial);
        R.sraf_mask_final      = _u8_to_eigen01(sraf_final);
        R.final_mask           = _u8_to_eigen01(final_bin);
        R.keep_out_zone        = _u8_to_eigen01(keep_out);
        return R;
    }

}