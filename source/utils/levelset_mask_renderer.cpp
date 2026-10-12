#include "levelset_mask_renderer.h"

#include <opencv2/imgproc.hpp>

namespace litho {

Eigen::MatrixXd render_levelset_mask(const Eigen::MatrixXd& phi, LevelSetMaskType type, int scale) {
    if (type == LevelSetMaskType::Binary) return (phi.array() >= 0.0).cast<double>().matrix();

    // Eigen 默认按列存储，逐像素写入 OpenCV 的按行矩阵；float32 与 Python renderer 的输入精度一致。
    const int height = phi.rows(), width = phi.cols();
    cv::Mat local_phi(height, width, CV_32FC1);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) local_phi.at<float>(y, x) = static_cast<float>(phi(y, x));
    }
    cv::Mat highres_phi;
    // INTER_LINEAR 使用像素中心对齐的双线性插值，越过边缘的采样沿用最边缘像素，与 cv2.resize 相同。
    cv::resize(local_phi, highres_phi, cv::Size(width * scale, height * scale), 0.0, 0.0, cv::INTER_LINEAR);
    Eigen::MatrixXd mask(height, width);
    const double subpixel_count = static_cast<double>(scale) * scale;
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            int inside = 0;
            // 原局部像素 (y,x) 对应高分辨率半开区间 [y*scale,(y+1)*scale) × [x*scale,(x+1)*scale)。
            for (int sy = 0; sy < scale; ++sy) {
                const float* row = highres_phi.ptr<float>(y * scale + sy);
                for (int sx = 0; sx < scale; ++sx) inside += row[x * scale + sx] >= 0.0f;
            }
            mask(y, x) = inside / subpixel_count;
        }
    }
    return mask;
}

} // namespace litho
