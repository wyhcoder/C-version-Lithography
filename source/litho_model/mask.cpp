#include "mask.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <stdexcept>

namespace litho {

Mask::Mask(const std::string& path, int size, bool normalize) {
    int src_w = 0, src_h = 0, ch = 0;
    unsigned char* src = stbi_load(path.c_str(), &src_w, &src_h, &ch, 1);
    if (!src) {
        throw std::runtime_error("Failed to load mask: " + path);
    }

    int ox = (size - src_w) / 2; // 左边留白加原图宽加右边留白 = size
    int oy = (size - src_h) / 2; // 上边留白加原图高加下边留白 = size


    _row_data_size = src_w;
    _data = Eigen::MatrixXd::Zero(size, size);


    const int copy_w = std::min(src_w, size - ox);
    const int copy_h = std::min(src_h, size - oy);
    const double scale = normalize ? (1.0 / 255.0) : 1.0;

    for (int r = 0; r < copy_h; ++r)
        for (int c = 0; c < copy_w; ++c) // 从一维的数据中取数据
            _data(oy + r, ox + c) = static_cast<double>(src[r * src_w + c]) * scale;

    stbi_image_free(src);
}

std::optional<Mask> Mask::from_file(const std::string& path, int size, bool normalize) {
    try {
        return Mask(path, size, normalize);
    } catch (...) {
        return std::nullopt;
    }
}

// 从矩阵直接构造（测试用）
Mask::Mask(Eigen::MatrixXd data) : _data(std::move(data)) {}

}  // namespace litho
