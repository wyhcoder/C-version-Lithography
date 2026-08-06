#pragma once

#include <Eigen/Dense>
#include <optional>
#include <string>

namespace litho {

class Mask {
public:
    // 从图片文件构造（归一化到 [0,1]）
    static std::optional<Mask> from_file(const std::string& path, int size,
                                         bool normalize = true);

    Mask(const std::string& path, int size, bool normalize = true);

    // 从矩阵直接构造（测试用，不依赖图片文件）
    Mask(Eigen::MatrixXd data);

    Eigen::MatrixXd& data() { return _data; }
    int  row_data_size() const { return _row_data_size; }
    int rows() const { return _data.rows(); }
    int cols() const { return _data.cols(); }

private:
    Eigen::MatrixXd _data;
    int _row_data_size;
};

}  // namespace litho
