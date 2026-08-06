#include "save_txt.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace litho{

std::string SaveTxt::_result_path(){
    std::filesystem::path src(__FILE__);
    std::filesystem::path root = src.parent_path().parent_path().parent_path();
    std::filesystem::path result = root / "result";
    if (!std::filesystem::exists(result)){
        std::filesystem::create_directory(result);
    }
    return result.string();
}

void SaveTxt::save_mat(const Eigen::MatrixXd& M, const std::string& fname, int precision){
    std::string full = _result_path() + "/" + fname;
    std::ofstream f(full);
    f << std::fixed << std::setprecision(precision);
    f << "# rows=" << M.rows() << " cols=" << M.cols()<< "\n";
    for (int r = 0; r < M.rows(); ++r){
        for (int c = 0; c < M.cols(); ++c){
            f << M(r,c);
            if (c < M.cols()-1) f << " ";
        }
        f << "\n";
    }
    // std::cout << "  saved: " << full << "\n";

}

void SaveTxt::save_mat_int(const Eigen::MatrixXd& M, const std::string& fname) {
    std::string full = _result_path() + "/" + fname;
    std::ofstream f(full);
    f << "# rows=" << M.rows() << " cols=" << M.cols() << "\n";
    for (int r = 0; r < M.rows(); ++r) {
        for (int c = 0; c < M.cols(); ++c) {
            f << static_cast<int>(std::round(M(r,c)));   // 四舍五入
            if (c < M.cols() - 1) f << " ";
        }
        f << "\n";
    }
    // std::cout << "  saved: " << full << "\n";
}

// 从文本文件读取矩阵到 M。
//   - 忽略以 '#' 开头的注释行
//   - 空行跳过
//   - 每行的列数必须一致；总行/列由文件内容推断
//   - 支持绝对路径；若给出相对文件名（不含 '/'），自动到 result 目录下找
void SaveTxt::load_txt(const std::string& file_path, Eigen::MatrixXd& M) {
    std::string full = file_path;
    if (full.find('/') == std::string::npos) {
        full = _result_path() + "/" + full;
    }

    std::ifstream f(full);
    if (!f) throw std::runtime_error("load_txt: cannot open " + full);

    std::vector<std::vector<double>> rows;
    std::string line;
    size_t cols = 0;

    while (std::getline(f, line)) {
        // 跳过空行和 '#' 注释
        size_t p = line.find_first_not_of(" \t\r\n");
        if (p == std::string::npos) continue;
        if (line[p] == '#') continue;

        std::istringstream iss(line);
        std::vector<double> vals;
        double v;
        while (iss >> v) vals.push_back(v);
        if (vals.empty()) continue;

        if (cols == 0) {
            cols = vals.size();
        } else if (vals.size() != cols) {
            throw std::runtime_error(
                "load_txt: inconsistent column count at row "
                + std::to_string(rows.size() + 1) + " in " + full);
        }
        rows.push_back(std::move(vals));
    }

    if (rows.empty())
        throw std::runtime_error("load_txt: empty matrix in " + full);

    M.resize(static_cast<Eigen::Index>(rows.size()),
             static_cast<Eigen::Index>(cols));
    for (size_t r = 0; r < rows.size(); ++r)
        for (size_t c = 0; c < cols; ++c)
            M(static_cast<Eigen::Index>(r), static_cast<Eigen::Index>(c))
                = rows[r][c];

    std::cout << "  loaded: " << full
              << "  (" << M.rows() << "x" << M.cols() << ")\n";
}

}