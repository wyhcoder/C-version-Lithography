#pragma once

#include <Eigen/Dense>


namespace litho {


class SaveTxt {
public:

    static  std::string _result_path();

    static void save_mat(const Eigen::MatrixXd& M, const std::string& fname, int precision = 4);
    
    static void save_mat_int(const Eigen::MatrixXd& M, const std::string& fname);

    static void load_txt(const std::string& file_path, Eigen::MatrixXd& M);
   
};

}