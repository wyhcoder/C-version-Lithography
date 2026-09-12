#pragma once
#include <Eigen/Dense>


namespace litho {

    // binarize_advanced 的完整输出，对应 Python 五元组
    //   initial_binary_mask   : 自适应阈值后的原始二值图
    //   filtered_binary_mask  : 去掉小连通域后的二值图
    //   sraf_mask_initial     : filtered ∩ ¬target  （初步 SRAF）
    //   sraf_mask_final       : 强制间距后的 SRAF   （远离 target）
    //   final_mask            : (target > 0.5) ∪ sraf_final
    struct BinarizeResult {
        Eigen::MatrixXd initial_binary_mask;
        Eigen::MatrixXd filtered_binary_mask;
        Eigen::MatrixXd sraf_mask_initial;
        Eigen::MatrixXd sraf_mask_final;
        Eigen::MatrixXd final_mask;
        Eigen::MatrixXd keep_out_zone;   // target 膨胀后的禁区（可视化用）
    };

    class Imbinarize {
    public:
      static Eigen::MatrixXd binarize(Eigen::MatrixXd& gray_mask, int block_size = 51, double c = -0.5);

      static Eigen::MatrixXd binarize_OTSU(Eigen::MatrixXd& gray_mask);

      // 多阶段自适应二值化 + SRAF 提取 + 最小间距约束
      //   gray_mask            : 灰度掩模 (期望 [0,1] 值域)
      //   target_pattern       : 目标图 (0/1)
      //   peak_min_intensity   : 局部极大值最低强度阈值
      //   quantile_val         : 峰值分位数 → 自适应阈值
      //   area_threshold       : 小于该像素面积的连通域被丢弃
      //   min_spacing_pixels   : SRAF 与主图形最小间距（膨胀半径）
      static BinarizeResult binarize_advanced(
          const Eigen::MatrixXd& gray_mask,
          const Eigen::MatrixXd& target_pattern,
          double peak_min_intensity = 0.1,
          double quantile_val       = 0.5,
          int    area_threshold     = 50,
          int    min_spacing_pixels = 15);
    };

}