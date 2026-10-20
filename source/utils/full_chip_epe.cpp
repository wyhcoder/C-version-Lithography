#include "full_chip_epe.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace litho {

namespace {

struct EpPoint {
    int id, micro_id; // 原全局 EP 编号、其唯一所属 Core 的 Micro 编号，均从 0 开始。
    double y, x; // 全局行/列像素索引；与原 ep_points.csv 一致，不转换为物理坐标。
    double weight_epe, weight_meef; // 保存原权重供分析；不乘入逐点绝对 EPE。
};

std::vector<EpPoint> read_ep_points(const std::filesystem::path& ep_points_csv) {
    std::ifstream input(ep_points_csv);
    std::string line;
    std::getline(input, line); // 原 CSV 表头：ep_id,y,x,micro_id,weight_epe,weight_meef。
    std::vector<EpPoint> points;
    while (std::getline(input, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream values(line);
        EpPoint point;
        values >> point.id >> point.y >> point.x >> point.micro_id >> point.weight_epe >> point.weight_meef;
        points.push_back(point);
    }
    return points;
}

// 两种成像方式共用原 CSV 行序/浮点坐标；所有评价向量的下标均为原 EP 行号。
void write_ep_epe(const std::vector<EpPoint>& points, const SparseEpeEvaluation& evaluation,
                  const std::filesystem::path& output_csv) {
    std::ofstream output(output_csv);
    output << "ep_id,y,x,micro_id,weight_epe,weight_meef,aerial_intensity,gradient_magnitude,signed_epe_pixel,"
           << "weighted_epe_squared_pixel,epe_nm\n" << std::setprecision(17);
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        const auto& p = points[i];
        output << p.id << ',' << p.y << ',' << p.x << ',' << p.micro_id << ',' << p.weight_epe << ',' << p.weight_meef
               << ',' << evaluation.aerial_samples(i) << ',' << evaluation.gradient_samples(i) << ','
               << evaluation.signed_epe_pixel(i) << ',' << evaluation.point_losses(i) << ',' << evaluation.absolute_epe_nm(i) << '\n';
    }
}

} // namespace

SparseEpeEvaluation save_full_chip_ep_epe(const Eigen::MatrixXd& aerial_image, const std::filesystem::path& ep_points_csv,
                                          double threshold, double pixel_size_nm, const std::filesystem::path& output_csv) {
    const auto points = read_ep_points(ep_points_csv);
    const int count = static_cast<int>(points.size());
    EpsResult eps_result{Eigen::MatrixXd(count, 2), Eigen::RowVectorXd(count), Eigen::RowVectorXd(count)};
    for (int i = 0; i < count; ++i) {
        eps_result.eps(i, 0) = points[i].y;
        eps_result.eps(i, 1) = points[i].x;
        eps_result.weight_epe(i) = points[i].weight_epe;
        eps_result.weight_meef(i) = points[i].weight_meef;
    }
    // 使用完整全图 AI 求空间差分，再在浮点 EP 坐标处双线性取样；不重新初始化光学模型。
    const auto evaluation = Loss::evaluate_sparse_epe(aerial_image, eps_result, threshold, pixel_size_nm);
    write_ep_epe(points, evaluation, output_csv);
    return evaluation;
}

SparseEpeEvaluation save_micro_ep_epe(const std::vector<Eigen::MatrixXd>& local_aerial_images, const PartitionMeta& meta,
                                     const std::filesystem::path& ep_points_csv, double threshold, double pixel_size_nm,
                                     const std::filesystem::path& output_csv) {
    const auto points = read_ep_points(ep_points_csv);
    const int count = static_cast<int>(points.size());
    const double nan = std::numeric_limits<double>::quiet_NaN();
    SparseEpeEvaluation evaluation{};
    evaluation.signed_epe_pixel = Eigen::RowVectorXd::Constant(count, nan);
    evaluation.absolute_epe_nm = Eigen::RowVectorXd::Constant(count, nan);
    evaluation.aerial_samples = Eigen::RowVectorXd::Constant(count, nan);
    evaluation.gradient_samples = Eigen::RowVectorXd::Constant(count, nan);
    evaluation.point_losses = Eigen::RowVectorXd::Zero(count);
    std::vector<std::vector<int>> rows_by_micro(meta.tiles.size());
    for (int i = 0; i < count; ++i) rows_by_micro[points[i].micro_id].push_back(i);
    for (const auto& tile : meta.tiles) {
        const auto& rows = rows_by_micro[tile.id];
        if (rows.empty()) continue; // 此块没有 EP，无需计算空间差分。
        EpsResult local_eps{Eigen::MatrixXd(rows.size(), 2), Eigen::RowVectorXd(rows.size()), Eigen::RowVectorXd(rows.size())};
        for (int j = 0; j < static_cast<int>(rows.size()); ++j) {
            const auto& p = points[rows[j]];
            local_eps.eps(j, 0) = p.y - tile.tile_global.y0;
            local_eps.eps(j, 1) = p.x - tile.tile_global.x0;
            local_eps.weight_epe(j) = p.weight_epe;
            local_eps.weight_meef(j) = p.weight_meef;
        }
        // 在完整 Halo AI 上求 ∇AI，再双线性采样本块拥有的 EP；包含 Core 边缘所需的 Guard/Halo 光强。
        const auto local = Loss::evaluate_sparse_epe(local_aerial_images[tile.id], local_eps, threshold, pixel_size_nm);
        for (int j = 0; j < static_cast<int>(rows.size()); ++j) {
            const int row = rows[j];
            evaluation.signed_epe_pixel(row) = local.signed_epe_pixel(j);
            evaluation.absolute_epe_nm(row) = local.absolute_epe_nm(j);
            evaluation.aerial_samples(row) = local.aerial_samples(j);
            evaluation.gradient_samples(row) = local.gradient_samples(j);
            evaluation.point_losses(row) = local.point_losses(j);
        }
        evaluation.weighted_squared_loss += local.weighted_squared_loss;
        evaluation.absolute_sum_nm += local.absolute_sum_nm;
        evaluation.active_count += local.active_count;
    }
    write_ep_epe(points, evaluation, output_csv);
    return evaluation;
}

} // namespace litho
