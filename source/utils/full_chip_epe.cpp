#include "full_chip_epe.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
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

// 两种成像方式共用原 CSV 行序/坐标，intensities 与 evaluation 的下标均为原 EP 行号。
void write_ep_epe(const std::vector<EpPoint>& points, const Eigen::VectorXd& intensities, const EpeEvaluation& evaluation,
                 const std::filesystem::path& output_csv) {
    std::ofstream output(output_csv);
    output << "ep_id,y,x,micro_id,weight_epe,weight_meef,aerial_intensity,epe_nm\n" << std::setprecision(17);
    for (int i = 0; i < static_cast<int>(points.size()); ++i) {
        const auto& p = points[i];
        output << p.id << ',' << p.y << ',' << p.x << ',' << p.micro_id << ',' << p.weight_epe << ',' << p.weight_meef
               << ',' << intensities(i) << ',' << evaluation.epe_vector(i) << '\n';
    }
}

} // namespace

EpeEvaluation save_full_chip_ep_epe(const Eigen::MatrixXd& aerial_image, const std::filesystem::path& ep_points_csv,
                                  double threshold, double pixel_size_nm, const std::filesystem::path& output_csv) {
    const auto points = read_ep_points(ep_points_csv);
    const int count = static_cast<int>(points.size());
    Eigen::MatrixXd eps(count, 2);
    Eigen::VectorXd intensities(count);
    for (int i = 0; i < count; ++i) {
        eps(i, 0) = points[i].y;
        eps(i, 1) = points[i].x;
        intensities(i) = aerial_image(static_cast<int>(points[i].y), static_cast<int>(points[i].x));
    }
    // 使用完整全图 AI 求空间差分，再取各 EP；不先裁 Core，也不重新初始化光学模型。
    // 原接口按像素单位求 |∇AI|，最后乘 pixel_size_nm，将 EPE 从像素换算为 nm。
    const auto evaluation = Loss::evaluate_epe(aerial_image, eps, threshold, pixel_size_nm);
    write_ep_epe(points, intensities, evaluation, output_csv);
    return evaluation;
}

EpeEvaluation save_micro_ep_epe(const std::vector<Eigen::MatrixXd>& local_aerial_images, const PartitionMeta& meta,
                               const std::filesystem::path& ep_points_csv, double threshold, double pixel_size_nm,
                               const std::filesystem::path& output_csv) {
    const auto points = read_ep_points(ep_points_csv);
    const int count = static_cast<int>(points.size());
    EpeEvaluation evaluation{};
    evaluation.epe_vector.resize(count);
    Eigen::VectorXd intensities(count);
    std::vector<std::vector<int>> rows_by_micro(meta.tiles.size());
    for (int i = 0; i < count; ++i) rows_by_micro[points[i].micro_id].push_back(i);
    for (const auto& tile : meta.tiles) {
        const auto& rows = rows_by_micro[tile.id];
        if (rows.empty()) continue; // 此块没有 EP，无需计算空间差分。
        Eigen::MatrixXd local_eps(rows.size(), 2);
        for (int j = 0; j < static_cast<int>(rows.size()); ++j) {
            const auto& p = points[rows[j]];
            local_eps(j, 0) = p.y - tile.tile_global.y0;
            local_eps(j, 1) = p.x - tile.tile_global.x0;
        }
        // 在完整 Halo AI 上求 ∇AI，再取本块拥有的 EP；包含 Core 边缘所需的 Guard/Halo 光强。
        const auto local = Loss::evaluate_epe(local_aerial_images[tile.id], local_eps, threshold, pixel_size_nm);
        for (int j = 0; j < static_cast<int>(rows.size()); ++j) {
            const int row = rows[j];
            intensities(row) = local_aerial_images[tile.id](static_cast<int>(local_eps(j, 0)), static_cast<int>(local_eps(j, 1)));
            evaluation.epe_vector(row) = local.epe_vector(j);
        }
    }
    evaluation.total_epe = evaluation.epe_vector.sum();
    write_ep_epe(points, intensities, evaluation, output_csv);
    return evaluation;
}

} // namespace litho
