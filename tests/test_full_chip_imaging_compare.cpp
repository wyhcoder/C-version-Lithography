#include "full_chip_imaging_compare.h"
#include "full_chip_epe.h"
#include "fft.h"
#include "lithography_simulator.h"
#include "loss.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace litho;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

SimulationParameters make_params() {
    SimulationParameters params{};
    params.system = {193.0, 8.0, 128};
    params.optics.na = 1.35;
    params.optics.refractive_index = 1.44;
    params.optics.aberrations = {{4, 0.07}, {5, 0.03}, {7, 0.02}};
    params.source = {"annular", 0.6, 0.9};
    params.resist = {"sigmoid", 0.25, 12};
    return params;
}

Eigen::MatrixXd make_mask(int n) {
    Eigen::MatrixXd mask = Eigen::MatrixXd::Constant(n, n, 0.15);
    mask.block(n / 4, n / 5, n / 3, n / 2).setConstant(0.85);
    mask.block(n / 2, n / 3, n / 8, n / 9).setConstant(0.35);
    // 靠近周期边界的灰度条带检验核补零后频率位置是否正确。
    mask.block(0, n / 2, 3, n / 5).setConstant(0.6);
    return mask;
}

// 检查奇偶、非 2 的幂及小尺寸：像素中心关于几何中心对称，频率轴与 FFT 移位的零频位置一致。
void check_grid_sizes() {
    for (int n : {1, 4, 5, 62, 63, 256, 257}) {
        constexpr double dx = 7.5; // nm；非整数像素尺寸也应保持精确的空间/频率间隔。
        const Grid grid(n, dx, 1.35, 193.0);
        const auto& coords = grid.grid_coords();
        require(grid.size() == n && coords.x.rows() == n && coords.Fx_2d.cols() == n, "exact grid rounded the matrix dimensions");
        const double df = 193.0 / (n * dx * 1.35);
        require(coords.Fx_1d(n / 2) == 0.0, "DC does not lie at the centered FFT index");
        require(std::abs(coords.Fx_1d(0) + n / 2 * df) < 1e-12, "wrong negative FFT endpoint");
        require(std::abs(coords.Fx_1d(n - 1) - (n - 1 - n / 2) * df) < 1e-12, "wrong positive FFT endpoint");
        require(std::abs(coords.x(0, 0) + (n - 1) * dx / 2) < 1e-12, "wrong left pixel center");
        require(std::abs(coords.x(0, n - 1) - (n - 1) * dx / 2) < 1e-12, "wrong right pixel center");
        require(coords.x(0, n / 2) == (n % 2 == 0 ? dx / 2 : 0.0), "wrong spatial origin convention");
        for (int i = 0; i < n; ++i) {
            require(coords.x(0, i) == -coords.x(0, n - 1 - i), "pixel centers are not geometrically symmetric");
            require(coords.y(i, 0) == coords.x(0, i), "x/y spatial axes disagree");
            require(coords.Fx_2d(n / 2, i) == coords.Fx_1d(i), "frequency x axis does not follow columns");
            require(coords.Fy_2d(i, n / 2) == coords.Fy_1d(i), "frequency y axis does not follow rows");
            if (i > 0) {
                require(coords.x(0, i) - coords.x(0, i - 1) == dx, "wrong pixel spacing");
                require(std::abs(coords.Fx_1d(i) - coords.Fx_1d(i - 1) - df) < 1e-12, "wrong frequency spacing");
            }
        }

        // 角点的单位脉冲代表零频位置，检验两种 fftshift 接口和反向移位都与频率轴一致。
        Eigen::MatrixXcd corner = Eigen::MatrixXcd::Zero(n, n);
        corner(0, 0) = 1.0;
        const auto returned = FFT::fftshift(corner);
        auto inplace = corner;
        FFT::fftshift_inplace(inplace);
        require(returned(n / 2, n / 2) == 1.0, "fftshift did not move DC to the frequency origin");
        require((returned - inplace).norm() == 0.0, "fftshift interfaces disagree for this parity");
        FFT::ifftshift_inplace(inplace);
        require((inplace - corner).norm() == 0.0, "FFT shifts are not inverse operations");
    }
}

void check_compact_socs() {
    const auto params = make_params();
    const Grid grid(128, 8.0, 1.35, 193.0);
    const auto& coords = grid.grid_coords();
    const Pupil pupil(LithographySimulator::_make_pupil_params(params), coords.Fx_2d, coords.Fy_2d);
    const Source source(LithographySimulator::_make_source_params(params, coords.Fx_1d));
    const LithoPrepare dense(grid, pupil, source, true);
    const LithoPrepare compact(grid, pupil, source, true, true);
    require(dense.cache().N == compact.cache().N, "compact path changed FFT dimensions");
    for (int k = 0; k < dense.cache().socs_eigenvalues.size(); ++k) {
        require(std::abs(dense.cache().socs_eigenvalues[k] - compact.cache().socs_eigenvalues[k]) < 1e-9,
                "compact path changed SOCS eigenvalues");
    }
    Imaging dense_imaging(dense.cache()), compact_imaging(compact.cache());
    const auto mask = make_mask(128);
    const auto a = dense_imaging.compute(mask, 0.25, 12.0);
    const auto b = compact_imaging.compute(mask, 0.25, 12.0);
    // SVD 相位可不同，所以比较成像可观测量，而不直接比较各个复数特征核。
    require((a.aerial_image - b.aerial_image).cwiseAbs().maxCoeff() < 1e-10, "compact path changed aerial image");
    require((a.wafer_image - b.wafer_image).cwiseAbs().maxCoeff() < 1e-10, "compact path changed wafer image");
}

void check_full_frame_comparison(int n) {
    const auto params = make_params();
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(n, n);
    target.block(12, 9, 32, 28).setOnes();
    const auto gray = make_mask(n);
    const auto comparison = compare_full_chip_imaging(params, target, gray, n);
    require(comparison.grid_size == n, "optical grid must match full target dimensions");
    require((comparison.target_mask - target).norm() == 0.0, "target mask was changed");
    require((comparison.optimized_mask - gray).norm() == 0.0, "optimized mask was binarized or otherwise changed");

    // 用独立构造的完整 SVD 全图模型校验两个输入的成像，避免只检查导出的 mask 而漏掉成像前的转换。
    const Grid grid(n, 8.0, 1.35, 193.0);
    const auto& coords = grid.grid_coords();
    const Pupil pupil(LithographySimulator::_make_pupil_params(params), coords.Fx_2d, coords.Fy_2d);
    const Source source(LithographySimulator::_make_source_params(params, coords.Fx_1d));
    const LithoPrepare prepare(grid, pupil, source, true);
    Imaging imaging(prepare.cache());
    const auto a = imaging.compute(target, params.resist.threshold, params.resist.alpha);
    const auto b = imaging.compute(gray, params.resist.threshold, params.resist.alpha);
    require((a.aerial_image - comparison.target_image.aerial_image).cwiseAbs().maxCoeff() < 1e-10, "wrong target AI");
    require((a.wafer_image - comparison.target_image.wafer_image).cwiseAbs().maxCoeff() < 1e-10, "wrong target WI");
    require((b.aerial_image - comparison.optimized_image.aerial_image).cwiseAbs().maxCoeff() < 1e-10, "wrong gray AI");
    require((b.wafer_image - comparison.optimized_image.wafer_image).cwiseAbs().maxCoeff() < 1e-10, "wrong gray WI");
    require(std::abs(comparison.target_pe - Loss::pe_loss(a.wafer_image, target)) < 1e-9, "wrong target PE");
    require(std::abs(comparison.optimized_pe - Loss::pe_loss(b.wafer_image, target)) < 1e-9, "wrong optimized PE");
}

// 非方形、不能整除的分区：独立 Guard 必须参与局部成像，回拼时每个全局 Core 像素只写一次。
void check_micro_comparison() {
    const auto params = make_params();
    const auto meta = Partition::build(55, 70, 2, 3, 8, 3);
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(meta.height, meta.width);
    target.block(9, 15, 32, 29).setOnes();
    Eigen::MatrixXd gray(meta.height, meta.width);
    for (int y = 0; y < meta.height; ++y) {
        for (int x = 0; x < meta.width; ++x) gray(y, x) = 0.15 + 0.4 * ((x + 2 * y) % 17) / 16.0;
    }
    auto locals = Partition::extract_all(gray, meta);
    for (const auto& tile : meta.tiles) {
        const auto& u = tile.update_local;
        const auto& c = tile.core_local;
        for (int y = u.y0; y < u.y1; ++y) {
            for (int x = u.x0; x < u.x1; ++x) {
                if (y < c.y0 || y >= c.y1 || x < c.x0 || x >= c.x1) locals[tile.id](y, x) = 0.8 + 0.02 * tile.id;
            }
        }
    }
    const Grid grid(meta.tile_size, 8.0, 1.35, 193.0);
    const auto& coords = grid.grid_coords();
    const Pupil pupil(LithographySimulator::_make_pupil_params(params), coords.Fx_2d, coords.Fy_2d);
    const Source source(LithographySimulator::_make_source_params(params, coords.Fx_1d));
    const LithoPrepare prepare(grid, pupil, source, true);
    const auto parallel = compare_micro_imaging(params, meta, prepare.cache(), target, locals, true);
    const auto serial = compare_micro_imaging(params, meta, prepare.cache(), target, locals, false);
    require(parallel.mode == FullChipImagingMode::MicroCore, "wrong imaging mode");
    require(parallel.grid_size == meta.tile_size && parallel.image_rows == 55 && parallel.image_cols == 70,
            "local optics and global image dimensions were mixed");
    require(parallel.micro_count == 6, "wrong Micro count");
    require((parallel.optimized_mask - gray).norm() == 0.0, "Core gray mask was changed");
    require((parallel.target_image.aerial_image - serial.target_image.aerial_image).norm() == 0.0, "parallel target AI changed");
    require((parallel.target_image.wafer_image - serial.target_image.wafer_image).norm() == 0.0, "parallel target WI changed");
    require((parallel.optimized_image.aerial_image - serial.optimized_image.aerial_image).norm() == 0.0, "parallel AI changed");
    require((parallel.optimized_image.wafer_image - serial.optimized_image.wafer_image).norm() == 0.0, "parallel WI changed");

    Imaging reference(prepare.cache());
    double guard_effect = 0.0;
    for (const auto& tile : meta.tiles) {
        const auto baseline = reference.compute(Partition::extract_local_tile(target, tile), params.resist.threshold, params.resist.alpha);
        const auto optimized = reference.compute(locals[tile.id], params.resist.threshold, params.resist.alpha);
        const auto collapsed = reference.compute(Partition::extract_local_tile(gray, tile), params.resist.threshold, params.resist.alpha);
        require((optimized.aerial_image - parallel.optimized_local_aerial_images[tile.id]).norm() == 0.0,
                "saved local AI lost independent Guard state");
        // 按全局/局部坐标逐像素核对，独立于 assemble_global() 的切片实现。
        for (int y = tile.core_global.y0; y < tile.core_global.y1; ++y) {
            for (int x = tile.core_global.x0; x < tile.core_global.x1; ++x) {
                const int ly = y - tile.tile_global.y0, lx = x - tile.tile_global.x0;
                require(parallel.target_image.aerial_image(y, x) == baseline.aerial_image(ly, lx), "wrong baseline Core AI");
                require(parallel.target_image.wafer_image(y, x) == baseline.wafer_image(ly, lx), "wrong baseline Core WI");
                require(parallel.optimized_image.aerial_image(y, x) == optimized.aerial_image(ly, lx), "wrong optimized Core AI");
                require(parallel.optimized_image.wafer_image(y, x) == optimized.wafer_image(ly, lx), "wrong optimized Core WI");
                guard_effect += std::abs(optimized.aerial_image(ly, lx) - collapsed.aerial_image(ly, lx));
            }
        }
    }
    require(guard_effect > 1e-3, "test did not distinguish independent Guard from re-extracted global mask");
    const double pe = (parallel.optimized_image.wafer_image - target).squaredNorm();
    require(std::abs(parallel.optimized_pe - pe) < 1e-10, "assembled PE counted overlapping Halo/Guard");
}

// 人为制造块间强度跳变；线性局部 AI 的解析斜率必须用于 Core 边缘 EP，不能在拼接 AI 上求导。
void check_micro_ep_epe() {
    const auto meta = Partition::build(55, 70, 2, 3, 8, 3);
    std::vector<Eigen::MatrixXd> ai(meta.tiles.size());
    for (const auto& tile : meta.tiles) {
        ai[tile.id].resize(meta.tile_size, meta.tile_size);
        for (int y = 0; y < meta.tile_size; ++y) {
            for (int x = 0; x < meta.tile_size; ++x) ai[tile.id](y, x) = 0.1 + 0.01 * y + 0.02 * x + 0.35 * tile.id;
        }
    }
    const auto directory = std::filesystem::temp_directory_path() / "litho_test_micro_imaging";
    std::filesystem::create_directories(directory);
    const int order[] = {4, 0, 5, 1, 3, 2}; // CSV 行序故意不同于 Micro 编号。
    Eigen::MatrixXd global_eps(6, 2);
    {
        std::ofstream points(directory / "ep_points.csv");
        points << "ep_id,y,x,micro_id,weight_epe,weight_meef\n";
        for (int i = 0; i < 6; ++i) {
            const auto& tile = meta.tiles[order[i]];
            global_eps(i, 0) = tile.core_global.y0 + 0.25;
            global_eps(i, 1) = tile.core_global.x0 + 0.5;
            points << 90 - i << ',' << global_eps(i, 0) << ',' << global_eps(i, 1) << ',' << tile.id << ',' << i + 1 << ",0.5\n";
        }
    }
    const auto evaluation = save_micro_ep_epe(ai, meta, directory / "ep_points.csv", 0.25, 8.0, directory / "ep_epe.csv");
    std::ifstream csv(directory / "ep_epe.csv");
    std::string line;
    std::getline(csv, line);
    double expected_loss = 0.0, expected_abs_sum_nm = 0.0;
    EpsResult eps_result{global_eps, Eigen::RowVectorXd(6), Eigen::RowVectorXd::Constant(6, 0.5)};
    for (int i = 0; i < 6; ++i) {
        const auto& tile = meta.tiles[order[i]];
        const double local_y = tile.core_local.y0 + 0.25, local_x = tile.core_local.x0 + 0.5;
        const double intensity = 0.1 + 0.01 * local_y + 0.02 * local_x + 0.35 * tile.id;
        const double gradient = std::sqrt(0.0005);
        const double signed_epe = (intensity - 0.25) / gradient;
        const double expected_nm = std::abs(signed_epe) * 8.0;
        const double point_loss = (i + 1) * signed_epe * signed_epe;
        eps_result.weight_epe(i) = i + 1;
        expected_loss += point_loss;
        expected_abs_sum_nm += expected_nm;
        require(std::abs(evaluation.signed_epe_pixel(i) - signed_epe) < 1e-9,
                "EP slope did not use its full owner Micro AI");
        std::getline(csv, line);
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream row(line);
        int id, owner;
        double y, x, weight, meef, stored_intensity, stored_gradient, stored_signed, stored_loss, stored_nm;
        row >> id >> y >> x >> owner >> weight >> meef >> stored_intensity >> stored_gradient >> stored_signed >> stored_loss >> stored_nm;
        require(id == 90 - i && owner == order[i] && y == global_eps(i, 0) && x == global_eps(i, 1),
                "EP order/coordinates changed");
        require(weight == i + 1 && meef == 0.5 && std::abs(stored_intensity - intensity) < 1e-14 &&
                std::abs(stored_gradient - gradient) < 1e-14,
                "EP intensity, gradient or weight export changed");
        require(std::abs(stored_signed - signed_epe) < 1e-9 && std::abs(stored_loss - point_loss) < 1e-9 &&
                std::abs(stored_nm - expected_nm) < 1e-9,
                "subpixel EP export is wrong");
    }
    const auto stitched = Loss::evaluate_sparse_epe(Partition::assemble_global(ai, meta), eps_result, 0.25, 8.0);
    require((stitched.absolute_epe_nm - evaluation.absolute_epe_nm).norm() > 1.0,
            "test did not detect cross-seam spatial differences");
    require(std::abs(evaluation.weighted_squared_loss - expected_loss) < 1e-9 &&
            std::abs(evaluation.absolute_sum_nm - expected_abs_sum_nm) < 1e-9 && evaluation.active_count == 6,
            "EP totals are wrong");
    std::filesystem::remove_all(directory);
}

int main() {
#ifdef _OPENMP
    omp_set_num_threads(2);
#endif
    check_grid_sizes();
    check_compact_socs();
    check_full_frame_comparison(63);
    check_full_frame_comparison(64);
    check_micro_comparison();
    check_micro_ep_epe();
    std::cout << "Compact SOCS, Micro Core assembly, independent Guard and owner-Micro EP evaluation passed.\n";
}
