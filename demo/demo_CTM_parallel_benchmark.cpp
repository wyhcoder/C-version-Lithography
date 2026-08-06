#include "CTM_Optimizer.h"
#include "litho_prepare.h"
#include "lithography_simulator.h"
#include "save_txt.h"
#include "simulation_parameters.h"

#include <Eigen/Dense>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace litho;

namespace {

struct BenchmarkResult {
    double seconds = 0.0;
    Eigen::MatrixXd mask;
    std::vector<double> pe_history;
};

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t middle = values.size() / 2;
    if (values.size() % 2 == 0) {
        return 0.5 * (values[middle - 1] + values[middle]);
    }
    return values[middle];
}

BenchmarkResult run_once(CTM_Optimizer& optimizer, int thread_count) {
#ifdef _OPENMP
    omp_set_num_threads(thread_count);
#else
    (void)thread_count;
#endif

    const auto start = std::chrono::steady_clock::now();
    Eigen::MatrixXd mask = optimizer.optimize(false, false);
    const auto end = std::chrono::steady_clock::now();

    BenchmarkResult result;
    result.seconds = std::chrono::duration<double>(end - start).count();
    result.mask = std::move(mask);
    result.pe_history = optimizer.error_history();
    return result;
}

}  // namespace

int main(int argc, char** argv) {
#ifndef _OPENMP
    std::cerr << "OpenMP is not enabled. Reconfigure with libomp before running this benchmark.\n"
              << "macOS example:\n"
              << "  cmake .. -DCMAKE_BUILD_TYPE=Release "
                 "-DOpenMP_ROOT=$(brew --prefix libomp)\n";
    return 2;
#else
    const std::string config_path = (argc > 1) ? argv[1] : "config.yaml";

    int iterations = 50;
    int repeats = 2;
    int parallel_threads = omp_get_max_threads();

    try {
        if (argc > 2) iterations = std::stoi(argv[2]);
        if (argc > 3) parallel_threads = std::stoi(argv[3]);
        if (argc > 4) repeats = std::stoi(argv[4]);
    } catch (const std::exception&) {
        std::cerr << "Usage: " << argv[0]
                  << " [config.yaml] [iterations] [parallel_threads] [repeats]\n";
        return 1;
    }

    if (iterations <= 0 || parallel_threads <= 0 || repeats <= 0) {
        std::cerr << "iterations, parallel_threads and repeats must all be positive.\n";
        return 1;
    }

    omp_set_dynamic(0);

    SimulationParameters params;
    try {
        params = SimulationParameters::from_yaml(config_path);
    } catch (const std::exception& error) {
        std::cerr << "Failed to load config: " << error.what() << '\n';
        return 1;
    }

    std::cout << "=== CTM serial vs OpenMP benchmark ===\n"
              << "Config           : " << config_path << '\n'
              << "Iterations       : " << iterations << '\n'
              << "Serial threads   : 1\n"
              << "Parallel threads : " << parallel_threads << '\n'
              << "Repeats          : " << repeats << '\n'
              << "OpenMP version   : " << _OPENMP << "\n\n";

    LithographySimulator simulator(params);

    const auto prepare_start = std::chrono::steady_clock::now();
    LithoPrepare prepare(simulator._grid, simulator._pupil,
                         simulator._source, true);
    ImagingCache cache = prepare.cache();
    const auto prepare_end = std::chrono::steady_clock::now();

    // 构造在计时外，避免 FFTW plan 创建时间影响优化速度比较。
    CTM_Optimizer optimizer(simulator, cache, iterations, 0.9);

    std::cout << "Prepare time     : "
              << std::chrono::duration<double>(prepare_end - prepare_start).count()
              << " s\n\n";

    std::vector<double> serial_times;
    std::vector<double> parallel_times;
    BenchmarkResult serial_reference;
    BenchmarkResult parallel_reference;

    for (int repeat = 0; repeat < repeats; ++repeat) {
        // 交替执行顺序，减小 CPU 缓存、温度和睿频造成的顺序偏差。
        const bool serial_first = (repeat % 2 == 0);
        std::cout << "Round " << (repeat + 1) << '/' << repeats
                  << (serial_first ? " (serial -> parallel)\n"
                                   : " (parallel -> serial)\n");

        BenchmarkResult serial_run;
        BenchmarkResult parallel_run;
        if (serial_first) {
            serial_run = run_once(optimizer, 1);
            parallel_run = run_once(optimizer, parallel_threads);
        } else {
            parallel_run = run_once(optimizer, parallel_threads);
            serial_run = run_once(optimizer, 1);
        }

        serial_times.push_back(serial_run.seconds);
        parallel_times.push_back(parallel_run.seconds);
        serial_reference = std::move(serial_run);
        parallel_reference = std::move(parallel_run);

        std::cout << "  serial   : " << serial_times.back() << " s\n"
                  << "  parallel : " << parallel_times.back() << " s\n";
    }

    if (serial_reference.mask.rows() != parallel_reference.mask.rows() ||
        serial_reference.mask.cols() != parallel_reference.mask.cols()) {
        std::cerr << "Serial and parallel masks have different dimensions.\n";
        return 3;
    }

    const Eigen::MatrixXd mask_diff =
        serial_reference.mask - parallel_reference.mask;
    const double mask_max_abs = mask_diff.cwiseAbs().maxCoeff();
    const double mask_rel_l2 = mask_diff.norm() /
        std::max(1.0, serial_reference.mask.norm());

    if (serial_reference.pe_history.size() !=
        parallel_reference.pe_history.size()) {
        std::cerr << "Serial and parallel PE histories have different lengths.\n";
        return 3;
    }

    double pe_max_abs = 0.0;
    for (std::size_t i = 0; i < serial_reference.pe_history.size(); ++i) {
        pe_max_abs = std::max(
            pe_max_abs,
            std::abs(serial_reference.pe_history[i] -
                     parallel_reference.pe_history[i]));
    }

    const double serial_median = median(serial_times);
    const double parallel_median = median(parallel_times);
    const double speedup = serial_median / parallel_median;

    std::cout << std::setprecision(10)
              << "\n=== Summary (median) ===\n"
              << "Serial time      : " << serial_median << " s\n"
              << "Parallel time    : " << parallel_median << " s\n"
              << "Speedup          : " << speedup << "x\n"
              << "Mask max |diff|  : " << mask_max_abs << '\n'
              << "Mask relative L2 : " << mask_rel_l2 << '\n'
              << "PE max |diff|    : " << pe_max_abs << '\n';

    const bool finite = serial_reference.mask.allFinite() &&
                        parallel_reference.mask.allFinite();
    const bool consistent = finite && mask_max_abs <= 1e-12 &&
                            pe_max_abs <= 1e-12;
    std::cout << "Consistency      : "
              << (consistent ? "PASS" : "FAIL") << '\n';

    SaveTxt::save_mat(serial_reference.mask,
                      "CTM_result/benchmark_serial_mask.txt");
    SaveTxt::save_mat(parallel_reference.mask,
                      "CTM_result/benchmark_parallel_mask.txt");
    SaveTxt::save_mat(mask_diff,
                      "CTM_result/benchmark_mask_diff.txt");

    return consistent ? 0 : 4;
#endif
}
