#pragma once

#include <Eigen/Dense>
#include <vector>
#include "grid.h"
#include "pupil.h"
#include "source.h"


// 前向声明 FFTW 类型（避免在头文件 include fftw3.h）
typedef double fftw_complex[2];
typedef struct fftw_plan_s* fftw_plan;

namespace litho {

// Abbe 成像所需的全部预计算量
// 由 LithoPrepare 一次性计算，后续成像循环直接使用
struct ImagingCache {
    // ── 空间网格 ────────────────────────────────────
    int             N;                  // 网格尺寸
    Eigen::MatrixXd X;                  // 空间 x 坐标 [N×N], nm
    Eigen::MatrixXd Y;                  // 空间 y 坐标 [N×N], nm

    // ── 频率网格（2D，来自 Grid）───────────────────
    Eigen::MatrixXd Fx_2d;             // 频率 fx [N×N], cycles/nm
    Eigen::MatrixXd Fy_2d;             // 频率 fy [N×N], cycles/nm

    // ── 光瞳函数 ───────────────────────────────────
    Eigen::MatrixXcd pupil_frequency;  // 原始复光瞳 P(fx, fy) [N×N]

    // ── 光源采样点（一维展开，仅非零点）────────────
    Eigen::VectorXd source_fs_phys;    // 物理频率 fs [Ns], cycles/nm
    Eigen::VectorXd source_gs_phys;    // 物理频率 gs [Ns], cycles/nm
    Eigen::VectorXd source_ws;         // 权重 [Ns]，总和 ≈ 1
    Eigen::VectorXd source_no_ws;      // 无权重 [Ns]

    // Abbe 模式：每个光源点对应的频移光瞳和空间域 PSF。
    std::vector<Eigen::MatrixXcd> abbe_kernels_frequency;
    std::vector<Eigen::MatrixXcd> abbe_kernels_spatial;
    // SOCS 模式：TCC/SVD 得到的特征值和相干模态核。
    std::vector<double> socs_eigenvalues;
    std::vector<Eigen::MatrixXcd> socs_kernels_frequency;
    std::vector<Eigen::MatrixXcd> socs_kernels_spatial;
    // 统一核缓存：前向/反传读取频域核；空间域缓存当前仅由 SOCS 准备路径填充。
    std::vector<Eigen::MatrixXcd> optical_kernels_spatial;
    std::vector<Eigen::MatrixXcd> optical_kernels_frequency;
};

class LithoPrepare {
public:
    // grid / pupil / source 必须已经完成构造（即 compute 已调用）
    LithoPrepare(const Grid& grid, const Pupil& pupil, const Source& source, bool use_socs);
    ~LithoPrepare();
    const ImagingCache& cache() const { return _cache; }

private:
    // 从 source_weight_map 中提取非零采样点，转换为物理频率，数据结构的大小依据非零光源点决定
    void _extract_source_points(const SourceMap& smap,
                                double f_max = 1.0,
                                double min_weight = 1e-8);
    
    // 预计算 Abbe 模式下每个光源点对应的频移光瞳核
    // 所有输入已存在 _cache，无需重复传参
    void _compute_abbe_kernels();
    void _compute_socs_kernels();

    ImagingCache _cache;
    bool _is_shift = false;

    fftw_complex* _fft_in;
    fftw_complex* _fft_out;
    fftw_plan _fft_plan;
    fftw_plan _ifft_plan;

    //socs计算中间量   
    bool _use_socs = false;
    
    int  _num_socs = 36;     // 0 = 用全部光源点；>0 = top-K 个相干核
    



};


}  // namespace litho
