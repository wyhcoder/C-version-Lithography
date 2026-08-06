#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include "ep_select.h"       // ControlPoints, EpsResult, EpSelect
#include "msaa.h"            // AntiAliasRenderer, Polygon, Polygons
#include "parametric.h"      // ParametricDemo

namespace litho {

// ── 管道配置（对应 MEEFPipelineConfig dataclass）────────────────────────────
struct MEEFConfig {
    // 必填
    std::string pattern_name;
    std::string ls_mask_path;
    std::string file_name;

    // 主图形 CP
    int         main_cp_interval = 5;
    std::string main_symmetry    = "none";

    // SRAF CP
    double sraf_cp_arclen = 3.0;
    int    sraf_min_cps   = 8;
    double sraf_smoothing = 0.3;
    int    sraf_curve_pts = 200;
    int    sraf_min_area  = 20;

    // 渲染
    int msaa_level = 16;

    // 优化器
    std::string curve_type = "BS";
    double      delta      = 0.15;

    // 拆分
    int dilate_radius = 2;

    // EP 点
    int    interval_line   = 5;
    int    interval_corner = 2;
    double mid_weight      = 4.0;
    double other_weight    = 1.0;
};

// ── 主管道类（对应 MEEFPipelineSetup）─────────────────────────────────────
class MEEFPipeline {
public:
    // 构造即运行全流程
    MEEFPipeline(const Eigen::MatrixXd& target_mask,
                 const MEEFConfig&      cfg);

    // ── 对外兼容 MEEF 优化器的字段 ──────────────────────────────────────
    Eigen::MatrixXd target_mask;
    std::string     curve_type;
    std::string     pattern_name;
    double          delta;

    ControlPoints   cps;            // 主图形 CP（优化变量）
    int             num_cps;

    EpsResult       eps_result;     // EP 点 + EPE/MEEF 权重
    int             num_eps;
    int             num_weps;

    Eigen::MatrixXd initial_mask;   // 主图形+SRAF 的完整初始 mask
    Eigen::MatrixXd sraf_mask;      // SRAF 常量 mask（优化期间不变）

    // SRAF 附加字段
    Polygons        sraf_cps;       // 每个 SRAF 块的闭合 CP
    int             num_sraf_cps;
    int             sraf_contour_points;

    // 中间产物
    Eigen::MatrixXd lsm_mask;
    Eigen::MatrixXd main_mask;

private:
    // ── 1. 加载 + 拆分 ─────────────────────────────────────────────────
    void _load_lsm();
    void _split_main_and_sraf();

    // ── 2. 主图形 CP + EP ──────────────────────────────────────────────
    void _extract_main();

    // ── 3. SRAF CP ─────────────────────────────────────────────────────

    // ── 4. SRAF 渲染 ───────────────────────────────────────────────────
    void _render_sraf();

    // ── 5. 组装 initial_mask ───────────────────────────────────────────
    void _assemble_initial();

    // ── helpers ─────────────────────────────────────────────────────────
    static Eigen::MatrixXd _load_txt(const std::string& path);
    static Eigen::MatrixXd _center_crop_pad(const Eigen::MatrixXd& src,
                                             int target_h, int target_w);
    static Eigen::MatrixXd _binary_dilation(const Eigen::MatrixXd& mask,
                                              int radius);
    static std::vector<Eigen::MatrixXi>
        _label_components(const Eigen::MatrixXd& bin, int min_area);

    // SRAF CP 弧长重采样
    static Eigen::MatrixXd _resample_arclen(const Eigen::MatrixXd& contour,
                                              double step, int min_n);

    // 对称配对
    static void _symmetry_pair_blocks(
        const std::vector<Eigen::MatrixXd>& contours,
        const std::vector<Eigen::Vector2d>& centroids,
        int h, int w, const std::string& symmetry,
        std::vector<int>& refs, std::vector<int>& mirrors,
        std::vector<int>& self_sym);

    static Eigen::MatrixXd _apply_symmetry(const Eigen::MatrixXd& pts,
                                            int h, int w,
                                            const std::string& symmetry);

    MEEFConfig _cfg;
};

}  // namespace litho
