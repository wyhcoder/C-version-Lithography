#pragma once

#include "parametric.h"
#include "ep_select.h"
#include <opencv2/core.hpp>
#include <vector>

namespace litho {

struct SrafShrinkConfig {
    double width_threshold_px = 6.0; // 最宽截面、法线材料厚度及点对距离的筛选下限，单位 pixel。
    double minimum_width_px = 3.0; // 限制单次位移时采用的剩余厚度目标，单位 pixel。
    double step_px = 0.5; // 单次最大法向内缩量，单位 pixel；不是强度损失的梯度步长。
};

struct SrafShrinkPoint {
    int contour = 0; // 整张输入图内的零基轮廓编号，孔洞轮廓也有独立编号。
    int point = 0; // 轮廓内部控制点编号，从 0 开始。
    Eigen::Vector2d inward_yx = Eigen::Vector2d::Zero(); // 指向 SRAF 材料内部的单位法线，顺序 (y,x)。
    double width_px = 0.0; // 控制点沿材料内法线方向的局部厚度，单位 pixel。
    double hotspot_distance_px = 0.0; // 到本块评价窗口内最近热点像素中心的距离，单位 pixel，供移动记录使用。
    double movement_px = 0.0; // 本次建议内缩量，单位 pixel；0 表示不移动。
};

struct SrafShrinkProposal {
    ControlPoints control_points; // 完整候选控制点，轮廓和点的顺序与输入一致，坐标单位 pixel。
    std::vector<SrafShrinkPoint> points; // 联合更新实际涉及的控制点，按 contour/point 识别。
    int moved_count = 0; // 去重后实际具有正位移的控制点数量。
};

// 输入闭合轮廓采样点与查询位置 (y,x)，按奇偶填充规则判断是否在 SRAF 材料内，兼容孔洞和反向轮廓。
bool inside_sraf_material(const Polygons& curves, const Eigen::Vector2d& position_yx);

// 检查闭合采样轮廓的非相邻线段是否出现严格交叉；相邻公共端点不算自交。
bool sraf_contour_self_intersects(const Polygon& curve);

struct SrafRegion {
    int outer_contour = -1; // SRAF 编号沿用其外轮廓的零基编号，不将孔洞单独作为另一块 SRAF。
    std::vector<int> contour_indices; // 本块外轮廓和直接孔洞的全局轮廓编号。
    Eigen::MatrixXd initial_mask; // 与整图同尺寸的初始灰度覆盖率，用于本块 printing 归属和固定评价范围。
};

// 对固定 0/255 窗口内全部像素求 mean(max(AI-target,0)^2)，target 与 AI 使用同一归一化强度。
// 窗口坐标为整图零基 (y,x) 像素索引，必须非空；不使用 WI 或只保留当前超阈值像素。
double sraf_printing_loss(const Eigen::MatrixXd& aerial, const cv::Mat& window, double target);

// 返回本块所有有效控制点的当前 CR 材料内法线和法线厚度，不按最宽区域或点对筛选。
// 输入输出的轮廓编号、点编号和 (y,x) 坐标均沿用整图定义，厚度单位 pixel。
std::vector<SrafShrinkPoint> sraf_control_normals(const ControlPoints& controls, const ParametricDemo& renderer,
                                                const SrafRegion& region);

struct SrafPointPair {
    bool valid = false; // 是否已通过点对几何筛选，联合更新跳过未通过的组。
    SrafShrinkPoint first; // 第一个端点的全局编号、本轮材料内法线及法线厚度。
    SrafShrinkPoint second; // 第二个端点，同组两端使用同一个位移标量，各自沿本点法线移动。
    Eigen::Vector2d section_center_yx = Eigen::Vector2d::Zero(); // 点对中点，整图零基 (y,x) 浮点像素坐标。
    double section_diameter_px = 0.0; // 本轮本块最大内切圆直径，单位 pixel，用来排除细长图形的长轴。
    double width_px = 0.0; // 该截面附近两侧控制点之间的距离，单位 pixel。
};

// 输入完整 CR 控制点及其渲染器，按闭合轮廓包含关系输出各 SRAF 的归属和初始覆盖率。
// 一块由外轮廓和一层孔洞组成，所有编号沿用原输入顺序，多层包含输入抛出 invalid_argument。
std::vector<SrafRegion> group_sraf_regions(const ControlPoints& controls, const ParametricDemo& renderer);

// 直接检查完整空间像 AI>threshold，再按初始材料覆盖匹配未完成的 SRAF，不接收 WI 或显影标记图。
// background_window 为同尺寸 0/255 检测范围，非零允许检测；坐标均为整图零基 (y,x) 像素索引。
// 返回峰值最高且可归属的 regions 零基下标；无超阈值归属时返回 -1。
int select_printing_sraf_from_aerial(const std::vector<SrafRegion>& regions, const Eigen::MatrixXd& aerial,
                                    double threshold, const cv::Mat& background_window, const std::vector<bool>& finished);

struct SrafPairGroupConfig {
    int maximum_pairs = 4; // 一轮最多联合更新的点对数，每对两个点。
    double minimum_center_spacing_px = 1.0; // 中点距离小于此值且轴向夹角小于 30 度时视为重复，单位 pixel。
};

// 在同一块 SRAF 内重新拟合、检测多组宽点对；结果不共享控制点，并排除相近、近同向的重复截面。
// 根据各点局部法线厚度及当前最宽截面的尺度筛选候选，法线由本轮周期 CR 的解析切线求得。
std::vector<SrafPointPair> find_sraf_pair_groups(const ControlPoints& controls, const ParametricDemo& renderer,
                                                const SrafRegion& region, const SrafShrinkConfig& geometry,
                                                const SrafPairGroupConfig& groups);

// 输入完整 controls、已选点对及逐组位移，输出完整候选和去重后的移动记录，坐标与位移单位 pixel。
// 所有点对基于同一份 controls 同时更新，steps_px 与 pairs 一一对应，0 表示暂缓该组。
// 若外部调用传入共享控制点，则只移动一次并采用较小的正位移，防止重复累加。
SrafShrinkProposal propose_sraf_group_shrink(const ControlPoints& controls, const std::vector<SrafPointPair>& pairs,
                                            const std::vector<double>& steps_px);

} // namespace litho
