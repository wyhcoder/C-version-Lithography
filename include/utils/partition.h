#pragma once

#include <Eigen/Dense>
#include <vector>

namespace litho {

// 坐标顺序为 (y, x)，范围统一为 [y0, y1) × [x0, x1)。
struct PartitionRect {
    int y0; // 起始行索引，包含该行。
    int y1; // 结束行索引，不包含该行。
    int x0; // 起始列索引，包含该列。
    int x1; // 结束列索引，不包含该列。

    int rows() const; // 矩形的像素行数。
    int cols() const; // 矩形的像素列数。
    PartitionRect shifted(int dy, int dx) const; // 将四条边界分别平移 dy 行、dx 列。
    PartitionRect expanded(int width) const; // 四侧各向外扩展 width 个像素。

    // 返回两个矩形共同覆盖的区域。
    static PartitionRect intersection(const PartitionRect& a, const PartitionRect& b);
};

enum class NeighborDirection { Left, Right, Top, Bottom };

struct MicroNeighbor {
    int id;
    NeighborDirection direction;

    // 本块 Guard ← 邻居 Core：两套局部坐标对应同一个 guard_global 区域。
    PartitionRect guard_global;
    PartitionRect my_guard_local;
    PartitionRect neighbor_overlap_local;

    // 本块 Core → 邻居 Guard：两套局部坐标对应同一个 overlap_global 区域。
    PartitionRect overlap_global;
    PartitionRect my_overlap_local;
    PartitionRect neighbor_guard_local;
};

struct MicroTile {
    int id;    // Micro 编号：row * grid_cols + col，也是 tiles/locals 数组的下标。
    int row;   // 本块在分块网格中的行号，从 0 开始；不是像素行坐标。
    int col;   // 本块在分块网格中的列号，从 0 开始；不是像素列坐标。
    int halo;  // Core 每侧的成像上下文扩展宽度，单位：像素。
    int guard; // Core 每侧的边界协调带宽度，单位：像素，位于 Halo 内。

    PartitionRect core_global;  // 本块拥有的全局区域，最终只回拼这里。
    PartitionRect tile_global;  // 含 Halo 的计算区域，可越过全图边界。
    PartitionRect core_local;   // Core 在局部矩阵中的坐标。
    PartitionRect update_local; // Core + Guard 的更新窗口。
    std::vector<MicroNeighbor> neighbors; // 实际存在的左、右、上、下邻居及其双向交换区域。
};

// 描述整张版图的划分；只保存几何信息，不保存 mask、梯度或水平集状态。
struct PartitionMeta {
    int height;    // 全局版图的像素行数（y 方向），对应 global.rows()。
    int width;     // 全局版图的像素列数（x 方向），对应 global.cols()。
    int grid_rows; // y 方向分成多少行 Micro；表示块数，不是像素数。
    int grid_cols; // x 方向分成多少列 Micro；Micro 总数为 grid_rows * grid_cols。
    int halo;      // Core 每侧向外扩展的成像上下文宽度，单位：像素。
    int guard;     // Core 每侧向外扩展的协调/更新带宽度，单位：像素。
    int tile_size; // 每块局部矩阵的统一边长：最大 Core 边长 + 2 * halo，再向上取偶数。
    std::vector<MicroTile> tiles; // 按行优先存储所有 Micro，tiles[id] 即对应块的几何信息。
};

class Partition {
public:
    // 输入 height/width 为全图像素尺寸，grid_rows/grid_cols 为分块行列数，halo/guard 为每侧扩展像素数。
    // 例如 build(1024, 1024, 4, 4, 128, 30)：16 块、Core 为 256×256、局部矩阵为 512×512。
    // 返回所有块的坐标和邻接关系；本函数不提取矩阵数据，提取由 extract_all 完成。
    // 调用约定：每行/列至少有一个 Core 像素；0 <= guard <= halo，且 guard 不超过最小 Core 边长。
    // 余数由整数分界线分配，支持不能整除和非正方形的大版图。
    static PartitionMeta build(int height, int width, int grid_rows, int grid_cols, int halo, int guard);

    // 与 Python 实现一致：超出芯片的 Halo 从全图另一侧周期取值。
    static Eigen::MatrixXd extract_local_tile(const Eigen::MatrixXd& global, const MicroTile& tile);

    // 数组下标就是 Micro id；每项保存完整的局部矩阵（含 Halo）。
    static std::vector<Eigen::MatrixXd> extract_all(const Eigen::MatrixXd& global, const PartitionMeta& meta);
    // 将所有局部矩阵拼接回全局矩阵；仅拼接 Core 区域，忽略 Halo。
    static Eigen::MatrixXd assemble_global(const std::vector<Eigen::MatrixXd>& locals, const PartitionMeta& meta);

    // 1 表示允许更新，0 表示仅提供成像上下文；芯片外侧的周期延拓也落在此局部窗口中。
    static Eigen::MatrixXd update_window(const MicroTile& tile);
};

} // namespace litho
