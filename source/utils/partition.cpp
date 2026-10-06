#include "partition.h"

#include <algorithm>
#include <array>

namespace litho {

int PartitionRect::rows() const { return y1 - y0; }

int PartitionRect::cols() const { return x1 - x0; }

PartitionRect PartitionRect::shifted(int dy, int dx) const { return {y0 + dy, y1 + dy, x0 + dx, x1 + dx}; }

PartitionRect PartitionRect::expanded(int width) const { return {y0 - width, y1 + width, x0 - width, x1 + width}; }

PartitionRect PartitionRect::intersection(const PartitionRect& a, const PartitionRect& b) {
    return {std::max(a.y0, b.y0), std::min(a.y1, b.y1), std::max(a.x0, b.x0), std::min(a.x1, b.x1)};
}

// 先生成全部 Micro 的几何区域，再根据分块行列号建立四邻域的双向切片。
PartitionMeta Partition::build(int height, int width, int grid_rows, int grid_cols, int halo, int guard) {
    PartitionMeta meta;
    meta.height = height;
    meta.width = width;
    meta.grid_rows = grid_rows;
    meta.grid_cols = grid_cols;
    meta.halo = halo;
    meta.guard = guard;
    const int max_core_rows = (height + grid_rows - 1) / grid_rows; // 向上取整，得到各 Core 中最大的像素行数。
    const int max_core_cols = (width + grid_cols - 1) / grid_cols;  // 向上取整，得到各 Core 中最大的像素列数。
    meta.tile_size = std::max(max_core_rows, max_core_cols) + 2 * halo;
    meta.tile_size += meta.tile_size % 2;
    meta.tiles.reserve(grid_rows * grid_cols);

    for (int row = 0; row < grid_rows; ++row) {
        for (int col = 0; col < grid_cols; ++col) {
            MicroTile tile;
            tile.id = row * grid_cols + col;
            tile.row = row;
            tile.col = col;
            tile.halo = halo;
            tile.guard = guard;
            tile.core_global = {row * height / grid_rows, (row + 1) * height / grid_rows,
                                col * width / grid_cols, (col + 1) * width / grid_cols};
            const int origin_y = tile.core_global.y0 - halo;
            const int origin_x = tile.core_global.x0 - halo; // halo区的起始索引
            tile.tile_global = {origin_y, origin_y + meta.tile_size, origin_x, origin_x + meta.tile_size};
            tile.core_local = tile.core_global.shifted(-origin_y, -origin_x); // Core 在局部矩阵中的坐标
            tile.update_local = tile.core_local.expanded(guard); // Core + Guard 的更新窗口
            meta.tiles.push_back(tile);
        }
    }

    // 按左、右、上、下排列；dr 改变分块行号，dc 改变分块列号。
    constexpr std::array<int, 4> dr = {0, 0, -1, 1};
    constexpr std::array<int, 4> dc = {-1, 1, 0, 0};
    constexpr std::array<NeighborDirection, 4> directions = {
        NeighborDirection::Left, NeighborDirection::Right, NeighborDirection::Top, NeighborDirection::Bottom
    };
    for (MicroTile& tile : meta.tiles) {
        for (int i = 0; i < 4; ++i) {
            const int row = tile.row + dr[i];
            const int col = tile.col + dc[i];
            // 芯片外侧没有邻居；Halo 的周期回绕不改变邻接关系。
            if (row < 0 || row >= grid_rows || col < 0 || col >= grid_cols) continue;
            const MicroTile& neighbor = meta.tiles[row * grid_cols + col];
            MicroNeighbor edge;
            edge.id = neighbor.id; // 邻居的 Micro 编号
            edge.direction = directions[i]; // 该邻居在本块的哪个方向
            edge.guard_global = PartitionRect::intersection(tile.core_global.expanded(guard), neighbor.core_global); // 本块 Guard的全局索引
            edge.my_guard_local = edge.guard_global.shifted(-tile.tile_global.y0, -tile.tile_global.x0); // 本块 Guard在本块局部矩阵的索引
            edge.neighbor_overlap_local = edge.guard_global.shifted(-neighbor.tile_global.y0, -neighbor.tile_global.x0); // 本块 Guard在邻居局部矩阵的索引
            edge.overlap_global = PartitionRect::intersection(tile.core_global, neighbor.core_global.expanded(guard)); // 本块 Core的全局索引
            edge.my_overlap_local = edge.overlap_global.shifted(-tile.tile_global.y0, -tile.tile_global.x0); // 本块 Core在本块局部矩阵的索引
            edge.neighbor_guard_local = edge.overlap_global.shifted(-neighbor.tile_global.y0, -neighbor.tile_global.x0); // 本块 Core在邻居局部矩阵的索引
            tile.neighbors.push_back(edge);
        }
    }
    return meta;
}

// 局部索引先转换为全局坐标，再周期映射回全图范围，填充含 Halo 的局部矩阵。公式为 (x % n + n) % n, 保证负数也能映射回正数。
Eigen::MatrixXd Partition::extract_local_tile(const Eigen::MatrixXd& global, const MicroTile& tile) {
    Eigen::MatrixXd local(tile.tile_global.rows(), tile.tile_global.cols());
    const int height = static_cast<int>(global.rows());
    const int width = static_cast<int>(global.cols());
    for (int y = 0; y < local.rows(); ++y) {
        const int gy = ((tile.tile_global.y0 + y) % height + height) % height;
        for (int x = 0; x < local.cols(); ++x) {
            const int gx = ((tile.tile_global.x0 + x) % width + width) % width;
            local(y, x) = global(gy, gx);
        }
    }
    return local;
}

std::vector<Eigen::MatrixXd> Partition::extract_all(const Eigen::MatrixXd& global, const PartitionMeta& meta) {
    std::vector<Eigen::MatrixXd> locals;
    locals.reserve(meta.tiles.size());
    for (const MicroTile& tile : meta.tiles) locals.push_back(extract_local_tile(global, tile));
    return locals;
}

// 每个 Core 只写回它拥有的全局位置，Halo/Guard 不重复覆盖全图。
Eigen::MatrixXd Partition::assemble_global(const std::vector<Eigen::MatrixXd>& locals, const PartitionMeta& meta) {
    Eigen::MatrixXd global(meta.height, meta.width);
    for (const MicroTile& tile : meta.tiles) {
        const PartitionRect& dst = tile.core_global;
        const PartitionRect& src = tile.core_local;
        global.block(dst.y0, dst.x0, dst.rows(), dst.cols()) = locals[tile.id].block(src.y0, src.x0, src.rows(), src.cols());
    }
    return global;
}

Eigen::MatrixXd Partition::update_window(const MicroTile& tile) {
    Eigen::MatrixXd window = Eigen::MatrixXd::Zero(tile.tile_global.rows(), tile.tile_global.cols());
    const PartitionRect& rect = tile.update_local;
    window.block(rect.y0, rect.x0, rect.rows(), rect.cols()).setOnes();
    return window;
}

} // namespace litho
