#include "partition_visualization.h"

#include <fstream>
#include <iomanip>

namespace litho {

namespace {

void write_rect(std::ostream& output, const PartitionRect& rect) {
    output << ',' << rect.y0 << ',' << rect.y1 << ',' << rect.x0 << ',' << rect.x1;
}

} // namespace

void save_partition_geometry(const PartitionMeta& meta, double pixel_size_nm, const std::filesystem::path& directory) {
    std::filesystem::create_directories(directory);
    std::ofstream sampling(directory / "partition_meta.csv");
    sampling << "height,width,grid_rows,grid_cols,halo,guard,tile_size,pixel_size_nm\n" << std::setprecision(17);
    sampling << meta.height << ',' << meta.width << ',' << meta.grid_rows << ',' << meta.grid_cols << ','
             << meta.halo << ',' << meta.guard << ',' << meta.tile_size << ',' << pixel_size_nm << '\n';
    std::ofstream regions(directory / "micro_regions.csv");
    regions << "micro_id,row,col,core_y0,core_y1,core_x0,core_x1,update_y0,update_y1,update_x0,update_x1,"
            << "tile_y0,tile_y1,tile_x0,tile_x1\n";
    for (const auto& tile : meta.tiles) {
        regions << tile.id << ',' << tile.row << ',' << tile.col;
        write_rect(regions, tile.core_global);
        // update_local 属于局部矩阵；加 tile_global 的起点，得到同一更新窗口的全局像素坐标。
        write_rect(regions, tile.update_local.shifted(tile.tile_global.y0, tile.tile_global.x0));
        // 直接导出实际 tile 范围，保留非整除/非方形分区为统一偶数网格增加的上下文像素。
        write_rect(regions, tile.tile_global);
        regions << '\n';
    }
}

} // namespace litho
