"""在同一张完整 target 上显示 C++ 导出的 Micro Core/Guard/Halo 分区。"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


COLORS = {"core": "#36df87", "guard": "#ff9a32", "halo": "#4d9eff"}


def rect(row: dict, prefix: str) -> tuple[int, int, int, int]:
    """CSV 矩形顺序是全局像素 (y0,y1,x0,x1)，区间不包含结束索引。"""
    return tuple(row[f"{prefix}_{key}"] for key in ("y0", "y1", "x0", "x1"))


def add_rect(ax, region, **style) -> None:
    y0, y1, x0, x1 = region
    # 索引 [x0,x1) 覆盖的像素外缘是 [x0-0.5,x1-0.5]，边界画在两个像素中心之间。
    ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, **style))


def fill_ring(ax, outer, inner, color: str) -> None:
    """只填充外矩形减去内矩形的环带，Guard 与 Halo 其余部分采用互不覆盖的颜色。"""
    y0, y1, x0, x1 = outer
    cy0, cy1, cx0, cx1 = inner
    for region in [(y0, cy0, x0, x1), (cy1, y1, x0, x1), (cy0, cy1, x0, cx0), (cy0, cy1, cx1, x1)]:
        if region[1] > region[0] and region[3] > region[2]:
            add_rect(ax, region, facecolor=color, edgecolor="none", alpha=0.28, zorder=2)


def plot_partition(directory: Path, micro_id: int | None = None) -> None:
    """只消费真实 C++ 分区坐标，不在 Python 中重新计算划分或邻居几何。"""
    font_path = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if font_path.exists():
        plt.rcParams["font.family"] = FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    with (directory / "partition_meta.csv").open() as file:
        meta = next(csv.DictReader(file))
    with (directory / "micro_regions.csv").open() as file:
        regions = [{key: int(value) for key, value in row.items()} for row in csv.DictReader(file)]
    target = np.asarray(Image.open(directory / "target.bmp").convert("L"))
    height, width = target.shape
    if micro_id is None:
        micro_id = (int(meta["grid_rows"]) // 2) * int(meta["grid_cols"]) + int(meta["grid_cols"]) // 2
    selected = next(row for row in regions if row["micro_id"] == micro_id)
    # 两张图都保留整张芯片视野；外侧额外留出越界 Halo，坐标仍是未经周期回绕的全局索引。
    ymin = min(row["tile_y0"] for row in regions)
    ymax = max(row["tile_y1"] for row in regions)
    xmin = min(row["tile_x0"] for row in regions)
    xmax = max(row["tile_x1"] for row in regions)
    margin = max(height, width) * 0.01
    handles = [
        Line2D([0], [0], color=COLORS["core"], lw=2, label="Core 边界"),
        Line2D([0], [0], color=COLORS["guard"], lw=2, ls="--", label="Core＋Guard 外边界"),
        Line2D([0], [0], color=COLORS["halo"], lw=2, ls=(0, (5, 3)), label="Core＋Halo 外边界"),
    ]
    sampling = f'{height}×{width} 像素 · {meta["grid_rows"]}×{meta["grid_cols"]} Micro · {meta["pixel_size_nm"]} nm/像素'
    for highlight in (False, True):
        fig, ax = plt.subplots(figsize=(11, 12))
        fig.subplots_adjust(left=0.09, right=0.96, bottom=0.18, top=0.88)
        ax.set_facecolor("#e8edf4")
        ax.imshow(target, cmap="gray", vmin=0, vmax=255, extent=(-0.5, width - 0.5, height - 0.5, -0.5), zorder=0)
        if highlight:
            fill_ring(ax, rect(selected, "tile"), rect(selected, "update"), COLORS["halo"])
            fill_ring(ax, rect(selected, "update"), rect(selected, "core"), COLORS["guard"])
            add_rect(ax, rect(selected, "core"), facecolor=COLORS["core"], edgecolor="none", alpha=0.18, zorder=2)
        for row in regions:
            if not highlight:
                add_rect(ax, rect(row, "tile"), fill=False, edgecolor=COLORS["halo"], ls=(0, (5, 3)), lw=0.6, alpha=0.55)
                add_rect(ax, rect(row, "update"), fill=False, edgecolor=COLORS["guard"], ls="--", lw=0.7, alpha=0.7)
            add_rect(ax, rect(row, "core"), fill=False, edgecolor=COLORS["core"], lw=0.8, alpha=0.5 if highlight else 0.9)
            y0, y1, x0, x1 = rect(row, "core")
            # 数字是 Micro id；与像素坐标、分块行列号区分开。
            ax.text((x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2, str(row["micro_id"]), ha="center", va="center",
                    fontsize=9, color="white", bbox={"facecolor": "black", "alpha": 0.65, "edgecolor": "none", "pad": 2}, zorder=5)
        if highlight:
            for prefix, color, style in [("tile", "halo", (0, (5, 3))), ("update", "guard", "--"), ("core", "core", "-")]:
                add_rect(ax, rect(selected, prefix), fill=False, edgecolor=COLORS[color], ls=style, lw=2.5, zorder=4)
        # 灰色外侧不是额外 Micro；这里仅展示芯片边缘的周期 Halo 几何。
        add_rect(ax, (0, height, 0, width), fill=False, edgecolor="#bac5d4", lw=1.5, zorder=3)
        ax.set_xlim(xmin - 0.5 - margin, xmax - 0.5 + margin)
        ax.set_ylim(ymax - 0.5 + margin, ymin - 0.5 - margin)
        ax.set_aspect("equal")
        ax.set_xlabel("全局 x：列像素索引")
        ax.set_ylabel("全局 y：行像素索引")
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.075), ncol=3, frameon=False, fontsize=10)
        if highlight:
            suffix = f"_micro_{micro_id}"
            fig.suptitle(f"完整 target 上高亮 Micro {micro_id}（分块行 {selected['row']}，列 {selected['col']}）\n{sampling}", y=0.95, fontsize=16)
            bounds = []
            for prefix, label in [("core", "Core"), ("update", "Core＋Guard"), ("tile", "Core＋Halo")]:
                y0, y1, x0, x1 = rect(selected, prefix)
                bounds.append(f"{label}：y[{y0},{y1}) × x[{x0},{x1})")
            footer = "\n".join(bounds) + "\n橙色环带为 Guard；蓝色环带为 Guard 之外的 Halo；Guard 本身也属于 Halo。"
        else:
            suffix = ""
            fig.suptitle(f"完整 target 上的全部 Micro 分区（数字为 Micro id）\n{sampling}", y=0.95, fontsize=16)
            footer = (f'每侧 Halo={meta["halo"]} 像素，Guard={meta["guard"]} 像素；局部成像网格={meta["tile_size"]}×{meta["tile_size"]}\n'
                      "Core 不重叠；各块 Guard/Halo 可以重叠。越过芯片边缘的 Halo 在实际提取时周期取值。")
        fig.text(0.5, 0.025, footer, ha="center", va="bottom", fontsize=10, linespacing=1.6)
        path = directory / f"partition{suffix}.png"
        fig.savefig(path, dpi=200, facecolor="white")
        plt.close(fig)
        print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="C++ 导出的分区数据目录")
    parser.add_argument("--micro-id", type=int, help="全图高亮的 Micro 编号，默认选网格中间的一块")
    args = parser.parse_args()
    plot_partition(args.directory, args.micro_id)
