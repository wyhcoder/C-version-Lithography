"""绘制 C++ 全芯片 LSM 实际使用的全局 EP，保存全图及局部放大图。"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


def plot_eps(directory: Path) -> None:
    """读取 target.bmp、ep_points.csv 和 core_regions.csv；所有绘图坐标为全局像素索引。"""
    font_path = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if font_path.exists():
        plt.rcParams["font.family"] = FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    target = np.asarray(Image.open(directory / "target.bmp").convert("L")) / 255.0
    with (directory / "ep_points.csv").open() as file:
        points = list(csv.DictReader(file))
    with (directory / "core_regions.csv").open() as file:
        regions = list(csv.DictReader(file))
    # 读取实际生效的 EPE 权重；epe_all_eps 开启时全部为 1，关闭时仍可显示原来的零权重点。
    y = np.array([float(row["y"]) for row in points])
    x = np.array([float(row["x"]) for row in points])
    active = np.array([float(row["weight_epe"]) != 0.0 for row in points], dtype=bool)
    height, width = target.shape
    cy, cx = height // 2, width // 2
    if points:
        nearest = np.argmin((y - cy) ** 2 + (x - cx) ** 2)
        cy, cx = int(y[nearest]), int(x[nearest])
    # 放大窗口靠近中心 EP，便于检查中点位置；显示全局坐标，不把窗口左上角重新标成 (0,0)。
    side = min(128, height, width)
    y0 = min(max(0, cy - side // 2), height - side)
    x0 = min(max(0, cx - side // 2), width - side)
    views = [("", (0, height, 0, width)), ("_zoom", (y0, y0 + side, x0, x0 + side))]
    for suffix, (top, bottom, left, right) in views:
        fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
        ax.imshow(target[top:bottom, left:right], cmap="gray", vmin=0, vmax=1,
                  extent=(left - 0.5, right - 0.5, bottom - 0.5, top - 0.5))
        visible = (y >= top) & (y < bottom) & (x >= left) & (x < right)
        for mask, color, label in [(active, "red", "EP（EPE 梯度权重=1）"), (~active, "dodgerblue", "EP（EPE 梯度权重=0）")]:
            selected = visible & mask
            if selected.any():
                ax.scatter(x[selected], y[selected], s=25 if suffix else 8, c=color, edgecolors="white", linewidths=0.25,
                           label=f"{label}，当前视图 {selected.sum()} 点", zorder=3)
        # Core 边界位于相邻像素中心之间；矩形 [y0,y1) 的像素外缘为 [y0-0.5,y1-0.5]。
        for row in regions:
            gy0, gy1, gx0, gx1 = (int(row[key]) for key in ("y0", "y1", "x0", "x1"))
            ax.add_patch(Rectangle((gx0 - 0.5, gy0 - 0.5), gx1 - gx0, gy1 - gy0,
                                   fill=False, edgecolor="cyan", linewidth=0.5, alpha=0.7))
            center_y, center_x = (gy0 + gy1 - 1) / 2, (gx0 + gx1 - 1) / 2
            if top <= center_y < bottom and left <= center_x < right:
                ax.text(center_x, center_y, f'Micro {row["micro_id"]}', color="cyan", ha="center", fontsize=7, alpha=0.8)
        if suffix and visible.sum() <= 32:
            for i in np.flatnonzero(visible):
                ax.annotate(f"({int(y[i])},{int(x[i])})", (x[i], y[i]), xytext=(4, 4),
                            textcoords="offset points", color="yellow", fontsize=6)
        ax.set_xlim(left - 0.5, right - 0.5)
        ax.set_ylim(bottom - 0.5, top - 0.5)
        ax.set_xlabel("全局 x：列像素索引")
        ax.set_ylabel("全局 y：行像素索引")
        ax.set_title(f"原始 target 上的 EP 选点 · 全图 {len(points)} 点\n青色线为不重叠 Core 分区")
        if visible.any():
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=9)
        fig.savefig(directory / f"ep_points{suffix}.png", dpi=220)
        plt.close(fig)
    print(f"Saved EP figures to {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="C++ 导出的 LSM 结果目录")
    plot_eps(parser.parse_args().directory)
