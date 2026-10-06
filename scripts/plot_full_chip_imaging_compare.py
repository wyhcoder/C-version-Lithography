"""比较原始 target 和 CTM/LSM 最佳版图在同一套全图光学模型下的成像。"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


def plot_comparison(directory: Path) -> None:
    # 使用本机中文字体；所有数值读取 double TXT，BMP 的 8 位量化不参与评价。
    font_path = Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf")
    if font_path.exists():
        plt.rcParams["font.family"] = FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    with (directory / "metrics.csv").open() as file:
        metrics = {row["case"]: row for row in csv.DictReader(file)}
    with (directory / "sampling.csv").open() as file:
        sampling = next(csv.DictReader(file))
    # 导出指标中的 case 决定阶段；LSM 直接显示最佳二值 mask，AI/WI 始终是原始连续计算值。
    is_lsm = "optimized_lsm" in metrics
    optimized_case = "optimized_lsm" if is_lsm else "optimized_gray"
    optimized_label = "LSM 优化版图" if is_lsm else "CTM 优化灰度版图"
    figure_prefix = "lsm_comparison" if is_lsm else "gray_ctm_comparison"
    stage_label = "LSM 优化" if is_lsm else "CTM 优化灰度"
    rows = [
        ("target", "未优化原始 target", ["target_mask", "target_ai", "target_wi"]),
        (optimized_case, optimized_label, [f"{optimized_case}_mask", "optimized_ai", "optimized_wi"]),
    ]
    matrices = {name: [np.loadtxt(directory / f"{key}.txt") for key in keys] for name, _, keys in rows}
    n = matrices["target"][0].shape[0]
    ai_max = max(values[1].max() for values in matrices.values())
    spatial = f'{sampling["grid_size"]}×{sampling["grid_size"]}，{sampling["pixel_size_nm"]} nm/像素'

    # 固定中心 256×256 像素窗口；全图与放大图用相同坐标、色标，便于比较边缘。
    side = min(256, n)
    start = (n - side) // 2
    views = [("", (slice(None), slice(None))), ("_zoom", (slice(start, start + side), slice(start, start + side)))]
    for suffix, view in views:
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        for i, (name, label, _) in enumerate(rows):
            for j, matrix in enumerate(matrices[name]):
                ax = axes[i, j]
                im = ax.imshow(matrix[view], cmap="gray", vmin=0, vmax=ai_max if j == 1 else 1)
                ax.set_title(["输入 mask", "全图空中像 AI", "全图连续显影 WI"][j], fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])
                fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
            # 两个 PE 都以原始 target 为目标；放大图仍显示整张全图的 PE。
            axes[i, 0].set_ylabel(f'{label}\n全图 PE={float(metrics[name]["pe"]):.2f}', fontsize=12)
        crop = f"；中心窗口 [{start},{start + side})" if suffix else ""
        fig.suptitle(f"同一全图光学模型：原始 target 与 {stage_label}成像 · {spatial}{crop}", fontsize=14)
        fig.savefig(directory / f"{figure_prefix}{suffix}.png", dpi=200)
        plt.close(fig)
    print(f"Saved comparison figures to {directory}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="C++ 导出的 imaging_comparison 目录")
    plot_comparison(parser.parse_args().directory)
