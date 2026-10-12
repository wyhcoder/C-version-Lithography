#!/usr/bin/env python3
"""绘制 CTM 四组对照的接缝放大图；输入 demo 导出的全局 ROI 矩阵与 CSV，不重新成像。"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image


CASES = ("off", "fusion_only", "cept_only", "on")
LABELS = {"off": "Fusion OFF / CEPT OFF", "fusion_only": "Fusion ON / CEPT OFF",
          "cept_only": "Fusion OFF / CEPT ON", "on": "Fusion ON / CEPT ON"}
COLORS = {"off": "#626262", "fusion_only": "#247ab4", "cept_only": "#ca7529", "on": "#269260"}


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def crossing_edges(profile, threshold, y0, center):
    """沿全局 y 像素中心线性插值上下阈值边缘；选最靠近目标主线的区间，ROI 截断的边缘记为 NaN。"""
    high = profile >= threshold
    starts = np.flatnonzero(high & ~np.r_[False, high[:-1]])
    ends = np.flatnonzero(high & ~np.r_[high[1:], False])
    if starts.size == 0:
        return np.nan, np.nan
    index = np.argmin(np.abs(y0 + (starts + ends) / 2 - center))
    start, end = starts[index], ends[index]
    upper = lower = np.nan
    if start > 0:
        upper = y0 + start - 1 + (threshold - profile[start - 1]) / (profile[start] - profile[start - 1])
    if end + 1 < profile.size:
        lower = y0 + end + (threshold - profile[end]) / (profile[end + 1] - profile[end])
    return upper, lower


def continuity_row(case, snapshot, iteration, fields, target, settings):
    """比較接缝左右相邻列的主线边缘；单位为像素，nm 值再乘实际 pixel_size_nm。"""
    y0, x0, seam = (int(settings[key]) for key in ("y0", "x0", "seam_x"))
    left, right = seam - 1 - x0, seam - x0
    foreground_y = np.nonzero(target[:, [left, right]] >= 0.5)[0]
    center = y0 + foreground_y.mean()
    row = {"case": case, "snapshot": snapshot, "iteration": iteration}
    for name, threshold in (("mask", 0.5), ("ai", float(settings["ai_threshold"]))):
        a = crossing_edges(fields[name][:, left], threshold, y0, center)
        b = crossing_edges(fields[name][:, right], threshold, y0, center)
        row.update({f"{name}_upper_left_y": a[0], f"{name}_lower_left_y": a[1],
                    f"{name}_upper_right_y": b[0], f"{name}_lower_right_y": b[1]})
        gaps = np.abs(np.asarray(a) - np.asarray(b))
        # 缺失任一侧轮廓时不把缺失当作连续；CSV 保留 nan，found=0。
        found = bool(np.isfinite(gaps).all())
        row[f"{name}_contour_found"] = int(found)
        row[f"{name}_upper_gap_px"] = gaps[0]
        row[f"{name}_lower_gap_px"] = gaps[1]
        row[f"{name}_gap_max_px"] = gaps.max() if found else np.nan
        row[f"{name}_gap_max_nm"] = row[f"{name}_gap_max_px"] * float(settings["pixel_size_nm"])
    return row


def decorate_axis(ax, field, target, settings, mode, ai_max):
    """像素中心使用全局索引，像素外缘为 ±0.5；seam_x 是右块首列，接缝位于 seam_x-0.5。"""
    y0, y1, x0, x1, seam = (int(settings[key]) for key in ("y0", "y1", "x0", "x1", "seam_x"))
    xx, yy = np.arange(x0, x1), np.arange(y0, y1)
    image = field[mode]
    ax.imshow(image, origin="upper", extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5),
              cmap="gray", vmin=0, vmax=1 if mode == "mask" else ai_max, interpolation="nearest")
    ax.contour(xx, yy, target, levels=[0.5], colors=["#37cde6"], linewidths=1.1)
    if mode == "ai":
        threshold = float(settings["ai_threshold"])
        if image.min() < threshold < image.max():
            ax.contour(xx, yy, image, levels=[threshold], colors=["#ff9952"], linewidths=1.3)
    ax.axvline(seam - 0.5, color="#f0df26", linestyle="--", linewidth=1.1)
    ax.set_xlim(x0 - 0.5, x1 - 0.5)
    ax.set_ylim(y1 - 0.5, y0 - 0.5)
    ax.set_xlabel("Global x (pixel index)")
    ax.set_ylabel("Global y (pixel index)")
    ax.tick_params(labelsize=9)


def legend_handles(include_ai):
    handles = [Line2D([], [], color="#37cde6", label="Target edge (0.5)"),
               Line2D([], [], color="#c0a800", linestyle="--", label="Core seam")]
    if include_ai:
        handles.append(Line2D([], [], color="#ff9952", label="AI threshold contour"))
    return handles


def gap_label(row, mode):
    value = row[f"{mode}_gap_max_px"]
    return f"max edge gap {value:.3f} px" if np.isfinite(value) else "edge contour missing/clipped"


def plot_grid(output, snapshot, fields, rows, target, settings, mode, ai_max):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
    for ax, case in zip(axes.flat, CASES):
        row = rows[(case, snapshot)]
        decorate_axis(ax, fields[(case, snapshot)], target, settings, mode, ai_max)
        ax.set_title(f"{LABELS[case]}\niter {row['iteration']} | {gap_label(row, mode)}", fontsize=11)
    subject = "Gray mask" if mode == "mask" else f"AI, threshold = {float(settings['ai_threshold']):g}"
    selection = "same update count" if snapshot == "final" else "each case's best total-loss state"
    fig.suptitle(f"CTM {subject}: {snapshot} ({selection})", fontsize=14)
    fig.legend(handles=legend_handles(mode == "ai"), loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.12, top=0.87, hspace=0.65, wspace=0.23)
    fig.savefig(output / f"{snapshot}_{mode}_comparison.png", dpi=200)
    plt.close(fig)


def plot_on_off(output, snapshot, fields, rows, target, settings, ai_max):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
    for column, case in enumerate(("on", "off")):
        row = rows[(case, snapshot)]
        for line, mode in enumerate(("mask", "ai")):
            decorate_axis(axes[line, column], fields[(case, snapshot)], target, settings, mode, ai_max)
            subject = "gray mask" if mode == "mask" else "AI threshold contour"
            axes[line, column].set_title(f"{case.upper()} {subject}\niter {row['iteration']} | {gap_label(row, mode)}", fontsize=11)
    fig.suptitle(f"CTM ON/OFF: {snapshot}; same ROI, same image scales", fontsize=14)
    fig.legend(handles=legend_handles(True), loc="lower center", ncol=3, frameon=False)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.12, top=0.87, hspace=0.65, wspace=0.23)
    fig.savefig(output / f"{snapshot}_on_off.png", dpi=200)
    plt.close(fig)


def write_report(output, metrics, rows, settings):
    """保存两种选轮方式的实际结果表；共同 Core PE 与额外 CEPT 项分别记录。"""
    iterations = metrics[("off", "final")]["iteration"]
    geometry = read_csv(output / "partition_meta.csv")[0]
    descriptions = {"off": "均关闭", "fusion_only": "仅梯度融合", "cept_only": "仅 CEPT", "on": "均开启"}
    lines = ["# CTM 接缝对照结果", "",
             f"四组各完成 {iterations} 次更新，目标尺寸为 {geometry['height']}×{geometry['width']}，"
             f"分区 {geometry['grid_rows']}×{geometry['grid_cols']}，Halo {geometry['halo']} px、Guard {geometry['guard']} px、"
             f"局部光学网格 {geometry['tile_size']}×{geometry['tile_size']}。共同数据目标为 Core PE，权重与步长见各组实际配置。", "",
             f"固定窗口：y∈[{settings['y0']},{settings['y1']})，x∈[{settings['x0']},{settings['x1']})；"
             f"接缝 x={int(settings['seam_x']) - 0.5:g}，像素间距 {settings['pixel_size_nm']} nm。", "",
             "[ON/OFF 放大图](final_on_off.png) · [四组 mask](final_mask_comparison.png) · "
             "[四组成像轮廓](final_ai_comparison.png) · [全图窗口位置](roi_location.png) · [Core PE 曲线](core_pe_history.png)", ""]
    for snapshot, heading in (("final", "相同更新次数的主对照"), ("best", "各组最佳总损失轮，供补充观察")):
        lines += [f"## {heading}", "",
                  "| 组别 | 轮次 | Core PE | Guard–Core RMSE | 前景接缝灰度跳变 MAE | mask 最大边缘错位 / px | AI 最大边缘错位 / px |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for case in CASES:
            metric, row = metrics[(case, snapshot)], rows[(case, snapshot)]
            def number(value):
                return f"{value:.6f}" if np.isfinite(value) else "轮廓缺失/截断"
            lines.append(f"| {descriptions[case]} | {metric['iteration']} | {float(metric['core_pe']):.3f} | "
                         f"{float(metric['overlap_rmse']):.6f} | {float(metric['foreground_seam_jump_mae']):.6f} | "
                         f"{number(row['mask_gap_max_px'])} | {number(row['ai_gap_max_px'])} |")
        lines.append("")
    lines += ["Guard–Core RMSE 使用双精度完整局部 mask 的同位置像素差，包含有向四邻域双方，不包含角区；"
              "关闭 CEPT 的组同样独立计算。前景接缝跳变只取 target 两侧均为前景的相邻 Core 像素。", "",
              "局部边缘错位为接缝左右两列主线的上下轮廓差中的较大值，沿 y 线性插值；"
              f"mask 阈值为 0.5，AI 阈值为 {settings['ai_threshold']}。乘 {settings['pixel_size_nm']} 可换算为 nm。", "",
              "最佳轮按各组自身 total_loss 选择，开启 CEPT 后含额外项，因此不能跨组直接比较最佳 total_loss。"
              "重叠区一致性与 Core 回拼连续性是不同指标，应结合上述图和表判断。", "",
              "原始数据：[全芯片统计](metrics.csv)、[窗口轮廓位置和错位](roi_continuity.csv)。"
              "各组目录另有实际配置、迭代历史、末轮和最佳轮灰度 mask。优化耗时记录在 metrics.csv，未包含光学准备和输出成像。", ""]
    (output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    output = Path(sys.argv[1]).resolve()
    settings = read_csv(output / "roi.csv")[0]
    metrics = {(row["case"], row["snapshot"]): row for row in read_csv(output / "metrics.csv")}
    target = np.loadtxt(output / "roi_target.txt")
    fields, rows = {}, {}
    for snapshot in ("final", "best"):
        for case in CASES:
            directory = output / case / snapshot
            values = {name: np.loadtxt(directory / f"roi_{name}.txt") for name in ("mask", "ai")}
            fields[(case, snapshot)] = values
            rows[(case, snapshot)] = continuity_row(case, snapshot, int(metrics[(case, snapshot)]["iteration"]),
                                                    values, target, settings)
    with (output / "roi_continuity.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(next(iter(rows.values()))))
        writer.writeheader()
        writer.writerows(rows.values())
    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})
    # AI 色阶跨四组和两个快照共用；mask 固定 [0,1]，避免各组自动色阶掩盖实际差异。
    ai_max = max(value["ai"].max() for value in fields.values())
    for snapshot in ("final", "best"):
        for mode in ("mask", "ai"):
            plot_grid(output, snapshot, fields, rows, target, settings, mode, ai_max)
        plot_on_off(output, snapshot, fields, rows, target, settings, ai_max)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for case in CASES:
        history = read_csv(output / case / "history.csv")
        ax.semilogy([int(row["iteration"]) for row in history], [float(row["data_loss"]) for row in history],
                    label=LABELS[case], color=COLORS[case], linewidth=1.4)
    ax.set(xlabel="CTM updates", ylabel="Core PE (sum)", title="Common data objective; CEPT excluded")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "core_pe_history.png", dpi=200)
    plt.close(fig)
    # 在整体 target 上标出固定窗口与 Core 分界，便于定位放大图对应的真实版图位置。
    image = np.asarray(Image.open(output / "target.bmp"))
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    regions = read_csv(output / "micro_regions.csv")
    for key, draw in (("core_x0", ax.axvline), ("core_y0", ax.axhline)):
        for boundary in sorted({int(row[key]) for row in regions} - {0}):
            draw(boundary - 0.5, color="#37cde6", linewidth=0.6, alpha=0.65)
    y0, y1, x0, x1 = (int(settings[key]) for key in ("y0", "y1", "x0", "x1"))
    ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False, edgecolor="#ff9952", linewidth=1.5))
    ax.annotate("Comparison ROI", xy=((x0 + x1) / 2, (y0 + y1) / 2), xytext=(x1 + 70, y0 - 50),
                color="#ff9952", arrowprops={"arrowstyle": "->", "color": "#ff9952"})
    ax.set(xlabel="Global x (pixel index)", ylabel="Global y (pixel index)", title="Target, Core boundaries and fixed ROI")
    fig.tight_layout()
    fig.savefig(output / "roi_location.png", dpi=200)
    plt.close(fig)
    write_report(output, metrics, rows, settings)
    print(f"Saved CTM comparison plots and ROI edge gaps to {output}")


if __name__ == "__main__":
    main()
