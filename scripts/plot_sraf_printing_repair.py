#!/usr/bin/env python3
"""读取真实连续 WI 绘制 comparison.png，并输出曲线及收敛诊断；坐标为整图零基 (y,x) pixel。"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache/matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from plot_meef_parametric_curves import close_curve, periodic_uniform_bspline


def load_controls(path: Path) -> list[np.ndarray]:
    """按 # 行和空行分组读取坐标，返回各轮廓的 N×2 数组，列顺序为整图 (y,x) pixel。"""
    groups: list[np.ndarray] = []
    current: list[list[float]] = []
    for line in path.read_text().splitlines() + [""]:
        if not line.strip() or line.lstrip().startswith("#"):
            if current:
                groups.append(np.asarray(current))
                current = []
        else:
            current.append([float(value) for value in line.split()])
    return groups


def load_metrics(path: Path) -> dict[str, float]:
    """读取每行 key/value 的指标文件，返回文件中记录的 pixel、强度或 nm 数值。"""
    metrics: dict[str, float] = {}
    for line in path.read_text().splitlines():
        key, value = line.split()
        metrics[key] = float(value)
    return metrics


def plot_initial_pair_groups(directory: Path, controls: list[np.ndarray], curves: list[np.ndarray], rows: list[dict[str, str]]) -> None:
    """画首次接受更新时的同块点对，连接线使用更新前的真实控制点位置。"""
    first_iteration = rows[0]["iteration"]
    first_groups = [row for row in rows if row["iteration"] == first_iteration]
    contour_ids = sorted({int(row[key]) for row in first_groups for key in ["first_contour", "second_contour"]})
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    for contour in contour_ids:
        curve, points = curves[contour], controls[contour]
        ax.plot(curve[:, 1], curve[:, 0], color="#475569", lw=1.2)
        ax.scatter(points[:, 1], points[:, 0], s=12, color="#a7a7a7", zorder=3)
        for index, point in enumerate(points):
            ax.annotate(str(index), (point[1], point[0]), xytext=(3, 4), textcoords="offset points", fontsize=8)
    for index, row in enumerate(first_groups):
        first = controls[int(row["first_contour"])][int(row["first_point"])]
        second = controls[int(row["second_contour"])][int(row["second_point"])]
        step = float(row["accepted_step_px"])
        color = plt.get_cmap("tab10")(index % 10)
        label = f"G{row['group']}: {row['first_point']}/{row['second_point']}, step={step:.4g} px"
        ax.plot([first[1], second[1]], [first[0], second[0]], "o--", color=color,
                alpha=1.0 if step > 0 else 0.35, lw=1.5, ms=5, label=label)
    points = np.vstack([curves[index] for index in contour_ids])
    lo, hi = points.min(axis=0) - 3, points.max(axis=0) + 3
    ax.set(xlim=(lo[1], hi[1]), ylim=(hi[0], lo[0]), aspect="equal", xlabel="x (pixel)", ylabel="y (pixel)",
           title=f"SRAF {rows[0]['sraf']}: joint pair groups at iteration {first_iteration}")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.savefig(directory / "initial_pair_groups.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    # 读取本次运行的阈值、实际矩阵和控制点，绘图全部使用同一目录中的实验状态。
    summary = dict(line.split(":", 1) for line in (directory / "summary.yaml").read_text().splitlines())
    threshold = float(summary["threshold"])
    safe_threshold = float(summary["safe_threshold"])
    wi_display_level = 0.5  # 仅用于 WI 绘图参考线；检测、选块及停止均使用 AI 阈值。
    guard_radius_px = int(summary["target_guard_px"])
    target = np.loadtxt(directory / "target_mask.txt")
    initial = {name: np.loadtxt(directory / "initial" / f"{name}.txt") for name in ["sraf_mask", "aerial", "wafer"]}
    final = {name: np.loadtxt(directory / "final" / f"{name}.txt") for name in initial}
    initial_metrics = load_metrics(directory / "initial/metrics.txt")
    final_metrics = load_metrics(directory / "final/metrics.txt")
    # 每次 3×3 膨胀扩展每轴 1 pixel，重复 r 次等价于 C++ 中每轴 [-r,r] 的正方形 Guard。
    guard = target > 0.5
    for _ in range(guard_radius_px):
        padded = np.pad(guard, 1)
        guard = np.logical_or.reduce([padded[y:y + target.shape[0], x:x + target.shape[1]]
                                     for y in range(3) for x in range(3)])
    peak_y, peak_x = np.unravel_index(np.where(~guard, initial["wafer"], -np.inf).argmax(), target.shape)
    # 放大窗口使用 [起点,终点) 整数切片，显示坐标仍为输入图的完整像素坐标。
    y0, y1 = max(0, peak_y - 20), min(target.shape[0], peak_y + 21)
    x0, x1 = max(0, peak_x - 20), min(target.shape[1], peak_x + 21)
    initial_sraf_controls = load_controls(directory / "initial/sraf_cps.txt")
    final_sraf_controls = load_controls(directory / "final/sraf_cps.txt")
    # 绘制 C++ 渲染器直接导出的实际拟合曲线，不重拟合或兼容历史实验目录。
    initial_sraf_curves = load_controls(directory / "initial/sraf_curves.txt")
    final_sraf_curves = load_controls(directory / "final/sraf_curves.txt")
    yy, xx = np.indices(target.shape)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fig.suptitle(f"Continuous WI comparison: before and after SRAF repair (AI threshold = {threshold:g})", fontsize=17)
    for ax, state, metrics, title in [(axes[0, 0], initial, initial_metrics, "Before"),
                                     (axes[0, 1], final, final_metrics, "After")]:
        # 全图和局部均直接读取成像导出的连续 WI，固定 [0,1] 色阶，红线为 WI=wi_display_level。
        im = ax.imshow(state["wafer"], cmap="gray", vmin=0, vmax=1, origin="upper")
        ax.contour(xx, yy, state["wafer"], levels=[wi_display_level], colors=["#ff514d"], linewidths=0.7)
        ax.contour(xx, yy, target, levels=[0.5], colors="white", linewidths=0.5)
        ax.plot(peak_x, peak_y, "o", ms=14, mfc="none", mec="#16c4e3", mew=1.5)
        ax.set_title(f"{title}: WI, AI above {threshold:g}: {int(metrics['background_printed_px'])} px")
        ax.set_xlabel("x (pixel)")
        ax.set_ylabel("y (pixel)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="WI")
    geometry = axes[0, 2]
    # 直接画控制点生成的周期 CR 曲线，并显式连接末采样点到首点；不用灰度 mask 的 0.5 等值线代替。
    for old, new, old_curve, new_curve in zip(initial_sraf_controls, final_sraf_controls, initial_sraf_curves, final_sraf_curves, strict=True):
        for curve, color in [(old_curve, "#d67016"), (new_curve, "#087fca")]:
            geometry.plot(curve[:, 1], curve[:, 0], color=color, lw=1.5)
        geometry.plot(old[:, 1], old[:, 0], ".", color="#d67016", ms=4)
        changed = np.linalg.norm(new - old, axis=1) > 1e-8
        geometry.plot(new[changed, 1], new[changed, 0], ".", color="#087fca", ms=5)
        for p, q in zip(old[changed], new[changed], strict=True):
            geometry.annotate("", xy=(q[1], q[0]), xytext=(p[1], p[0]),
                              arrowprops={"arrowstyle": "->", "color": "#087fca", "lw": 1})
    geometry.set(xlim=(x0, x1 - 1), ylim=(y1 - 1, y0), aspect="equal", title="Closed CR curves and moved controls")
    geometry.legend(handles=[Line2D([], [], color="#d67016", label="Before"),
                             Line2D([], [], color="#087fca", label="After")], fontsize=9)
    geometry.set_xlabel("x (pixel)")
    geometry.set_ylabel("y (pixel)")
    for ax, state, title in [(axes[1, 0], initial, "Before"), (axes[1, 1], final, "After")]:
        im = ax.imshow(state["wafer"], origin="upper", cmap="gray", vmin=0, vmax=1)
        local = state["wafer"][y0:y1, x0:x1]
        if local.min() < wi_display_level < local.max():
            ax.contour(xx[y0:y1, x0:x1], yy[y0:y1, x0:x1], local, levels=[wi_display_level], colors="red", linewidths=1.5)
        ax.contour(xx, yy, state["sraf_mask"], levels=[0.5], colors="#16c4e3", linewidths=0.7)
        ax.set(xlim=(x0, x1 - 1), ylim=(y1 - 1, y0), title=f"{title}: continuous WI, red contour = {wi_display_level:g}")
        ax.set_xlabel("x (pixel)")
        ax.set_ylabel("y (pixel)")
        fig.colorbar(im, ax=ax, shrink=0.8, label="WI")
    ax = axes[1, 2]
    ax.plot(np.arange(x0, x1), initial["wafer"][peak_y, x0:x1], color="#d67016", label="Before")
    ax.plot(np.arange(x0, x1), final["wafer"][peak_y, x0:x1], color="#087fca", label="After")
    ax.axhline(wi_display_level, color="#e33b36", ls="--", label=f"WI = {wi_display_level:g} (display reference)")
    ax.set(title=f"WI profile at y = {peak_y} pixels", xlabel="x (pixel)", ylabel="WI", ylim=(0, 1))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.savefig(directory / "comparison.png", dpi=180)
    plt.close(fig)

    # 使用相同 [0,1] 色阶展示真实连续 WI，红线为 WI=wi_display_level，不对结果做截断或重新显影。
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    fig.suptitle(f"Continuous wafer image: AI threshold = {threshold:g}, AI safety target = {safe_threshold:g}")
    for ax, state, title in [(axes[0], initial, "Before"), (axes[1], final, "After")]:
        local = state["wafer"][y0:y1, x0:x1]
        im = ax.imshow(local, cmap="gray", vmin=0, vmax=1, origin="upper", extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5))
        if local.min() < wi_display_level < local.max():
            ax.contour(xx[y0:y1, x0:x1], yy[y0:y1, x0:x1], local, levels=[wi_display_level], colors="red", linewidths=1)
        ax.set(title=f"{title}: WI near the initial hotspot", xlabel="x (pixel)", ylabel="y (pixel)")
        fig.colorbar(im, ax=ax, shrink=0.85, label="WI")
    ax = axes[2]
    ax.plot(np.arange(x0, x1), initial["wafer"][peak_y, x0:x1], color="#d67016", label="Before")
    ax.plot(np.arange(x0, x1), final["wafer"][peak_y, x0:x1], color="#087fca", label="After")
    ax.axhline(wi_display_level, color="red", ls="--", label=f"WI = {wi_display_level:g} (display reference)")
    ax.set(title=f"WI profile at y = {peak_y} pixels", xlabel="x (pixel)", ylabel="WI", ylim=(0, 1))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    fig.savefig(directory / "wafer_comparison.png", dpi=180)
    plt.close(fig)

    # 全局曲线展示背景峰值与源 EP 的 EPE，局部峰值另从组更新记录中读取。
    with (directory / "history.csv").open() as handle:
        history = list(csv.DictReader(handle))
    values = {key: np.asarray([float(row[key]) for row in history]) for key in history[0]}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(values["iteration"], values["background_peak"], "o-", label="Background peak")
    axes[0].axhline(threshold, color="gray", ls=":", label="Resist threshold")
    axes[0].axhline(safe_threshold, color="red", ls="--", label="AI safety target")
    axes[0].set(xlabel="Accepted iteration", ylabel="Intensity", title="Printing suppression")
    axes[1].plot(values["iteration"], values["mean_epe_nm"], "o-", label="Mean absolute EPE")
    axes[1].plot(values["iteration"], values["max_epe_nm"], "s-", label="Maximum absolute EPE")
    axes[1].set(xlabel="Accepted iteration", ylabel="EPE (nm)", title="Main-pattern EPE with fixed main mask")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(directory / "convergence.png", dpi=180)
    plt.close(fig)

    # 完整参数化掩模：保持主图形 B 样条；SRAF 直接使用移动后的原表示控制点重建周期 CR 曲线。
    main_controls = load_controls(directory / "main_cps.txt")
    main_curves = [close_curve(periodic_uniform_bspline(control, 200)) for control in main_controls]
    fig, ax = plt.subplots(figsize=(10, 10), constrained_layout=True)
    ax.contour(xx, yy, target, levels=[0.5], colors="black", linestyles="dashdot", linewidths=1)
    for curve in main_curves:
        ax.plot(curve[:, 1], curve[:, 0], color="#3152e8", lw=1.3)
    for old, new, curve in zip(initial_sraf_controls, final_sraf_controls, final_sraf_curves, strict=True):
        changed = np.linalg.norm(new - old, axis=1) > 1e-8
        ax.plot(curve[:, 1], curve[:, 0], color="#0b9cc3", lw=1)
        ax.scatter(new[~changed, 1], new[~changed, 0], s=5, color="#a7a7a7", zorder=3)
        ax.scatter(new[changed, 1], new[changed, 0], s=15, color="#e23b2f", zorder=4)
    # 曲线可能略超出输入像素域，画完整范围，避免边界裁切制造“未闭合”的视觉印象。
    complete_points = np.vstack(main_curves + final_sraf_curves)
    curve_lo = np.minimum(complete_points.min(axis=0), [0, 0]) - 2
    curve_hi = np.maximum(complete_points.max(axis=0), np.array(target.shape) - 1) + 2
    ax.set(xlim=(curve_lo[1], curve_hi[1]), ylim=(curve_hi[0], curve_lo[0]), aspect="equal",
           xlabel="x (pixel)", ylabel="y (pixel)", title="Final parametric curves from moved SRAF control points")
    ax.grid(alpha=0.2)
    ax.legend(handles=[Line2D([], [], color="#3152e8", label="Main B-spline (fixed)"),
                       Line2D([], [], color="#0b9cc3", label="SRAF periodic Catmull-Rom"),
                       Line2D([], [], color="black", ls="-.", label="Target contour"),
                       Line2D([], [], color="#a7a7a7", marker=".", ls="", label="Unchanged SRAF controls"),
                       Line2D([], [], color="#e23b2f", marker=".", ls="", label="Moved SRAF controls")], fontsize=9)
    fig.savefig(directory / "final_parametric_curves.png", dpi=220)
    plt.close(fig)

    # 完整显示离原始热点最近的一条闭合轮廓，避免裁切窗口把曲线画成看似断开的几段。
    contour_id = min(range(len(initial_sraf_controls)), key=lambda c: np.linalg.norm(
        initial_sraf_controls[c] - np.array([peak_y, peak_x]), axis=1).min())
    old, new = initial_sraf_controls[contour_id], final_sraf_controls[contour_id]
    fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)
    for curve, color, style, label in [(initial_sraf_curves[contour_id], "#d67016", "--", "Before: closed CR curve"),
                                       (final_sraf_curves[contour_id], "#087fca", "-", "After: closed CR curve")]:
        ax.plot(curve[:, 1], curve[:, 0], color=color, ls=style, lw=1.8, label=label)
    changed = np.linalg.norm(new - old, axis=1) > 1e-8
    ax.scatter(old[:, 1], old[:, 0], s=20, color="#d67016", label="Original controls", zorder=3)
    ax.scatter(new[:, 1], new[:, 0], s=20, color="#087fca", label="Final controls", zorder=4)
    for index, (p, q) in enumerate(zip(old, new, strict=True)):
        ax.annotate(str(index), (q[1], q[0]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        if changed[index]:
            ax.annotate("", xy=(q[1], q[0]), xytext=(p[1], p[0]),
                        arrowprops={"arrowstyle": "->", "color": "#087fca", "lw": 1})
    all_curve_points = np.vstack([initial_sraf_curves[contour_id], final_sraf_curves[contour_id]])
    lo, hi = all_curve_points.min(axis=0) - 3, all_curve_points.max(axis=0) + 3
    ax.set(xlim=(lo[1], hi[1]), ylim=(hi[0], lo[0]), aspect="equal", xlabel="x (pixel)", ylabel="y (pixel)",
           title=f"SRAF contour {contour_id}: complete fitted periodic curves and control indices")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=9)
    fig.savefig(directory / "sraf_hotspot_parametric_curve.png", dpi=220)
    plt.close(fig)
    # 点对记录只含实际移动组，组记录还保留步长为 0 的暂缓组，因此两者分别绘制。
    pair_file = directory / "pair_history.csv"
    group_file = directory / "group_history.csv"
    with pair_file.open() as handle:
        pairs = list(csv.DictReader(handle))
    if pairs:
        iterations = np.asarray([int(row["iteration"]) for row in pairs])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        axes[0].plot(iterations, [float(row["width_px"]) for row in pairs], "o-", label="Selected opposing controls")
        axes[0].plot(iterations, [float(row["section_diameter_px"]) for row in pairs], "s-", label="Local inscribed diameter")
        axes[0].set(xlabel="Accepted iteration", ylabel="Width (pixel)", title="Widest section selected each iteration")
        axes[1].plot(iterations, [float(row["region_peak_after"]) for row in pairs], "o-", label="Locked SRAF peak")
        axes[1].axhline(threshold, color="gray", ls=":", label="Resist threshold")
        axes[1].axhline(safe_threshold, color="red", ls="--", label="AI safety target")
        title = "One locked SRAF, joint pair-group updates"
        axes[1].set(xlabel="Accepted iteration", ylabel="Intensity", title=title)
        for ax in axes:
            ax.legend(fontsize=8)
            ax.grid(alpha=0.2)
        fig.savefig(directory / "pair_updates.png", dpi=180)
        plt.close(fig)
    with group_file.open() as handle:
        group_rows = list(csv.DictReader(handle))
    if group_rows:
        plot_initial_pair_groups(directory, initial_sraf_controls, initial_sraf_curves, group_rows)
        iteration_ids = sorted({int(row["iteration"]) for row in group_rows})
        # 同一接受轮次可能有多组记录，先按原文件顺序归组，再统计选中组、有效组和更新后峰值。
        groups_by_iteration: list[list[dict[str, str]]] = []
        for iteration in iteration_ids:
            iteration_groups = []
            for row in group_rows:
                if int(row["iteration"]) == iteration:
                    iteration_groups.append(row)
            groups_by_iteration.append(iteration_groups)
        selected_counts = [len(rows) for rows in groups_by_iteration]
        active_counts = [sum(float(row["accepted_step_px"]) > 0 for row in rows) for rows in groups_by_iteration]
        peaks = [float(rows[0]["region_peak_after"]) for rows in groups_by_iteration]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
        axes[0].bar(iteration_ids, selected_counts, color="#cbd5e1", label="Selected groups")
        axes[0].bar(iteration_ids, active_counts, color="#087fca", width=0.5, label="Active groups")
        axes[0].set(xlabel="Accepted iteration", ylabel="Pair groups", title="Independent step reduction / pausing")
        axes[1].plot(iteration_ids, peaks, "o-", label="Locked SRAF peak")
        axes[1].axhline(threshold, color="gray", ls=":", label="Resist threshold")
        axes[1].axhline(safe_threshold, color="red", ls="--", label="AI safety target")
        axes[1].set(xlabel="Accepted iteration", ylabel="Intensity", title="Joint update result")
        for ax in axes:
            ax.grid(alpha=0.2)
            ax.legend(fontsize=9)
        fig.savefig(directory / "group_updates.png", dpi=180)
        plt.close(fig)
    if summary["repair_mode"].strip() == "single_sraf_normal_gradient":
        # 各 SRAF 锁定时固定窗口；分阶段画损失，避免把不同窗口的均值误当成同一个目标。
        with (directory / "loss_history.csv").open() as handle:
            loss_rows = list(csv.DictReader(handle))
        if loss_rows:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
            for stage in sorted({int(row["stage"]) for row in loss_rows}):
                rows = [row for row in loss_rows if int(row["stage"]) == stage]
                iterations = [int(rows[0]["iteration"]) - 1] + [int(row["iteration"]) for row in rows]
                losses = [float(rows[0]["loss_before"])] + [float(row["loss_after"]) for row in rows]
                axes[0].plot(iterations, losses, "o-", label=f"SRAF {rows[0]['sraf']}")
            axes[0].set(xlabel="Iteration", ylabel="Mean squared AI excess", title="Finite-difference loss descent")
            axes[1].bar([int(row["iteration"]) for row in loss_rows], [int(row["moved_controls"]) for row in loss_rows])
            axes[1].set(xlabel="Iteration", ylabel="Moved controls", title="Joint inward normal updates")
            axes[0].legend()
            for ax in axes:
                ax.grid(alpha=0.2)
            fig.savefig(directory / "gradient_updates.png", dpi=180)
            plt.close(fig)
    print(directory / "comparison.png")
    print(directory / "wafer_comparison.png")
    print(directory / "convergence.png")
    print(directory / "final_parametric_curves.png")
    print(directory / "sraf_hotspot_parametric_curve.png")


if __name__ == "__main__":
    main()
