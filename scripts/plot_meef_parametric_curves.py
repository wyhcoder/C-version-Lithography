#!/usr/bin/env python3
"""Plot pre-rasterization MEEF curves, control points, and target contour.

The default input/output directory is result/MEEF_result/test.  Main control
points come from the selected best snapshot, while the fixed SRAF control
points come from sraf_cps.txt.  Coordinates in all project files are (y, x).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


@dataclass(frozen=True)
class PlotConfig:
    curve_type: str
    pattern_name: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _unquoted_value(text: str) -> str:
    value = text.split("#", 1)[0].strip()
    quoted = re.findall(r'["\']([^"\']*)["\']', value)
    return quoted[-1] if quoted else value.split()[-1]


def load_plot_config(path: Path) -> PlotConfig:
    """Read the few needed YAML scalars without requiring PyYAML."""
    if not path.is_file():
        raise FileNotFoundError(f"configuration snapshot not found: {path}")

    curve_type: str | None = None
    pattern_name = ""
    section = ""

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if not raw_line[0].isspace() and stripped.endswith(":"):
            section = stripped[:-1]
            continue

        if not raw_line[0].isspace() and stripped.startswith("PATTERN_NAME:"):
            pattern_name = _unquoted_value(stripped.split(":", 1)[1])
            continue

        if ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        if section == "meef" and key == "curve_type":
            curve_type = _unquoted_value(value)
        elif section == "meef" and key == "pattern_name" and not pattern_name:
            pattern_name = _unquoted_value(value)

    if curve_type is None:
        raise ValueError(f"meef.curve_type is missing from {path}")
    if curve_type != "BS":
        raise ValueError(
            f"this script currently reproduces the project's BS curve only; got {curve_type!r}"
        )
    return PlotConfig(curve_type, pattern_name)


def load_grouped_yx(path: Path) -> list[np.ndarray]:
    """Load groups separated by comment headers or blank lines."""
    if not path.is_file():
        raise FileNotFoundError(f"control-point file not found: {path}")

    groups: list[np.ndarray] = []
    current: list[tuple[float, float]] = []

    def finish_group() -> None:
        if current:
            points = np.asarray(current, dtype=float)
            if points.shape[0] < 2:
                raise ValueError(f"a contour in {path} contains fewer than two points")
            if not np.all(np.isfinite(points)):
                raise ValueError(f"a contour in {path} contains a non-finite coordinate")
            groups.append(points)
            current.clear()

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            finish_group()
            continue
        fields = stripped.split()
        if len(fields) != 2:
            raise ValueError(f"expected exactly 'y x' at {path}:{line_number}")
        try:
            current.append((float(fields[0]), float(fields[1])))
        except ValueError as error:
            raise ValueError(f"invalid coordinate at {path}:{line_number}") from error

    finish_group()
    if not groups:
        raise ValueError(f"no control-point groups found in {path}")
    return groups


def periodic_uniform_bspline(control_yx: np.ndarray, sample_count: int) -> np.ndarray:
    """Match ParametricDemo::b_spline() for a closed periodic contour."""
    control_count = control_yx.shape[0]
    if control_count < 2:
        return control_yx.copy()
    if sample_count < 3:
        raise ValueError("sample_count must be at least 3")

    degree = min(3, control_count - 1)
    fitted = np.empty((sample_count, 2), dtype=float)

    def point_at(index: int) -> np.ndarray:
        return control_yx[index % control_count]

    for sample in range(sample_count):
        global_t = sample / sample_count * control_count
        segment = int(np.floor(global_t))
        t = global_t - segment

        if degree == 1:
            fitted[sample] = (1.0 - t) * point_at(segment) + t * point_at(segment + 1)
        elif degree == 2:
            b0 = 0.5 * (1.0 - t) ** 2
            b1 = 0.5 * (-2.0 * t * t + 2.0 * t + 1.0)
            b2 = 0.5 * t * t
            fitted[sample] = (
                b0 * point_at(segment - 1)
                + b1 * point_at(segment)
                + b2 * point_at(segment + 1)
            )
        else:
            t2 = t * t
            t3 = t2 * t
            b0 = (1.0 - 3.0 * t + 3.0 * t2 - t3) / 6.0
            b1 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0
            b2 = (1.0 + 3.0 * t + 3.0 * t2 - 3.0 * t3) / 6.0
            b3 = t3 / 6.0
            fitted[sample] = (
                b0 * point_at(segment - 1)
                + b1 * point_at(segment)
                + b2 * point_at(segment + 1)
                + b3 * point_at(segment + 2)
            )
    return fitted


def to_plot_pixels(points_yx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert project rows (y, x) into plotting arrays (x, y), in pixels."""
    return points_yx[:, 1], points_yx[:, 0]


def close_curve(points_yx: np.ndarray) -> np.ndarray:
    return np.vstack((points_yx, points_yx[0]))


def plot_result(
    result_dir: Path,
    best_name: str,
    sample_count: int,
    output_path: Path,
) -> tuple[int, int, int]:
    config = load_plot_config(result_dir / "config_used.yaml")
    target_mask = np.loadtxt(result_dir / "target_mask.txt")
    if target_mask.ndim != 2 or min(target_mask.shape) < 2:
        raise ValueError("target_mask.txt must be a two-dimensional matrix")

    main_path = result_dir / best_name / "control_points.txt"
    sraf_path = result_dir / "sraf_cps.txt"
    main_controls = load_grouped_yx(main_path)
    sraf_controls = load_grouped_yx(sraf_path)
    main_curves = [periodic_uniform_bspline(points, sample_count) for points in main_controls]
    sraf_curves = [periodic_uniform_bspline(points, sample_count) for points in sraf_controls]

    rows, columns = target_mask.shape
    x_axis_pixel = np.arange(columns)
    y_axis_pixel = np.arange(rows)

    figure, axis = plt.subplots(figsize=(8.2, 8.2), constrained_layout=True)

    target_level = 0.5 * (float(np.nanmin(target_mask)) + float(np.nanmax(target_mask)))
    axis.contour(
        x_axis_pixel,
        y_axis_pixel,
        target_mask,
        levels=[target_level],
        colors="black",
        linewidths=1.25,
        linestyles="dashdot",
        zorder=1,
    )

    for curve in main_curves:
        x_pixel, y_pixel = to_plot_pixels(close_curve(curve))
        axis.plot(x_pixel, y_pixel, color="#2457ff", linewidth=1.7, zorder=3)

    for curve in sraf_curves:
        x_pixel, y_pixel = to_plot_pixels(close_curve(curve))
        axis.plot(x_pixel, y_pixel, color="#19a0d8", linewidth=1.05, linestyle="--", zorder=2)

    main_all = np.vstack(main_controls)
    sraf_all = np.vstack(sraf_controls)
    main_x, main_y = to_plot_pixels(main_all)
    sraf_x, sraf_y = to_plot_pixels(sraf_all)
    axis.scatter(main_x, main_y, s=17, color="#e2231a", edgecolors="white",
                 linewidths=0.25, zorder=5)
    axis.scatter(sraf_x, sraf_y, s=7, color="#e2231a", linewidths=0.0, zorder=4)

    title_prefix = "wEPE optimal" if best_name == "best_wepe" else "EPE optimal"
    axis.set_title(f"{title_prefix} B-spline mask (MEEF)")
    axis.set_xlabel("x (pixel)")
    axis.set_ylabel("y (pixel)")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(x_axis_pixel[0], x_axis_pixel[-1])
    axis.set_ylim(y_axis_pixel[-1], y_axis_pixel[0])
    axis.grid(True, linestyle=":", linewidth=0.6, alpha=0.65)

    legend_handles = [
        Line2D([0], [0], color="#2457ff", linewidth=1.7, label="Main B-spline"),
        Line2D([0], [0], color="#19a0d8", linewidth=1.05, linestyle="--",
               label="SRAF B-spline"),
        Line2D([0], [0], color="black", linewidth=1.25, linestyle="dashdot",
               label="Target contour"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#e2231a",
               markeredgecolor="white", markersize=5.0, label="Control points"),
    ]
    axis.legend(handles=legend_handles, loc="upper right", framealpha=0.92)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)
    return len(main_controls), len(sraf_controls), len(main_all) + len(sraf_all)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=root / "result" / "MEEF_result" / "test",
        help="completed MEEF result directory",
    )
    parser.add_argument(
        "--best",
        choices=("best_wepe", "best_epe"),
        default="best_wepe",
        help="which optimized main-control-point snapshot to plot",
    )
    parser.add_argument(
        "--samples-per-contour",
        type=int,
        default=200,
        help="B-spline samples per closed contour (C++ history uses 200)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output PNG path; defaults inside --result-dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else result_dir / f"{args.best}_parametric_curves.png"
    )
    main_count, sraf_count, control_count = plot_result(
        result_dir,
        args.best,
        args.samples_per_contour,
        output_path,
    )
    print(f"result directory : {result_dir}")
    print(f"main contours    : {main_count}")
    print(f"SRAF contours    : {sraf_count}")
    print(f"control points   : {control_count}")
    print(f"saved figure     : {output_path}")


if __name__ == "__main__":
    main()
