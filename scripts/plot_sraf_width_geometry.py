#!/usr/bin/env python3
"""Plot SRAF width-optimization results as pre-rasterization geometry.

The default inputs are the current outputs/sraf_optimizer_init files.  The
figure combines the target contour, main-mask periodic B-spline, SRAF skeleton
control points, fitted SRAF center curves, and bands with the optimized widths.
All coordinates and widths are expressed in pixels.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, PathPatch
from matplotlib.path import Path as MatplotlibPath
import numpy as np

from plot_meef_parametric_curves import (
    close_curve,
    load_grouped_yx,
    periodic_uniform_bspline,
    to_plot_pixels,
)


@dataclass(frozen=True)
class SrafCenterline:
    component_id: int
    closed: bool
    controls_xy: np.ndarray


@dataclass(frozen=True)
class SrafWidth:
    sraf_index: int
    component_id: int
    initial_half_width: float
    optimized_half_width: float


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_sraf_centerlines(path: Path) -> list[SrafCenterline]:
    """Load component metadata and (x,y) skeleton controls from the C++ output."""
    if not path.is_file():
        raise FileNotFoundError(f"SRAF control-point file not found: {path}")

    header_pattern = re.compile(
        r"^#\s*component=(\d+)\s+closed=([01])\s+count=(\d+)\s*$"
    )
    centerlines: list[SrafCenterline] = []
    component_id: int | None = None
    closed = False
    expected_count = 0
    points: list[tuple[float, float]] = []

    def finish_component() -> None:
        nonlocal component_id, closed, expected_count, points
        if component_id is None:
            return
        if len(points) != expected_count:
            raise ValueError(
                f"component {component_id} in {path} declares {expected_count} "
                f"points but contains {len(points)}"
            )
        controls = np.asarray(points, dtype=float).reshape((-1, 2))
        if controls.shape[0] == 0 or not np.all(np.isfinite(controls)):
            raise ValueError(f"component {component_id} in {path} has invalid controls")
        centerlines.append(SrafCenterline(component_id, closed, controls))
        component_id = None
        expected_count = 0
        points = []

    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = raw_line.strip()
        match = header_pattern.match(stripped)
        if match:
            finish_component()
            component_id = int(match.group(1))
            closed = match.group(2) == "1"
            expected_count = int(match.group(3))
            continue
        if not stripped or stripped.startswith("#"):
            continue
        if component_id is None:
            raise ValueError(f"coordinate appears before a component header at {path}:{line_number}")
        fields = stripped.split()
        if len(fields) != 2:
            raise ValueError(f"expected exactly 'x y' at {path}:{line_number}")
        try:
            points.append((float(fields[0]), float(fields[1])))
        except ValueError as error:
            raise ValueError(f"invalid coordinate at {path}:{line_number}") from error

    finish_component()
    if not centerlines:
        raise ValueError(f"no SRAF centerlines found in {path}")
    return centerlines


def load_independent_widths(path: Path) -> list[SrafWidth]:
    """Load per-SRAF optimized half widths and preserve the C++ index order."""
    if not path.is_file():
        raise FileNotFoundError(f"optimized-width file not found: {path}")

    widths: list[SrafWidth] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "sraf_index",
            "component_id",
            "initial_half_width",
            "optimized_half_width",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"unexpected columns in {path}")
        for row in reader:
            width = SrafWidth(
                sraf_index=int(row["sraf_index"]),
                component_id=int(row["component_id"]),
                initial_half_width=float(row["initial_half_width"]),
                optimized_half_width=float(row["optimized_half_width"]),
            )
            if width.optimized_half_width < 0.0 or not math.isfinite(
                width.optimized_half_width
            ):
                raise ValueError(f"invalid optimized half width at SRAF {width.sraf_index}")
            widths.append(width)

    widths.sort(key=lambda value: value.sraf_index)
    if any(width.sraf_index != index for index, width in enumerate(widths)):
        raise ValueError(f"SRAF indices in {path} are not contiguous from zero")
    return widths


def validate_component_mapping(
    centerlines: list[SrafCenterline], widths: list[SrafWidth]
) -> None:
    if len(centerlines) != len(widths):
        raise ValueError(
            f"centerline count {len(centerlines)} does not match width count {len(widths)}"
        )
    for index, (centerline, width) in enumerate(zip(centerlines, widths)):
        if width.sraf_index != index or width.component_id != centerline.component_id:
            raise ValueError(
                "SRAF/width mapping mismatch at index "
                f"{index}: component {centerline.component_id} versus {width.component_id}"
            )


def _clamped_uniform_knots(control_count: int, degree: int) -> np.ndarray:
    knots = np.empty(control_count + degree + 1, dtype=float)
    knots[: degree + 1] = 0.0
    knots[-(degree + 1) :] = 1.0
    for index in range(degree + 1, control_count):
        knots[index] = (index - degree) / (control_count - degree)
    return knots


def _de_boor_point(
    controls_xy: np.ndarray, knots: np.ndarray, degree: int, parameter: float
) -> np.ndarray:
    """Evaluate one clamped B-spline point with the de Boor algorithm."""
    control_count = len(controls_xy)
    if parameter >= 1.0:
        return controls_xy[-1].copy()
    span = int(np.searchsorted(knots, parameter, side="right") - 1)
    span = min(max(span, degree), control_count - 1)
    values = [controls_xy[span - degree + offset].copy() for offset in range(degree + 1)]

    for recursion in range(1, degree + 1):
        for offset in range(degree, recursion - 1, -1):
            control_index = span - degree + offset
            denominator = (
                knots[control_index + degree - recursion + 1] - knots[control_index]
            )
            alpha = 0.0 if abs(denominator) <= 1.0e-15 else (
                parameter - knots[control_index]
            ) / denominator
            values[offset] = (
                (1.0 - alpha) * values[offset - 1] + alpha * values[offset]
            )
    return values[degree]


def _periodic_bspline_point(
    controls_xy: np.ndarray, degree: int, segment: int, parameter: float
) -> np.ndarray:
    count = len(controls_xy)

    def point_at(index: int) -> np.ndarray:
        return controls_xy[index % count]

    if degree == 1:
        return (1.0 - parameter) * point_at(segment) + parameter * point_at(segment + 1)
    if degree == 2:
        b0 = 0.5 * (1.0 - parameter) ** 2
        b1 = 0.5 * (-2.0 * parameter**2 + 2.0 * parameter + 1.0)
        b2 = 0.5 * parameter**2
        return b0 * point_at(segment - 1) + b1 * point_at(segment) + b2 * point_at(segment + 1)

    t2 = parameter * parameter
    t3 = t2 * parameter
    b0 = (1.0 - 3.0 * parameter + 3.0 * t2 - t3) / 6.0
    b1 = (4.0 - 6.0 * t2 + 3.0 * t3) / 6.0
    b2 = (1.0 + 3.0 * parameter + 3.0 * t2 - 3.0 * t3) / 6.0
    b3 = t3 / 6.0
    return (
        b0 * point_at(segment - 1)
        + b1 * point_at(segment)
        + b2 * point_at(segment + 1)
        + b3 * point_at(segment + 2)
    )


def fit_sraf_centerline(
    centerline: SrafCenterline, sample_spacing: float = 0.25
) -> np.ndarray:
    """Match SrafCurve::fit() for open and closed skeleton centerlines."""
    controls = centerline.controls_xy
    control_count = len(controls)
    if sample_spacing <= 0.0:
        raise ValueError("sample_spacing must be positive")
    if control_count == 1:
        return controls.copy()

    degree = min(3, control_count - 1)
    if not centerline.closed:
        knots = _clamped_uniform_knots(control_count, degree)
        polygon_length = float(np.linalg.norm(np.diff(controls, axis=0), axis=1).sum())
        sample_count = max(2, int(math.ceil(polygon_length / sample_spacing)) + 1)
        parameters = np.linspace(0.0, 1.0, sample_count)
        return np.vstack(
            [_de_boor_point(controls, knots, degree, parameter) for parameter in parameters]
        )

    samples: list[np.ndarray] = []
    for segment in range(control_count):
        begin = controls[segment]
        end = controls[(segment + 1) % control_count]
        steps = max(1, int(math.ceil(np.linalg.norm(end - begin) / sample_spacing)))
        for step in range(steps):
            samples.append(
                _periodic_bspline_point(controls, degree, segment, step / steps)
            )
    return np.vstack(samples)


def _curve_normals(curve_xy: np.ndarray, closed: bool) -> np.ndarray:
    if len(curve_xy) < 2:
        return np.zeros_like(curve_xy)
    if closed:
        tangents = np.roll(curve_xy, -1, axis=0) - np.roll(curve_xy, 1, axis=0)
    else:
        tangents = np.empty_like(curve_xy)
        tangents[0] = curve_xy[1] - curve_xy[0]
        tangents[-1] = curve_xy[-1] - curve_xy[-2]
        if len(curve_xy) > 2:
            tangents[1:-1] = curve_xy[2:] - curve_xy[:-2]

    lengths = np.linalg.norm(tangents, axis=1)
    for index in np.flatnonzero(lengths <= 1.0e-12):
        if index > 0:
            tangents[index] = tangents[index - 1]
        elif len(tangents) > 1:
            tangents[index] = tangents[index + 1]
    lengths = np.maximum(np.linalg.norm(tangents, axis=1), 1.0e-12)
    return np.column_stack((-tangents[:, 1], tangents[:, 0])) / lengths[:, None]


def _signed_area(points_xy: np.ndarray) -> float:
    following = np.roll(points_xy, -1, axis=0)
    return 0.5 * float(
        np.sum(points_xy[:, 0] * following[:, 1] - following[:, 0] * points_xy[:, 1])
    )


def _ring_patch(
    first_xy: np.ndarray, second_xy: np.ndarray, facecolor: str, alpha: float
) -> PathPatch:
    outer, inner = (
        (first_xy, second_xy)
        if abs(_signed_area(first_xy)) >= abs(_signed_area(second_xy))
        else (second_xy, first_xy)
    )
    if _signed_area(outer) < 0.0:
        outer = outer[::-1]
    if _signed_area(inner) > 0.0:
        inner = inner[::-1]

    vertices = np.vstack((outer, outer[0], inner, inner[0]))
    codes = np.concatenate(
        (
            [MatplotlibPath.MOVETO],
            np.full(len(outer) - 1, MatplotlibPath.LINETO),
            [MatplotlibPath.CLOSEPOLY, MatplotlibPath.MOVETO],
            np.full(len(inner) - 1, MatplotlibPath.LINETO),
            [MatplotlibPath.CLOSEPOLY],
        )
    )
    return PathPatch(
        MatplotlibPath(vertices, codes),
        facecolor=facecolor,
        edgecolor="none",
        alpha=alpha,
    )


def draw_sraf_band(
    axis: plt.Axes,
    curve_xy: np.ndarray,
    half_width: float,
    closed: bool,
    facecolor: str,
    edgecolor: str,
) -> None:
    """Draw a geometric width band around a densely sampled center curve."""
    if len(curve_xy) == 1:
        axis.add_patch(
            Circle(curve_xy[0], half_width, facecolor=facecolor, edgecolor=edgecolor,
                   linewidth=0.55, alpha=0.34, zorder=1)
        )
        return

    normals = _curve_normals(curve_xy, closed)
    first = curve_xy + half_width * normals
    second = curve_xy - half_width * normals

    if closed:
        patch = _ring_patch(first, second, facecolor, 0.34)
        patch.set_zorder(1)
        axis.add_patch(patch)
        for boundary in (first, second):
            closed_boundary = np.vstack((boundary, boundary[0]))
            axis.plot(closed_boundary[:, 0], closed_boundary[:, 1], color=edgecolor,
                      linewidth=0.55, alpha=0.9, zorder=2)
        return

    polygon = np.vstack((first, second[::-1]))
    axis.fill(polygon[:, 0], polygon[:, 1], color=facecolor, alpha=0.34, zorder=1)
    axis.plot(first[:, 0], first[:, 1], color=edgecolor, linewidth=0.55, alpha=0.9, zorder=2)
    axis.plot(second[:, 0], second[:, 1], color=edgecolor, linewidth=0.55, alpha=0.9, zorder=2)
    for endpoint in (curve_xy[0], curve_xy[-1]):
        axis.add_patch(
            Circle(endpoint, half_width, facecolor=facecolor, edgecolor=edgecolor,
                   linewidth=0.55, alpha=0.34, zorder=1)
        )


def plot_geometry(
    result_dir: Path,
    target_mask_path: Path,
    output_path: Path,
    main_samples: int,
    sraf_sample_spacing: float,
) -> tuple[int, int, float, float]:
    target_mask = np.loadtxt(target_mask_path)
    if target_mask.ndim != 2 or min(target_mask.shape) < 2:
        raise ValueError("target mask must be a two-dimensional matrix")

    main_controls_yx = load_grouped_yx(result_dir / "main_control_points_loaded_yx.txt")
    main_curves_yx = [
        periodic_uniform_bspline(controls, main_samples) for controls in main_controls_yx
    ]
    centerlines = load_sraf_centerlines(result_dir / "sraf_control_points_xy.txt")
    widths = load_independent_widths(result_dir / "optimized_independent_widths.csv")
    validate_component_mapping(centerlines, widths)
    sraf_curves_xy = [
        fit_sraf_centerline(centerline, sraf_sample_spacing) for centerline in centerlines
    ]

    rows, columns = target_mask.shape
    x_pixels = np.arange(columns)
    y_pixels = np.arange(rows)
    target_level = 0.5 * (float(np.nanmin(target_mask)) + float(np.nanmax(target_mask)))

    band_fill = "#5bc0be"
    band_edge = "#168c8c"
    centerline_color = "#7b2cbf"
    skeleton_point_color = "#f28e2b"
    main_curve_color = "#2457ff"
    main_point_color = "#e2231a"

    figure, axis = plt.subplots(figsize=(9.5, 8.4), constrained_layout=True)
    axis.contour(
        x_pixels,
        y_pixels,
        target_mask,
        levels=[target_level],
        colors="black",
        linewidths=1.2,
        linestyles="dashdot",
        zorder=3,
    )

    for centerline, curve, width in zip(centerlines, sraf_curves_xy, widths):
        draw_sraf_band(
            axis,
            curve,
            width.optimized_half_width,
            centerline.closed,
            band_fill,
            band_edge,
        )
        axis.plot(
            curve[:, 0],
            curve[:, 1],
            color=centerline_color,
            linewidth=0.9,
            linestyle="--",
            zorder=4,
        )
        axis.scatter(
            centerline.controls_xy[:, 0],
            centerline.controls_xy[:, 1],
            s=11,
            marker="s",
            color=skeleton_point_color,
            edgecolors="white",
            linewidths=0.2,
            zorder=6,
        )

    for controls_yx, curve_yx in zip(main_controls_yx, main_curves_yx):
        curve_x, curve_y = to_plot_pixels(close_curve(curve_yx))
        control_x, control_y = to_plot_pixels(controls_yx)
        axis.plot(curve_x, curve_y, color=main_curve_color, linewidth=1.7, zorder=5)
        axis.scatter(
            control_x,
            control_y,
            s=18,
            marker="o",
            color=main_point_color,
            edgecolors="white",
            linewidths=0.25,
            zorder=7,
        )

    axis.set_title("Optimized independent SRAF widths (parametric geometry)")
    axis.set_xlabel("x (pixel)")
    axis.set_ylabel("y (pixel)")
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(0, columns - 1)
    axis.set_ylim(rows - 1, 0)
    axis.grid(True, linestyle=":", linewidth=0.55, alpha=0.58)

    legend_handles = [
        Line2D([0], [0], color=main_curve_color, linewidth=1.7, label="Main B-spline"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=main_point_color,
               markeredgecolor="white", markersize=5.0, label="Main control points"),
        Patch(facecolor=band_fill, edgecolor=band_edge, alpha=0.34,
              label="Optimized SRAF region (full width = 2w)"),
        Line2D([0], [0], color=centerline_color, linewidth=0.9, linestyle="--",
               label="SRAF fitted center curve"),
        Line2D([0], [0], marker="s", linestyle="none",
               markerfacecolor=skeleton_point_color, markeredgecolor="white",
               markersize=4.5, label="SRAF skeleton control points"),
        Line2D([0], [0], color="black", linewidth=1.2, linestyle="dashdot",
               label="Target contour"),
    ]
    axis.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        framealpha=0.95,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)

    optimized = np.asarray([width.optimized_half_width for width in widths])
    return (
        len(main_controls_yx),
        len(centerlines),
        float(np.min(optimized)),
        float(np.max(optimized)),
    )


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=root / "result" / "SRAF_result",
        help="SRAF width-optimization output directory",
    )
    parser.add_argument(
        "--target-mask",
        type=Path,
        default=root / "result" / "MEEF_result" / "test" / "target_mask.txt",
        help="target mask used only to extract its contour",
    )
    parser.add_argument(
        "--main-samples",
        type=int,
        default=200,
        help="periodic main B-spline samples per contour",
    )
    parser.add_argument(
        "--sraf-sample-spacing",
        type=float,
        default=0.25,
        help="SRAF fitted-curve sample spacing in pixels (matches C++ default)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output PNG; defaults inside --result-dir",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    target_mask = args.target_mask.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else result_dir / "optimized_independent_parametric_geometry.png"
    )
    main_count, sraf_count, minimum_width, maximum_width = plot_geometry(
        result_dir,
        target_mask,
        output_path,
        args.main_samples,
        args.sraf_sample_spacing,
    )
    print(f"result directory   : {result_dir}")
    print(f"target mask        : {target_mask}")
    print(f"main contours      : {main_count}")
    print(f"SRAF center curves : {sraf_count}")
    print(f"half-width range   : [{minimum_width:.6f}, {maximum_width:.6f}] pixel")
    print(f"full-width range   : [{2.0 * minimum_width:.6f}, {2.0 * maximum_width:.6f}] pixel")
    print(f"saved figure       : {output_path}")


if __name__ == "__main__":
    main()
