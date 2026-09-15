#!/usr/bin/env python3
"""Visualize three-point discrete curvature on a closed 2-D contour.

The project stores contour rows as (y, x).  Internally this script converts
them to plotting coordinates (x, y), computes the circumcircle curvature for
every point, and demonstrates curvature-weighted point sampling.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np


EPSILON = 1.0e-12


def _polyline_segment(start: tuple[float, float], end: tuple[float, float], count: int) -> np.ndarray:
    """Return a dense segment in (x, y), excluding its last endpoint."""
    t = np.linspace(0.0, 1.0, count, endpoint=False)
    return np.column_stack(
        (start[0] + (end[0] - start[0]) * t,
         start[1] + (end[1] - start[1]) * t)
    )


def _circle_arc(
    center: tuple[float, float], start_angle: float, end_angle: float, count: int
) -> np.ndarray:
    """Return a dense circular arc in (x, y), excluding its last endpoint."""
    angle = np.linspace(start_angle, end_angle, count, endpoint=False)
    return np.column_stack(
        (center[0] + np.cos(angle), center[1] + np.sin(angle))
    )


def resample_closed_contour(points_xy: np.ndarray, sample_count: int) -> np.ndarray:
    """Resample a closed contour approximately uniformly in arc length."""
    closed = np.vstack((points_xy, points_xy[0]))
    segment_lengths = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    targets = np.linspace(0.0, cumulative[-1], sample_count, endpoint=False)
    x = np.interp(targets, cumulative, closed[:, 0])
    y = np.interp(targets, cumulative, closed[:, 1])
    return np.column_stack((x, y))


def make_demo_contour(sample_count: int = 160) -> np.ndarray:
    """Create a lithography-like rounded rectangle in project order (y, x)."""
    half_width, half_height, radius = 6.0, 4.0, 1.0
    dense_count = 160

    # Counter-clockwise path. Arcs below are unit circles, hence radius = 1.
    pieces = [
        _polyline_segment((half_width - radius, half_height),
                          (-half_width + radius, half_height), dense_count),
        _circle_arc((-half_width + radius, half_height - radius),
                    math.pi / 2.0, math.pi, dense_count),
        _polyline_segment((-half_width, half_height - radius),
                          (-half_width, -half_height + radius), dense_count),
        _circle_arc((-half_width + radius, -half_height + radius),
                    math.pi, 3.0 * math.pi / 2.0, dense_count),
        _polyline_segment((-half_width + radius, -half_height),
                          (half_width - radius, -half_height), dense_count),
        _circle_arc((half_width - radius, -half_height + radius),
                    3.0 * math.pi / 2.0, 2.0 * math.pi, dense_count),
        _polyline_segment((half_width, -half_height + radius),
                          (half_width, half_height - radius), dense_count),
        _circle_arc((half_width - radius, half_height - radius),
                    0.0, math.pi / 2.0, dense_count),
    ]
    xy = resample_closed_contour(np.vstack(pieces), sample_count)
    return xy[:, [1, 0]]  # project convention: (y, x)


def load_contour(path: Path, coordinate_order: str) -> np.ndarray:
    """Load a whitespace/comma separated N x 2 contour and return (y, x)."""
    try:
        points = np.loadtxt(path, delimiter=None)
    except ValueError:
        points = np.loadtxt(path, delimiter=",")
    points = np.atleast_2d(points).astype(float)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("contour input must contain at least three rows and two columns")
    if coordinate_order == "xy":
        points = points[:, [1, 0]]
    return points


def discrete_curvature(
    contour_yx: np.ndarray, neighbor_step: int = 1, closed: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Compute signed and absolute circumcircle curvature at every point.

    For P_prev, P_i, P_next:
        kappa_signed = 2 * cross(P_i-P_prev, P_next-P_i) / (a*b*c)
    where a, b and c are the three triangle side lengths.
    """
    point_count = contour_yx.shape[0]
    if neighbor_step < 1:
        raise ValueError("neighbor_step must be at least 1")
    if closed and 2 * neighbor_step >= point_count:
        raise ValueError("closed contour requires 2 * neighbor_step < point count")

    xy = contour_yx[:, [1, 0]]
    signed = np.full(point_count, np.nan, dtype=float)
    indices = range(point_count) if closed else range(neighbor_step, point_count - neighbor_step)

    for index in indices:
        previous = xy[(index - neighbor_step) % point_count]
        current = xy[index]
        following = xy[(index + neighbor_step) % point_count]
        incoming = current - previous
        outgoing = following - current
        a = np.linalg.norm(incoming)
        b = np.linalg.norm(outgoing)
        c = np.linalg.norm(following - previous)
        denominator = a * b * c
        if denominator <= EPSILON:
            signed[index] = 0.0
            continue
        cross = incoming[0] * outgoing[1] - incoming[1] * outgoing[0]
        signed[index] = 2.0 * cross / denominator

    return signed, np.abs(signed)


def curvature_weighted_indices(
    curvature: np.ndarray, selected_count: int, strength: float = 6.0
) -> np.ndarray:
    """Distribute samples by cumulative weight, favoring high curvature."""
    finite = np.nan_to_num(curvature, nan=0.0, posinf=0.0, neginf=0.0)
    if selected_count < 1:
        return np.empty(0, dtype=int)
    selected_count = min(selected_count, len(finite))
    maximum = float(np.max(finite))
    normalized = finite / maximum if maximum > EPSILON else np.zeros_like(finite)
    weights = 1.0 + strength * normalized
    cumulative = np.cumsum(weights)
    targets = (np.arange(selected_count) + 0.5) * cumulative[-1] / selected_count
    indices = np.searchsorted(cumulative, targets)
    return np.unique(np.clip(indices, 0, len(finite) - 1))


def circumcenter(points_xy: np.ndarray) -> tuple[np.ndarray, float]:
    """Return circumcenter and radius for exactly three non-collinear points."""
    p0, p1, p2 = points_xy
    matrix = 2.0 * np.vstack((p1 - p0, p2 - p0))
    rhs = np.array(
        [np.dot(p1, p1) - np.dot(p0, p0),
         np.dot(p2, p2) - np.dot(p0, p0)]
    )
    center = np.linalg.solve(matrix, rhs)
    return center, float(np.linalg.norm(center - p1))


def choose_example_index(curvature: np.ndarray, neighbor_step: int) -> int:
    """Choose a high-curvature point away from a rounded-corner transition."""
    finite = np.nan_to_num(curvature, nan=-1.0)
    candidate_order = np.argsort(finite)[::-1]
    for index in candidate_order:
        neighborhood = finite.take(
            [(index - neighbor_step) % len(finite), index, (index + neighbor_step) % len(finite)]
        )
        if np.all(neighborhood > 0.75 * finite[index]):
            return int(index)
    return int(candidate_order[0])


def plot_visualization(
    contour_yx: np.ndarray,
    signed_curvature: np.ndarray,
    curvature: np.ndarray,
    neighbor_step: int,
    selected: np.ndarray,
    output_path: Path,
    closed: bool,
) -> int:
    """Create the four-panel explanation figure and return the example index."""
    xy = contour_yx[:, [1, 0]]
    example = choose_example_index(curvature, neighbor_step)
    previous_index = (example - neighbor_step) % len(xy)
    following_index = (example + neighbor_step) % len(xy)
    triple = xy[[previous_index, example, following_index]]
    center, radius = circumcenter(triple)

    incoming = triple[1] - triple[0]
    outgoing = triple[2] - triple[1]
    a = np.linalg.norm(incoming)
    b = np.linalg.norm(outgoing)
    c = np.linalg.norm(triple[2] - triple[0])
    area = 0.5 * abs(incoming[0] * outgoing[1] - incoming[1] * outgoing[0])

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    fig.suptitle("Three-point curvature on a discrete contour", fontsize=17)

    axis = axes[0, 0]
    display_xy = np.vstack((xy, xy[0])) if closed else xy
    segments = np.stack((display_xy[:-1], display_xy[1:]), axis=1)
    segment_values = curvature if closed else curvature[:-1]
    norm = Normalize(vmin=0.0, vmax=max(float(np.nanmax(curvature)), EPSILON))
    colored = LineCollection(segments, cmap="viridis", norm=norm, linewidth=3.0)
    colored.set_array(segment_values)
    axis.add_collection(colored)
    axis.scatter(xy[:, 0], xy[:, 1], s=11, color="0.65", zorder=2, label="discrete points")
    axis.scatter(xy[example, 0], xy[example, 1], s=90, marker="*", color="crimson",
                 zorder=4, label=f"example P[{example}]")
    axis.autoscale()
    axis.set_aspect("equal")
    axis.set_title("1. Curvature at every contour point")
    axis.set_xlabel("x (pixel)")
    axis.set_ylabel("y (pixel)")
    axis.legend(loc="center", ncols=2)
    fig.colorbar(colored, ax=axis, label="|curvature| (1/pixel)", shrink=0.82)

    axis = axes[0, 1]
    angle = np.linspace(0.0, 2.0 * math.pi, 400)
    circle = center + radius * np.column_stack((np.cos(angle), np.sin(angle)))
    axis.plot(circle[:, 0], circle[:, 1], "--", color="tab:blue", label="circumcircle")
    closed_triple = np.vstack((triple, triple[0]))
    axis.plot(closed_triple[:, 0], closed_triple[:, 1], "-", color="0.35", label="triangle")
    axis.scatter(triple[:, 0], triple[:, 1], s=[55, 95, 55],
                 color=["tab:orange", "crimson", "tab:green"], zorder=4)
    axis.scatter(center[0], center[1], marker="x", s=80, color="tab:blue", zorder=4)
    axis.plot([center[0], triple[1, 0]], [center[1], triple[1, 1]],
              ":", color="tab:blue")
    offsets = [(-8, 8), (6, -18), (6, -15)]
    labels = [r"$P_{i-s}$", r"$P_i$", r"$P_{i+s}$"]
    for point, label, offset in zip(triple, labels, offsets):
        axis.annotate(label, point, xytext=offset, textcoords="offset points", fontsize=12)
    axis.annotate("O", center, xytext=(6, 5), textcoords="offset points", fontsize=12)
    values = (
        f"a={a:.3f}, b={b:.3f}, c={c:.3f}\n"
        f"area A={area:.4f}\n"
        f"R=abc/(4A)={radius:.3f}\n"
        f"|kappa|=1/R={curvature[example]:.3f}"
    )
    axis.text(0.02, 0.02, values, transform=axis.transAxes, va="bottom",
              bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88})
    axis.set_aspect("equal")
    axis.set_title(f"2. Local triangle at P[{example}] (neighbor step s={neighbor_step})")
    axis.set_xlabel("x (pixel)")
    axis.set_ylabel("y (pixel)")
    axis.legend(loc="upper right")

    axis = axes[1, 0]
    point_indices = np.arange(len(curvature))
    axis.plot(point_indices, curvature, color="tab:blue", linewidth=1.5, label="|curvature|")
    axis.scatter(point_indices, curvature, s=13, color="tab:blue")
    axis.scatter([example], [curvature[example]], s=90, marker="*", color="crimson",
                 zorder=4)
    axis.axhline(1.0 / radius, color="crimson", linestyle=":", linewidth=1.2)
    axis.annotate(f"P[{example}]", (example, curvature[example]), xytext=(7, -18),
                  textcoords="offset points", color="crimson")
    axis.set_title("3. Curvature versus contour index")
    axis.set_xlabel("contour point index i")
    axis.set_ylabel("|curvature| (1/pixel)")

    axis = axes[1, 1]
    axis.plot(display_xy[:, 0], display_xy[:, 1], color="0.78", linewidth=1.5)
    axis.scatter(xy[:, 0], xy[:, 1], s=10, color="0.72", label="all contour points")
    selected_sizes = 45.0 + 95.0 * curvature[selected] / max(float(np.nanmax(curvature)), EPSILON)
    axis.scatter(xy[selected, 0], xy[selected, 1], s=selected_sizes,
                 facecolors="none", edgecolors="tab:red", linewidths=1.6,
                 label=f"curvature-weighted samples ({len(selected)})")
    for rank, index in enumerate(selected):
        if rank % max(1, len(selected) // 6) == 0:
            axis.annotate(str(index), xy[index], xytext=(4, 4), textcoords="offset points", fontsize=8)
    axis.set_aspect("equal")
    axis.set_title("4. One way to sample more densely at bends")
    axis.set_xlabel("x (pixel)")
    axis.set_ylabel("y (pixel)")
    axis.legend(loc="center")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return example


def save_values(
    output_path: Path,
    contour_yx: np.ndarray,
    signed_curvature: np.ndarray,
    curvature: np.ndarray,
    selected: np.ndarray,
) -> None:
    selected_set = set(int(index) for index in selected)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["index", "y", "x", "signed_curvature", "curvature", "radius", "selected"])
        for index, ((y, x), signed, absolute) in enumerate(
            zip(contour_yx, signed_curvature, curvature)
        ):
            radius = math.inf if not np.isfinite(absolute) or absolute <= EPSILON else 1.0 / absolute
            writer.writerow([index, y, x, signed, absolute, radius, int(index in selected_set)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="optional N x 2 contour TXT/CSV")
    parser.add_argument("--coordinate-order", choices=("yx", "xy"), default="yx",
                        help="column order of --input (default: project convention yx)")
    parser.add_argument("--open", action="store_true", help="treat input as an open contour")
    parser.add_argument("--neighbor-step", type=int, default=3,
                        help="index distance from P_i to its two neighbors")
    parser.add_argument("--selected-count", type=int, default=28,
                        help="number of curvature-weighted example samples")
    parser.add_argument("--demo-points", type=int, default=160,
                        help="number of points in the built-in rounded rectangle")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    closed = not args.open
    contour_yx = (
        load_contour(args.input, args.coordinate_order)
        if args.input
        else make_demo_contour(args.demo_points)
    )
    signed, curvature = discrete_curvature(contour_yx, args.neighbor_step, closed)
    selected = curvature_weighted_indices(curvature, args.selected_count)
    figure_path = args.output_dir / "discrete_curvature_visualization.png"
    values_path = args.output_dir / "curvature_values.csv"
    example = plot_visualization(
        contour_yx, signed, curvature, args.neighbor_step, selected, figure_path, closed
    )
    save_values(values_path, contour_yx, signed, curvature, selected)
    print(f"contour points : {len(contour_yx)}")
    print(f"example index  : {example}")
    print(f"example kappa  : {curvature[example]:.8f}")
    print(f"example radius : {1.0 / curvature[example]:.8f}")
    print(f"selected points: {len(selected)}")
    print(f"figure         : {figure_path}")
    print(f"values         : {values_path}")


if __name__ == "__main__":
    main()
