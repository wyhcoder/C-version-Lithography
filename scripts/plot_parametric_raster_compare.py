#!/usr/bin/env python3
"""Visualize MSAA and paper Dirac rasterization from identical curve points."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.input_dir / "rasterizer_comparison.png"
    msaa = np.loadtxt(args.input_dir / "mask_msaa.txt")
    dirac = np.loadtxt(args.input_dir / "mask_dirac.txt")
    difference = dirac - msaa
    controls = np.loadtxt(args.input_dir / "control_points_yx.txt")
    curve = np.loadtxt(args.input_dir / "curve_points_yx.txt")

    if msaa.shape != dirac.shape:
        raise ValueError(f"mask shapes differ: {msaa.shape} vs {dirac.shape}")

    max_row, max_col = np.unravel_index(np.argmax(np.abs(difference)), difference.shape)
    radius = 10
    row0 = max(0, max_row - radius)
    row1 = min(msaa.shape[0], max_row + radius + 1)
    col0 = max(0, max_col - radius)
    col1 = min(msaa.shape[1], max_col + radius + 1)
    extent = (col0 - 0.5, col1 - 0.5, row1 - 0.5, row0 - 0.5)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10), constrained_layout=True)
    panels = [
        (axes[0, 0], msaa, "Original render_curve: 16x MSAA", "gray", 0.0, 1.0),
        (axes[0, 1], dirac, "New render_curve_dirac: h/4 boundary segments", "gray", 0.0, 1.0),
        (
            axes[0, 2],
            difference,
            "Difference: Dirac - MSAA",
            "coolwarm",
            -max(np.max(np.abs(difference)), 1e-12),
            max(np.max(np.abs(difference)), 1e-12),
        ),
    ]
    for index, (axis, image, title, cmap, vmin, vmax) in enumerate(panels):
        handle = axis.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
        axis.set_title(title)
        axis.set_xlabel("x (pixel)")
        axis.set_ylabel("y (pixel)")
        if index < 2:
            closed_curve = np.vstack([curve, curve[0]])
            axis.plot(closed_curve[:, 1], closed_curve[:, 0], color="tab:cyan", linewidth=0.8)
            axis.scatter(
                controls[:, 1], controls[:, 0],
                color="tab:orange", edgecolor="black", linewidth=0.4, s=24,
                label="same control points",
            )
            axis.legend(loc="upper right")
        fig.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)

    zoom_panels = [
        (axes[1, 0], msaa[row0:row1, col0:col1], "MSAA boundary zoom", "gray", 0.0, 1.0),
        (axes[1, 1], dirac[row0:row1, col0:col1], "Dirac boundary zoom", "gray", 0.0, 1.0),
        (
            axes[1, 2],
            difference[row0:row1, col0:col1],
            f"Difference zoom; max at ({max_row}, {max_col})",
            "coolwarm",
            -max(np.max(np.abs(difference)), 1e-12),
            max(np.max(np.abs(difference)), 1e-12),
        ),
    ]
    for axis, image, title, cmap, vmin, vmax in zoom_panels:
        handle = axis.imshow(
            image,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            origin="upper",
            interpolation="nearest",
            extent=extent,
        )
        axis.set_title(title)
        axis.set_xlabel("x (pixel)")
        axis.set_ylabel("y (pixel)")
        axis.set_xticks(np.arange(col0, col1, 4))
        axis.set_yticks(np.arange(row0, row1, 4))
        fig.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)

    binary_iou = np.count_nonzero((msaa >= 0.5) & (dirac >= 0.5)) / np.count_nonzero(
        (msaa >= 0.5) | (dirac >= 0.5)
    )
    fig.suptitle(
        "Same B-spline geometry, different rasterizers | "
        f"MAE={np.mean(np.abs(difference)):.6f}, "
        f"RMSE={np.sqrt(np.mean(difference**2)):.6f}, IoU={binary_iou:.6f}"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
