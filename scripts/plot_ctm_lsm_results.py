#!/usr/bin/env python3
"""Visualize target, CTM/LSM masks, and the CTM-binary wafer image.

Run without arguments from the project root.  The script reads the current
result/System_result, result/CTM_result, and result/LSM_result text matrices,
then saves one comparison figure under result/.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class ResultMatrices:
    target: np.ndarray
    ctm_gray_mask: np.ndarray
    ctm_binary_mask: np.ndarray
    lsm_mask: np.ndarray
    ctm_binary_wafer: np.ndarray


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_matrix(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"result matrix not found: {path}")
    matrix = np.loadtxt(path)
    if matrix.ndim != 2 or min(matrix.shape) < 2:
        raise ValueError(f"expected a two-dimensional matrix in {path}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"matrix contains non-finite values: {path}")
    return matrix


def load_results(result_root: Path) -> ResultMatrices:
    system_dir = result_root / "System_result"
    ctm_dir = result_root / "CTM_result"
    lsm_dir = result_root / "LSM_result"
    matrices = ResultMatrices(
        target=load_matrix(system_dir / "target_mask.txt"),
        ctm_gray_mask=load_matrix(ctm_dir / "gray_mask.txt"),
        ctm_binary_mask=load_matrix(ctm_dir / "binary_mask.txt"),
        lsm_mask=load_matrix(lsm_dir / "LSM_mask.txt"),
        ctm_binary_wafer=load_matrix(ctm_dir / "wafer_image_B.txt"),
    )

    expected_shape = matrices.target.shape
    for name, matrix in matrices.__dict__.items():
        if matrix.shape != expected_shape:
            raise ValueError(
                f"shape mismatch: target is {expected_shape}, {name} is {matrix.shape}"
            )
    return matrices


def configure_pixel_axis(axis: plt.Axes, shape: tuple[int, int]) -> None:
    rows, columns = shape
    axis.set_xlim(-0.5, columns - 0.5)
    axis.set_ylim(rows - 0.5, -0.5)
    axis.set_xlabel("x (pixel)")
    axis.set_ylabel("y (pixel)")
    axis.set_aspect("equal", adjustable="box")


def show_matrix(axis: plt.Axes, matrix: np.ndarray, title: str) -> None:
    axis.imshow(
        matrix,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        origin="upper",
        interpolation="nearest",
    )
    axis.set_title(title)
    configure_pixel_axis(axis, matrix.shape)


def plot_results(matrices: ResultMatrices, output_path: Path) -> None:
    figure = plt.figure(figsize=(11.2, 7.3), constrained_layout=True)
    grid = figure.add_gridspec(2, 6)
    axes = (
        figure.add_subplot(grid[0, 0:2]),
        figure.add_subplot(grid[0, 2:4]),
        figure.add_subplot(grid[0, 4:6]),
        figure.add_subplot(grid[1, 1:3]),
        figure.add_subplot(grid[1, 3:5]),
    )
    figure.suptitle("CTM and LSM optimization results", fontsize=17)

    show_matrix(axes[0], matrices.target, "Target layout")
    show_matrix(axes[1], matrices.ctm_gray_mask, "CTM optimized mask (gray)")
    show_matrix(axes[2], matrices.ctm_binary_mask, "CTM binarized mask")
    show_matrix(axes[3], matrices.lsm_mask, "LSM optimized mask")
    show_matrix(axes[4], matrices.ctm_binary_wafer, "Wafer from CTM binary mask")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(figure)


def binary_pixel_count(matrix: np.ndarray) -> int:
    return int(np.count_nonzero(matrix >= 0.5))


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=root / "result",
        help="root containing System_result, CTM_result, and LSM_result",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="output PNG; defaults to result/ctm_lsm_result_overview.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_root = args.result_root.resolve()
    output_path = (
        args.output.resolve()
        if args.output
        else result_root / "ctm_lsm_result_overview.png"
    )
    matrices = load_results(result_root)
    plot_results(matrices, output_path)

    print(f"matrix shape          : {matrices.target.shape}")
    print(f"target foreground     : {binary_pixel_count(matrices.target)} pixels")
    print(f"CTM binary foreground : {binary_pixel_count(matrices.ctm_binary_mask)} pixels")
    print(f"LSM foreground        : {binary_pixel_count(matrices.lsm_mask)} pixels")
    print(f"saved figure          : {output_path}")


if __name__ == "__main__":
    main()
