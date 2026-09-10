#!/usr/bin/env python3
"""Compare two completed MEEF runs that differ only by curve rasterizer."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_history(run_dir: Path) -> np.ndarray:
    return np.genfromtxt(run_dir / "errors.csv", delimiter=",", names=True)


def load_best(run_dir: Path, name: str) -> np.ndarray:
    return np.loadtxt(run_dir / "best_epe" / name)


def matrix_metrics(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    difference = a - b
    return {
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference * difference))),
        "max_abs": float(np.max(np.abs(difference))),
    }


def binary_metrics(a: np.ndarray, b: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    a_binary = a >= threshold
    b_binary = b >= threshold
    union = np.logical_or(a_binary, b_binary).sum()
    return {
        "iou": float(np.logical_and(a_binary, b_binary).sum() / max(1, union)),
        "different_pixels": float(np.count_nonzero(a_binary != b_binary)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--msaa", required=True, type=Path)
    parser.add_argument("--dirac", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    msaa_history = load_history(args.msaa)
    dirac_history = load_history(args.dirac)

    msaa_mask = load_best(args.msaa, "mask.txt")
    dirac_mask = load_best(args.dirac, "mask.txt")
    msaa_wafer = load_best(args.msaa, "wafer.txt")
    dirac_wafer = load_best(args.dirac, "wafer.txt")
    msaa_cps = load_best(args.msaa, "control_points.txt")
    dirac_cps = load_best(args.dirac, "control_points.txt")

    if msaa_mask.shape != dirac_mask.shape or msaa_cps.shape != dirac_cps.shape:
        raise ValueError("The two runs do not have matching matrix/control-point shapes")

    mask_stats = matrix_metrics(msaa_mask, dirac_mask)
    wafer_stats = matrix_metrics(msaa_wafer, dirac_wafer)
    mask_binary_stats = binary_metrics(msaa_mask, dirac_mask)
    wafer_binary_stats = binary_metrics(msaa_wafer, dirac_wafer)
    cp_distance = np.linalg.norm(msaa_cps - dirac_cps, axis=1)

    last_msaa = msaa_history[-1]
    last_dirac = dirac_history[-1]
    summary = {
        "msaa_final_epe": float(last_msaa["epe"]),
        "dirac_final_epe": float(last_dirac["epe"]),
        "msaa_final_wepe": float(last_msaa["wepe"]),
        "dirac_final_wepe": float(last_dirac["wepe"]),
        "msaa_final_pe": float(last_msaa["pe"]),
        "dirac_final_pe": float(last_dirac["pe"]),
        "msaa_time_seconds": float(last_msaa["time_seconds"]),
        "dirac_time_seconds": float(last_dirac["time_seconds"]),
        "speedup_msaa_over_dirac": float(
            last_msaa["time_seconds"] / last_dirac["time_seconds"]
        ),
        "dirac_time_reduction_percent": float(
            100.0
            * (last_msaa["time_seconds"] - last_dirac["time_seconds"])
            / last_msaa["time_seconds"]
        ),
        "dirac_epe_reduction_percent": float(
            100.0 * (last_msaa["epe"] - last_dirac["epe"]) / last_msaa["epe"]
        ),
        "dirac_wepe_reduction_percent": float(
            100.0 * (last_msaa["wepe"] - last_dirac["wepe"]) / last_msaa["wepe"]
        ),
        "dirac_pe_reduction_percent": float(
            100.0 * (last_msaa["pe"] - last_dirac["pe"]) / last_msaa["pe"]
        ),
        "mask_mae": mask_stats["mae"],
        "mask_rmse": mask_stats["rmse"],
        "mask_max_abs": mask_stats["max_abs"],
        "mask_binary_iou_at_0_5": mask_binary_stats["iou"],
        "mask_binary_different_pixels_at_0_5": mask_binary_stats["different_pixels"],
        "wafer_mae": wafer_stats["mae"],
        "wafer_rmse": wafer_stats["rmse"],
        "wafer_max_abs": wafer_stats["max_abs"],
        "wafer_binary_iou_at_0_5": wafer_binary_stats["iou"],
        "wafer_binary_different_pixels_at_0_5": wafer_binary_stats[
            "different_pixels"
        ],
        "control_point_mean_distance_pixel": float(np.mean(cp_distance)),
        "control_point_max_distance_pixel": float(np.max(cp_distance)),
    }

    with (args.output_dir / "comparison_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value"])
        writer.writerows(summary.items())

    figure, axes = plt.subplots(3, 3, figsize=(15, 14), constrained_layout=True)
    image_extent = [0, msaa_mask.shape[1], msaa_mask.shape[0], 0]

    for axis, image, title in (
        (axes[0, 0], msaa_mask, "MSAA best-EPE mask"),
        (axes[0, 1], dirac_mask, "Dirac best-EPE mask"),
    ):
        shown = axis.imshow(image, cmap="gray", vmin=0.0, vmax=1.0, extent=image_extent)
        axis.set_title(title)
        figure.colorbar(shown, ax=axis, fraction=0.046)
    mask_diff = np.abs(msaa_mask - dirac_mask)
    shown = axes[0, 2].imshow(mask_diff, cmap="magma", vmin=0.0, extent=image_extent)
    axes[0, 2].set_title(f"|Mask difference|  MAE={mask_stats['mae']:.4f}")
    figure.colorbar(shown, ax=axes[0, 2], fraction=0.046)

    wafer_min = min(float(msaa_wafer.min()), float(dirac_wafer.min()))
    wafer_max = max(float(msaa_wafer.max()), float(dirac_wafer.max()))
    for axis, image, title in (
        (axes[1, 0], msaa_wafer, "MSAA best-EPE wafer"),
        (axes[1, 1], dirac_wafer, "Dirac best-EPE wafer"),
    ):
        shown = axis.imshow(
            image, cmap="gray", vmin=wafer_min, vmax=wafer_max, extent=image_extent
        )
        axis.set_title(title)
        figure.colorbar(shown, ax=axis, fraction=0.046)
    wafer_diff = np.abs(msaa_wafer - dirac_wafer)
    shown = axes[1, 2].imshow(wafer_diff, cmap="magma", vmin=0.0, extent=image_extent)
    axes[1, 2].set_title(f"|Wafer difference|  MAE={wafer_stats['mae']:.4f}")
    figure.colorbar(shown, ax=axes[1, 2], fraction=0.046)

    for field, label, axis in (
        ("epe", "Mean EPE", axes[2, 0]),
        ("pe", "PE", axes[2, 1]),
        ("time_seconds", "Cumulative time (s)", axes[2, 2]),
    ):
        axis.plot(msaa_history["iteration"], msaa_history[field], "o-", label="MSAA")
        axis.plot(dirac_history["iteration"], dirac_history[field], "s-", label="Dirac")
        axis.set_xlabel("Iteration (0 is shared LSM baseline)")
        axis.set_ylabel(label)
        axis.grid(alpha=0.3)
        axis.legend()

    figure.suptitle(
        "MEEF optimization: MSAA vs Dirac rasterization\n"
        f"190 main CPs, 287 SRAF CPs, 290 EPs, 10 iterations; "
        f"CP mean distance={np.mean(cp_distance):.3f} px",
        fontsize=14,
    )
    output_path = args.output_dir / "optimization_comparison.png"
    figure.savefig(output_path, dpi=180)
    plt.close(figure)

    for key, value in summary.items():
        print(f"{key},{value:.12g}")
    print(f"figure,{output_path}")


if __name__ == "__main__":
    main()
