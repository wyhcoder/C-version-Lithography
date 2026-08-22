#!/usr/bin/env python3
"""Convert NumPy control-point contours into the C++ MEEF text format.

The historical Python MEEF pipeline stores ragged contours as an object-dtype
``.npy`` file. That payload is a Python pickle, so the C++ demo performs this
small, explicit conversion before handing the points to the native optimizer.
Coordinates are preserved in the project's native ``(y, x)`` order.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _as_contours(value: np.ndarray) -> list[np.ndarray]:
    array = np.asarray(value)

    if array.ndim == 2 and array.shape[1] == 2:
        candidates = [array]
    elif array.ndim == 3 and array.shape[2] == 2:
        candidates = [array[index] for index in range(array.shape[0])]
    elif array.ndim == 1:
        candidates = [np.asarray(item) for item in array]
    else:
        raise ValueError(
            "expected (N,2), (C,N,2), or a ragged object array of (N,2) contours; "
            f"got shape={array.shape}, dtype={array.dtype}"
        )

    contours: list[np.ndarray] = []
    for index, candidate in enumerate(candidates):
        contour = np.asarray(candidate, dtype=np.float64)
        if contour.ndim != 2 or contour.shape[1] != 2:
            raise ValueError(
                f"contour {index} must have shape (N,2); got {contour.shape}"
            )
        if contour.shape[0] < 2:
            raise ValueError(f"contour {index} has fewer than 2 points")
        if not np.isfinite(contour).all():
            raise ValueError(f"contour {index} contains NaN or infinity")
        contours.append(contour)
    return contours


def convert(input_path: Path, output_path: Path) -> tuple[int, int]:
    # allow_pickle is required for the ragged object arrays produced by the
    # Python MEEF project. The input path is an explicit user YAML setting.
    contours = _as_contours(np.load(input_path, allow_pickle=True))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    point_count = 0
    with output_path.open("w", encoding="utf-8") as output:
        output.write("# litho-control-points-v1\n")
        output.write("# coordinate_order: y x\n")
        for index, contour in enumerate(contours):
            output.write(f"# contour {index}\n")
            np.savetxt(output, contour, fmt="%.17g")
            output.write("\n")
            point_count += int(contour.shape[0])
    return len(contours), point_count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contour_count, point_count = convert(args.input, args.output)
    print(
        f"converted {contour_count} contours / {point_count} points: "
        f"{args.input} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
