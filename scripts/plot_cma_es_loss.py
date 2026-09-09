#!/usr/bin/env python3
"""Plot the CMA-ES candidate loss and best-so-far convergence curve."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = PROJECT_ROOT / ".cache" / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_loss_history(csv_path: Path) -> tuple[list[int], list[float]]:
    """Read the evaluation index and total normalized cost from a history CSV."""
    evaluations: list[int] = []
    costs: list[float] = []

    with csv_path.open(newline="", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or not {"evaluation", "cost"}.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                "history CSV must contain the 'evaluation' and 'cost' columns"
            )

        for row_number, row in enumerate(reader, start=2):
            try:
                evaluation = int(row["evaluation"])
                cost = float(row["cost"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"invalid value at CSV row {row_number}") from error
            if not math.isfinite(cost):
                raise ValueError(f"non-finite cost at CSV row {row_number}")
            evaluations.append(evaluation)
            costs.append(cost)

    if not evaluations:
        raise ValueError("history CSV contains no loss records")
    return evaluations, costs


def plot_loss_curve(csv_path: Path, output_path: Path) -> tuple[int, float]:
    """Create a PNG showing raw candidate costs and cumulative best cost."""
    evaluations, costs = read_loss_history(csv_path)
    best_so_far: list[float] = []
    current_best = math.inf
    for cost in costs:
        current_best = min(current_best, cost)
        best_so_far.append(current_best)

    best_index = min(range(len(costs)), key=costs.__getitem__)
    best_evaluation = evaluations[best_index]
    best_cost = costs[best_index]

    fig, axis = plt.subplots(figsize=(10.0, 5.8))
    axis.plot(
        evaluations,
        costs,
        color="#76A5D5",
        linewidth=0.8,
        alpha=0.42,
        label="Candidate cost",
        zorder=1,
    )
    axis.plot(
        evaluations,
        best_so_far,
        color="#D55E00",
        linewidth=2.2,
        label="Best-so-far cost",
        zorder=3,
    )
    axis.scatter(
        [best_evaluation],
        [best_cost],
        color="#D55E00",
        edgecolor="white",
        linewidth=0.8,
        s=48,
        zorder=4,
    )
    axis.annotate(
        f"best = {best_cost:.6f}\neval = {best_evaluation}",
        xy=(best_evaluation, best_cost),
        xytext=(-14, 24),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.8},
    )

    axis.set_title("CMA-ES Loss Convergence")
    axis.set_xlabel("Objective-function evaluation")
    axis.set_ylabel("Normalized joint cost")
    axis.grid(True, color="#D9D9D9", linewidth=0.7, alpha=0.75)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    axis.margins(x=0.02, y=0.08)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return best_evaluation, best_cost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the loss convergence recorded by SRAF CMA-ES."
    )
    parser.add_argument("--input", required=True, type=Path, help="history CSV path")
    parser.add_argument("--output", required=True, type=Path, help="output PNG path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    best_evaluation, best_cost = plot_loss_curve(args.input, args.output)
    print(
        f"Saved CMA-ES loss curve: {args.output} "
        f"(best={best_cost:.6f} at evaluation {best_evaluation})"
    )


if __name__ == "__main__":
    main()
