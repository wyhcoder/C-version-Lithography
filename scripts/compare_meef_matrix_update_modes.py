#!/usr/bin/env python3
"""Run and compare initial-only versus every-iteration MEEF matrix updates."""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class RunResult:
    mode: str
    directory: Path
    history: dict[str, np.ndarray]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def replace_meef_fields(config_text: str, replacements: dict[str, str]) -> str:
    lines = config_text.splitlines()
    in_meef = False
    replaced: set[str] = set()
    for index, line in enumerate(lines):
        if re.fullmatch(r"meef:\s*", line):
            in_meef = True
            continue
        if in_meef and line and not line.startswith((" ", "#")):
            break
        if not in_meef:
            continue
        for key, value in replacements.items():
            if re.match(rf"  {re.escape(key)}\s*:", line):
                lines[index] = f"  {key}: {value}"
                replaced.add(key)
                break
    missing = set(replacements) - replaced
    if missing:
        raise ValueError(f"missing meef fields in base config: {sorted(missing)}")
    return "\n".join(lines) + "\n"


def load_history(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty optimization history: {path}")
    return {name: np.asarray([float(row[name]) for row in rows]) for name in rows[0]}


def run_mode(base_config: Path, executable: Path, output_root: Path, mode: str, iterations: int | None) -> RunResult:
    run_directory = output_root / mode
    if run_directory.exists():
        shutil.rmtree(run_directory)
    config_path = output_root / f"config_{mode}.yaml"
    relative_output = run_directory.relative_to(project_root()).as_posix()
    replacements = {
        "output_dir": f'"{relative_output}"',
        "meef_matrix_update_mode": f'"{mode}"',
    }
    if iterations is not None:
        replacements["iter"] = str(iterations)
    config_text = replace_meef_fields(base_config.read_text(encoding="utf-8"), replacements)
    config_path.write_text(config_text, encoding="utf-8")
    environment = os.environ.copy()
    cache_directory = project_root() / ".cache" / "matplotlib"
    cache_directory.mkdir(parents=True, exist_ok=True)
    environment["MPLBACKEND"] = "Agg"
    environment["MPLCONFIGDIR"] = str(cache_directory)
    subprocess.run([str(executable), str(config_path)], cwd=executable.parent, env=environment, check=True)
    return RunResult(mode, run_directory, load_history(run_directory / "errors.csv"))


def best_index(history: dict[str, np.ndarray], metric: str) -> int:
    values = history[metric]
    return int(np.argmin(values[1:]) + 1) if len(values) > 1 else 0


def write_summary(results: list[RunResult], output_root: Path) -> None:
    summary_path = output_root / "comparison_summary.csv"
    fieldnames = ["mode", "iterations", "best_epe_iteration", "best_epe", "best_wepe_iteration", "best_wepe", "final_epe", "final_wepe", "final_pe", "time_seconds"]
    with summary_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            history = result.history
            epe_index = best_index(history, "epe")
            wepe_index = best_index(history, "wepe")
            writer.writerow({
                "mode": result.mode,
                "iterations": int(history["iteration"][-1]),
                "best_epe_iteration": int(history["iteration"][epe_index]),
                "best_epe": f'{history["epe"][epe_index]:.10f}',
                "best_wepe_iteration": int(history["iteration"][wepe_index]),
                "best_wepe": f'{history["wepe"][wepe_index]:.10f}',
                "final_epe": f'{history["epe"][-1]:.10f}',
                "final_wepe": f'{history["wepe"][-1]:.10f}',
                "final_pe": f'{history["pe"][-1]:.10f}',
                "time_seconds": f'{history["time_seconds"][-1]:.10f}',
            })

    by_mode = {result.mode: result for result in results}
    every = by_mode["every_iteration"].history
    initial = by_mode["initial_only"].history
    every_best_epe = best_index(every, "epe")
    initial_best_epe = best_index(initial, "epe")
    every_best_wepe = best_index(every, "wepe")
    initial_best_wepe = best_index(initial, "wepe")
    mask_every = np.loadtxt(by_mode["every_iteration"].directory / "iterations" / "mask.txt")
    mask_initial = np.loadtxt(by_mode["initial_only"].directory / "iterations" / "mask.txt")
    wafer_every = np.loadtxt(by_mode["every_iteration"].directory / "iterations" / "wafer.txt")
    wafer_initial = np.loadtxt(by_mode["initial_only"].directory / "iterations" / "wafer.txt")
    mx_every = np.loadtxt(by_mode["every_iteration"].directory / "iterations" / "meef_mx.txt")
    mx_initial = np.loadtxt(by_mode["initial_only"].directory / "iterations" / "meef_mx.txt")
    my_every = np.loadtxt(by_mode["every_iteration"].directory / "iterations" / "meef_my.txt")
    my_initial = np.loadtxt(by_mode["initial_only"].directory / "iterations" / "meef_my.txt")
    report = [
        "MEEF matrix update comparison",
        f"every_iteration best EPE: {every['epe'][every_best_epe]:.10f} at iteration {int(every['iteration'][every_best_epe])}",
        f"initial_only best EPE: {initial['epe'][initial_best_epe]:.10f} at iteration {int(initial['iteration'][initial_best_epe])}",
        f"initial_only best EPE gap: {(initial['epe'][initial_best_epe] / every['epe'][every_best_epe] - 1.0) * 100.0:.6f}%",
        f"every_iteration best wEPE: {every['wepe'][every_best_wepe]:.10f} at iteration {int(every['iteration'][every_best_wepe])}",
        f"initial_only best wEPE: {initial['wepe'][initial_best_wepe]:.10f} at iteration {int(initial['iteration'][initial_best_wepe])}",
        f"initial_only best wEPE gap: {(initial['wepe'][initial_best_wepe] / every['wepe'][every_best_wepe] - 1.0) * 100.0:.6f}%",
        f"every_iteration final EPE: {every['epe'][-1]:.10f}",
        f"initial_only final EPE: {initial['epe'][-1]:.10f}",
        f"initial_only - every_iteration EPE: {initial['epe'][-1] - every['epe'][-1]:.10f}",
        f"every_iteration final wEPE: {every['wepe'][-1]:.10f}",
        f"initial_only final wEPE: {initial['wepe'][-1]:.10f}",
        f"initial_only - every_iteration wEPE: {initial['wepe'][-1] - every['wepe'][-1]:.10f}",
        f"every_iteration final PE: {every['pe'][-1]:.10f}",
        f"initial_only final PE: {initial['pe'][-1]:.10f}",
        f"initial_only - every_iteration PE: {initial['pe'][-1] - every['pe'][-1]:.10f}",
        f"every_iteration time: {every['time_seconds'][-1]:.10f} s",
        f"initial_only time: {initial['time_seconds'][-1]:.10f} s",
        f"speedup (every / initial): {every['time_seconds'][-1] / initial['time_seconds'][-1]:.6f}x",
        f"final mask MAE: {np.mean(np.abs(mask_initial - mask_every)):.10f}",
        f"final mask max absolute difference: {np.max(np.abs(mask_initial - mask_every)):.10f}",
        f"final wafer differing pixels: {np.count_nonzero(wafer_initial != wafer_every)}",
        f"Mx relative change from initial to dynamic iteration end: {np.linalg.norm(mx_every - mx_initial) / np.linalg.norm(mx_initial):.10f}",
        f"My relative change from initial to dynamic iteration end: {np.linalg.norm(my_every - my_initial) / np.linalg.norm(my_initial):.10f}",
    ]
    (output_root / "comparison_summary.txt").write_text("\n".join(report) + "\n", encoding="utf-8")


def plot_results(results: list[RunResult], output_path: Path) -> None:
    labels = {"every_iteration": "Rebuild every iteration", "initial_only": "Build once initially"}
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    specifications = (("epe", "Mean EPE", "EPE (nm)"), ("wepe", "Mean weighted EPE", "Weighted EPE (nm)"), ("pe", "Pattern error", "PE (pixel)"), ("time_seconds", "Cumulative optimization time", "Time (s)"))
    for axis, (metric, title, ylabel) in zip(axes.flat, specifications):
        for result in results:
            axis.plot(result.history["iteration"], result.history[metric], marker="o", markersize=3, linewidth=1.6, label=labels[result.mode])
        axis.set_title(title)
        axis.set_xlabel("Iteration")
        axis.set_ylabel(ylabel)
        if metric != "time_seconds":
            axis.set_yscale("log")
        axis.grid(True, alpha=0.3)
    axes[0, 0].legend()
    figure.suptitle("MEEF matrix update strategy comparison")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config.yaml")
    parser.add_argument("--executable", type=Path, default=root / "build-release" / "demo_MEEF_Optimizer_init")
    parser.add_argument("--output-root", type=Path, default=root / "result" / "MEEF_result" / "matrix_update_comparison")
    parser.add_argument("--iterations", type=int, default=None, help="Override meef.iter for both runs")
    arguments = parser.parse_args()
    config = arguments.config.resolve()
    executable = arguments.executable.resolve()
    output_root = arguments.output_root.resolve()
    output_root.relative_to(root)
    output_root.mkdir(parents=True, exist_ok=True)
    if arguments.iterations is not None and arguments.iterations <= 0:
        raise ValueError("--iterations must be positive")
    results = [run_mode(config, executable, output_root, mode, arguments.iterations) for mode in ("every_iteration", "initial_only")]
    write_summary(results, output_root)
    plot_results(results, output_root / "comparison.png")
    print(f"Comparison saved to {output_root}")


if __name__ == "__main__":
    main()
