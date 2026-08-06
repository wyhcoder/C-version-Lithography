#!/usr/bin/env python3
"""实时弹窗显示多个 txt 矩阵到一张图上，看完关窗口即继续 C++ 程序。

用法:
    system("python3 scripts/show_multi.py phi_M seismic temp gray wafer hot");
    system("python3 scripts/show_multi.py phi_M seismic temp gray wafer hot --title_prefix 'iter 1'");
"""
import os
import sys
from pathlib import Path

# 在项目内缓存 Matplotlib 配置，避免受系统用户目录权限或当前工作目录影响。
_cache_dir = Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_cache_dir.parent))
os.environ.setdefault("MPLCONFIGDIR", str(_cache_dir))

import numpy as np
import matplotlib.pyplot as plt

def main():
    args = sys.argv[1:]
    # 解析 --title_prefix
    title_prefix = ""
    positional = []
    i = 0
    while i < len(args):
        if args[i] == "--title_prefix" and i + 1 < len(args):
            title_prefix = args[i+1] + " "
            i += 2
        else:
            positional.append(args[i])
            i += 1

    if len(positional) < 2:
        print("用法: show_multi.py <file1> <cmap1> [<file2> <cmap2> ...] [--title_prefix 'iter N']")
        return

    # positional 是 file/cmap 交替对
    base = "/Users/wyh/Desktop/学校/Litho_cpp/result/"
    pairs = []
    for j in range(0, len(positional), 2):
        fname = positional[j]
        if "/" not in fname:
            fname = base + fname
        cmap = positional[j+1] if j+1 < len(positional) else "hot"
        pairs.append((fname, cmap))

    n = len(pairs)
    cols = min(n, 3)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4.5*cols, 4.5*rows))
    axes = np.array(axes).reshape(-1)

    for idx, (fname, cmap) in enumerate(pairs):
        arr = np.loadtxt(fname, comments="#")
        title = title_prefix + Path(fname).stem
        im = axes[idx].imshow(arr, cmap=cmap, interpolation="nearest")
        axes[idx].set_title(f"{title}\nmin={arr.min():.4f} max={arr.max():.4f}")
        axes[idx].axis("off")
        fig.colorbar(im, ax=axes[idx], fraction=0.046, pad=0.04)

    for idx in range(n, len(axes)):
        axes[idx].axis("off")

    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
