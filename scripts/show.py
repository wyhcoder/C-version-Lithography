#!/usr/bin/env python3
"""实时弹窗显示 txt 矩阵，看完关窗口即继续 C++ 程序。

用法（在 C++ 里调用）:
    system("python3 scripts/show.py result/temp.txt 'temp mask'");
    system("python3 scripts/show.py result/wafer_image.txt 'wafer' hot");
"""
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    args = sys.argv[1:]
    if not args:
        print("用法: show.py <file.txt> [title] [cmap]")
        return
    fpath = args[0]
    title = args[1] if len(args) > 1 else Path(fpath).stem
    cmap  = args[2] if len(args) > 2 else "hot"

    arr = np.loadtxt(fpath, comments="#")
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(arr, cmap=cmap, interpolation="nearest")
    ax.set_title(f"{title}\nshape={arr.shape} min={arr.min():.4f} max={arr.max():.4f}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.show()   # 阻塞，关窗口后 C++ 继续

if __name__ == "__main__":
    main()
