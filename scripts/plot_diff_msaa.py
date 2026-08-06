"""plot_diff_msaa.py
比较 C++ MSAA 和 Python MSAA 输出的 gray mask 差异。

用法：
    python3 scripts/plot_diff_msaa.py \
        --cpp /path/to/msaa_cp_gray.txt \
        --py  /path/to/gray_mask.txt \
        --save diff_msaa.png
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_matrix(path: str) -> np.ndarray:
    return np.loadtxt(path, comments="#")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpp",  required=True, help="C++ MSAA 输出 txt")
    ap.add_argument("--py",   required=True, help="Python MSAA 输出 txt")
    ap.add_argument("--save", default="diff_msaa.png")
    ap.add_argument("--dpi",  type=int, default=200)
    args = ap.parse_args()

    cpp = load_matrix(args.cpp)
    py  = load_matrix(args.py)

    print(f"[diff] cpp shape={cpp.shape}  min={cpp.min():.6f} max={cpp.max():.6f}")
    print(f"[diff] py  shape={py.shape}   min={py.min():.6f}  max={py.max():.6f}")

    if cpp.shape != py.shape:
        print(f"[error] shape 不一致！cpp={cpp.shape} vs py={py.shape}")
        return

    diff = cpp - py
    abs_diff = np.abs(diff)

    # 统计
    print("\n--- 差异统计 (cpp - py) ---")
    print(f"  max abs  = {abs_diff.max():.6f}")
    print(f"  mean abs = {abs_diff.mean():.6f}")
    print(f"  rms      = {np.sqrt((diff**2).mean()):.6f}")
    print(f"  L1 sum   = {abs_diff.sum():.4f}")
    print(f"  像素差>0       : {(abs_diff > 0).sum()}  ({100*(abs_diff>0).mean():.2f}%)")
    print(f"  像素差>0.001   : {(abs_diff > 0.001).sum()}")
    print(f"  像素差>0.01    : {(abs_diff > 0.01).sum()}")
    print(f"  像素差>0.05    : {(abs_diff > 0.05).sum()}")
    print(f"  像素差>0.1     : {(abs_diff > 0.1).sum()}")

    # 画图：3 列 = cpp / py / diff
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    im0 = axes[0].imshow(cpp, cmap="hot", origin="upper", vmin=0, vmax=1)
    axes[0].set_title(f"C++ MSAA (max={cpp.max():.3f})")
    axes[0].set_axis_off()
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(py, cmap="hot", origin="upper", vmin=0, vmax=1)
    axes[1].set_title(f"Python MSAA (max={py.max():.3f})")
    axes[1].set_axis_off()
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # diff 用发散 colormap，蓝(cpp<py) - 白(0) - 红(cpp>py)
    vmax = max(abs_diff.max(), 1e-6)
    im2 = axes[2].imshow(diff, cmap="bwr", origin="upper",
                         vmin=-vmax, vmax=vmax)
    axes[2].set_title(f"diff = cpp - py\n(max|d|={abs_diff.max():.4f}, "
                      f"mean|d|={abs_diff.mean():.4f})")
    axes[2].set_axis_off()
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    fig.tight_layout()
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[plot] saved: {args.save}")


if __name__ == "__main__":
    main()
