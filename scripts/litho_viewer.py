#!/usr/bin/env python3
"""
litho_viewer.py — 光刻仿真结果通用可视化工具
打包后用法（exe）:
  litho_viewer.exe --mode matrix  --files mask.txt aerial.txt wafer.txt --labels Mask Aerial Wafer --save out.png
  litho_viewer.exe --mode profile --files profile_mask.txt profile_aerial.txt --save out.png
  litho_viewer.exe --mode ep      --files eps_others.txt --mask mask.txt --save out.png
  litho_viewer.exe --mode matrix  --files result.txt --title "My Result" --cmap inferno
  litho_viewer.exe --mode compare --files before.txt after.txt --labels Before After --save diff.png

全局参数:
  --save  [路径]   保存 PNG（不弹窗），不加则弹交互窗口
  --dpi   整数     保存分辨率，默认 200
  --title 字符串   图形总标题
"""

import argparse
import sys
import os
import numpy as np
import matplotlib
# 无 DISPLAY 时也能保存 PNG
matplotlib.use("Agg" if "--save" in sys.argv else "TkAgg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─────────────────────────────── 数据加载 ───────────────────────────────────

def load_matrix(path: str) -> np.ndarray:
    """加载空格/制表符分隔的矩阵 txt，忽略 # 注释行"""
    if not os.path.isfile(path):
        sys.exit(f"[litho_viewer] 文件不存在: {path}")
    return np.loadtxt(path, comments="#", ndmin=2)


def parse_header(path: str):
    """尝试从第一行读取 # rows=N cols=M，返回 (rows, cols) 或 None"""
    with open(path, "r") as f:
        first = f.readline().strip()
    if first.startswith("#"):
        import re
        m = re.search(r"rows=(\d+).*cols=(\d+)", first)
        if m:
            return int(m.group(1)), int(m.group(2))
    return None

# ─────────────────────────────── 各模式绘图 ─────────────────────────────────

def mode_matrix(args):
    """显示一组二维矩阵（掩模/aerial/wafer/任意灰度图）"""
    files = args.files
    labels = args.labels if args.labels else [os.path.basename(f) for f in files]
    cmaps  = args.cmaps  if args.cmaps  else ["gray"] * len(files)
    # 补齐 cmap 列表
    while len(cmaps) < len(files):
        cmaps.append("inferno")

    n = len(files)
    ncols = min(n, 4)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), squeeze=False)
    if args.title:
        fig.suptitle(args.title, fontsize=13)

    for i, (fpath, label, cmap) in enumerate(zip(files, labels, cmaps)):
        data = load_matrix(fpath)
        r, c = i // ncols, i % ncols
        ax = axes[r][c]
        im = ax.imshow(data, cmap=cmap, interpolation="none")
        ax.set_title(label, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 隐藏多余子图
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    plt.tight_layout()
    _output(fig, args)


def mode_profile(args):
    """绘制 1D 剖面曲线，每个文件一列数据或两列 (x, y)"""
    files  = args.files
    labels = args.labels if args.labels else [os.path.basename(f) for f in files]
    colors = ["k", "r", "b", "g", "m", "c"]

    fig, ax = plt.subplots(figsize=(10, 5))
    if args.title:
        ax.set_title(args.title, fontsize=12)

    for i, (fpath, label) in enumerate(zip(files, labels)):
        data = load_matrix(fpath)
        if data.ndim == 1 or data.shape[1] == 1:
            y = data.ravel()
            x = np.arange(len(y))
        else:
            x, y = data[:, 0], data[:, 1]

        ax.plot(x, y, color=colors[i % len(colors)], label=label, lw=1.5)

    if args.threshold is not None:
        ax.axhline(args.threshold, color="gray", ls=":", label=f"threshold={args.threshold}")

    ax.set_xlabel("x")
    ax.set_ylabel("value")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _output(fig, args)


def mode_ep(args):
    """可视化 EP 点（evaluation points）叠加在掩模图上"""
    if not args.files:
        sys.exit("[litho_viewer] --mode ep 需要通过 --files 提供 eps txt 文件")

    eps_data = load_matrix(args.files[0])
    # 列顺序：y x w_epe w_meef
    if eps_data.shape[1] < 4:
        sys.exit("[litho_viewer] ep 文件应有 4 列: y x w_epe w_meef")

    y, x, w_epe, w_meef = eps_data[:,0], eps_data[:,1], eps_data[:,2], eps_data[:,3]

    fig, ax = plt.subplots(figsize=(8, 8))
    title = args.title or "EP Select Result"
    ax.set_title(title, fontsize=12)

    if args.mask:
        mask = load_matrix(args.mask)
        N = mask.shape[0]
        ax.imshow(mask, cmap="gray", vmin=0, vmax=1,
                  extent=[-0.5, mask.shape[1]-0.5, mask.shape[0]-0.5, -0.5])
    else:
        N = int(max(y.max(), x.max())) + 2

    core   = w_epe > 0.5
    others = ~core

    if others.any():
        ax.scatter(x[others], y[others], s=30, c="dodgerblue",
                   marker="o", linewidths=0.5, edgecolors="white",
                   label=f"w_epe=0 (n={others.sum()})", zorder=3)
    if core.any():
        ax.scatter(x[core], y[core], s=60, c="red",
                   marker="o", linewidths=0.5, edgecolors="white",
                   label=f"w_epe=1 (n={core.sum()})", zorder=4)

    ax.set_xlim(-0.5, N - 0.5)
    ax.set_ylim(N - 0.5, -0.5)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")

    if len(eps_data) <= 60:
        for i in range(len(eps_data)):
            ax.annotate(f"({int(y[i])},{int(x[i])})",
                        (x[i], y[i]), textcoords="offset points",
                        xytext=(4, 4), fontsize=5, color="yellow", zorder=5)

    plt.tight_layout()
    _output(fig, args)


def mode_compare(args):
    """并排对比两张矩阵（before / after）"""
    if len(args.files) < 2:
        sys.exit("[litho_viewer] --mode compare 需要至少 2 个文件")

    files  = args.files[:2]
    labels = (args.labels or [])[:2]
    while len(labels) < 2:
        labels.append(os.path.basename(files[len(labels)]))
    cmaps = (args.cmaps or ["gray","gray"])[:2]
    while len(cmaps) < 2:
        cmaps.append("gray")

    a = load_matrix(files[0])
    b = load_matrix(files[1])

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    if args.title:
        fig.suptitle(args.title, fontsize=13)

    for ax, data, lbl, cm in zip(axes[:2], [a, b], labels, cmaps):
        im = ax.imshow(data, cmap=cm, interpolation="none")
        ax.set_title(lbl, fontsize=10)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 差值图
    diff = b.astype(float) - a.astype(float)
    im3 = axes[2].imshow(diff, cmap="bwr",
                         vmin=-np.abs(diff).max(), vmax=np.abs(diff).max(),
                         interpolation="none")
    axes[2].set_title("Diff (after - before)", fontsize=10)
    axes[2].axis("off")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    _output(fig, args)


# ─────────────────────────────── 输出 ───────────────────────────────────────

def _output(fig, args):
    if args.save:
        out_path = args.save if isinstance(args.save, str) and args.save != "True" else "litho_out.png"
        fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
        print(f"[litho_viewer] saved: {out_path}")
    else:
        matplotlib.use("TkAgg")
        plt.show()

# ─────────────────────────────── 主入口 ─────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="litho_viewer",
        description="光刻仿真结果通用可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode",   choices=["matrix","profile","ep","compare"],
                        default="matrix", help="可视化模式")
    parser.add_argument("--files",  nargs="+", metavar="FILE",
                        help="输入 txt 文件（一个或多个）")
    parser.add_argument("--labels", nargs="+", metavar="LABEL",
                        help="对应每个文件的标签")
    parser.add_argument("--cmaps",  nargs="+", metavar="CMAP",
                        help="colormap，如 gray inferno hot（每个文件一个）")
    parser.add_argument("--mask",   metavar="FILE",
                        help="ep 模式下背景掩模文件")
    parser.add_argument("--title",  metavar="STR",
                        help="图形总标题")
    parser.add_argument("--save",   nargs="?", const="litho_out.png", metavar="PATH",
                        help="保存路径，默认 litho_out.png")
    parser.add_argument("--dpi",    type=int, default=200,
                        help="保存 DPI，默认 200")
    parser.add_argument("--threshold", type=float, default=None,
                        help="profile 模式下画阈值水平线")

    args = parser.parse_args()

    if not args.files:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "matrix":  mode_matrix,
        "profile": mode_profile,
        "ep":      mode_ep,
        "compare": mode_compare,
    }
    dispatch[args.mode](args)


if __name__ == "__main__":
    main()
