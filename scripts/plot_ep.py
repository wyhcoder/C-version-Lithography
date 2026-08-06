#!/usr/bin/env python3
"""EP Select 结果可视化
用法: python3 plot_ep.py          → 显示交互窗口
     python3 plot_ep.py --save    → 保存 png
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys, os

SAVE = "--save" in sys.argv
# 切换到 build 目录（txt 文件在那里）
BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
os.chdir(BUILD_DIR)

# ── 读取数据 ──────────────────────────────────────────────────────────────
def load_eps(fname):
    """读取 eps txt，列：y x w_epe w_meef"""
    return np.loadtxt(fname, comments="#")

eps_others = load_eps("eps_others.txt")   # line/rect 图案
eps_via    = load_eps("eps_via.txt")      # via 孔图案

# ── 重建 mask（与 demo 里一样）────────────────────────────────────────────
N = 64
cy, cx = N//2, N//2

mask_rect = np.zeros((N, N))
mask_rect[cy-10:cy+11, cx-5:cx+6] = 1.0

mask_via = np.zeros((N, N))
mask_via[cy-5:cy+6, cx-5:cx+6] = 1.0

# ── 绘图 ─────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle("EP Select Results", fontsize=14)

def draw_ep(ax, mask, eps_data, title):
    ax.imshow(mask, cmap="gray", vmin=0, vmax=1,
              extent=[-0.5, mask.shape[1]-0.5, mask.shape[0]-0.5, -0.5])

    # 分两类：w_epe=1（核心点，红色大圆）和 w_epe=0（其余点，蓝色小圆）
    y, x, w_epe, w_meef = eps_data[:,0], eps_data[:,1], eps_data[:,2], eps_data[:,3]
    core   = w_epe > 0.5
    others = ~core

    if others.any():
        ax.scatter(x[others], y[others], s=30, c="dodgerblue",
                   marker="o", linewidths=0.5, edgecolors="white",
                   label=f"w_epe=0  (n={others.sum()})", zorder=3)
    if core.any():
        ax.scatter(x[core], y[core], s=60, c="red",
                   marker="o", linewidths=0.5, edgecolors="white",
                   label=f"w_epe=1  (n={core.sum()})", zorder=4)

    # 用颜色深浅显示 w_meef 大小
    sc = ax.scatter(x, y, s=0)   # 占位，用于 colorbar 参考
    norm = plt.Normalize(w_meef.min(), w_meef.max())

    ax.set_title(title)
    ax.set_xlim(-0.5, mask.shape[1]-0.5)
    ax.set_ylim(mask.shape[0]-0.5, -0.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")

    # 标注每个 EP 点的坐标（可选，图少时好看）
    if len(eps_data) <= 40:
        for i in range(len(eps_data)):
            ax.annotate(f"({int(y[i])},{int(x[i])})",
                        (x[i], y[i]), textcoords="offset points",
                        xytext=(4, 4), fontsize=5, color="yellow", zorder=5)

draw_ep(axes[0], mask_rect, eps_others,
        f"Rect Pattern - select_eps_others  (N={len(eps_others)})")
draw_ep(axes[1], mask_via,   eps_via,
        f"Via Pattern  - select_eps_via     (N={len(eps_via)})")

plt.tight_layout()

if SAVE:
    out = os.path.join(os.path.dirname(BUILD_DIR), "ep_select_result.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"已保存: {out}")
else:
    plt.show()
