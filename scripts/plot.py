#!/usr/bin/env python3
"""光刻仿真结果可视化
用法: python3 plot.py          → 显示交互窗口
     python3 plot.py --save    → 保存为 png 文件
"""
from winsound import PlaySound
import numpy as np
import matplotlib.pyplot as plt
import sys, os

SAVE = "--save" in sys.argv
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_mat(fname):
    """加载评论，# 开头的行会被自动跳过"""
    return np.loadtxt(fname, comments="#", ndmin=2)

# ── 所有文件清单 ──
files_2d = {
    "掩模（孤立线）":      ("mask_isolated.txt",   "aerial_isolated.txt",   "wafer_isolated.txt"),
    "掩模（密集 L/S）":    ("mask_dense.txt",       "aerial_dense.txt",      "wafer_dense.txt"),
}
profiles = {
    "孤立线": ("profile_mask_iso.txt", "profile_aerial_iso.txt", "profile_wafer_iso.txt"),
    "密集L/S": ("profile_mask_ls.txt", "profile_aerial_ls.txt", "profile_wafer_ls.txt"),
}

# ── 2D 子图 ──
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("光刻 Abbe 成像仿真结果", fontsize=14)

for row_idx, (title, (mask_f, aerial_f, wafer_f)) in enumerate(files_2d.items()):
    for col_idx, (fname, label, cmap) in enumerate([
        (mask_f, "Mask", "gray"),
        (aerial_f, "Aerial Image", "inferno"),
        (wafer_f, "Wafer Pattern", "gray"),
    ]):
        ax = axes[row_idx, col_idx]
        data = load_mat(fname)
        im = ax.imshow(data, cmap=cmap, interpolation="none")
        ax.set_title(label)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
if SAVE:
    plt.savefig("litho_results_2d.png", dpi=200, bbox_inches="tight")
    print("已保存 litho_results_2d.png")
else:
    plt.show()

# ── 1D 剖面 ──
fig2, axes2 = plt.subplots(2, 1, figsize=(12, 8))
fig2.suptitle("中心行剖面 (y=0)", fontsize=14)

for idx, (title, (p_mask, p_aerial, p_wafer)) in enumerate(profiles.items()):
    ax = axes2[idx]
    m  = load_mat(p_mask)
    ap = load_mat(p_aerial)
    wp = load_mat(p_wafer)

    x_nm = (m[:, 0] - m.shape[0] / 2) * 4.0   # pixel size = 4nm

    ax.plot(x_nm, m[:, 1],   "k--", label="Mask",     lw=1)
    ax.plot(x_nm, ap[:, 1],  "r-",  label="Aerial",   lw=1.5)
    ax.plot(x_nm, wp[:, 1],  "b-",  label="Wafer",    lw=1.5)
    ax.axhline(0.3, color="gray", ls=":", label="threshold=0.3")
    ax.set_xlabel("x [nm]")
    ax.set_ylabel("Intensity")
    ax.set_title(f"Profile: {title}")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
if SAVE:
    plt.savefig("litho_profiles.png", dpi=200, bbox_inches="tight")
    print("已保存 litho_profiles.png")
else:
    plt.show()

bs = np.loadtxt("/Users/wyh/Desktop/学校/Litho_cpp/build/parametric_bs.txt")
plt.imshow(bs)
plt.show()