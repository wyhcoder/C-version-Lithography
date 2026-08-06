"""plot_msaa_cp.py
可视化 demo_msaa_cp 的输出：gray / binary 并排显示，叠加多边形边界。

用法：
    python3 scripts/plot_msaa_cp.py
        --gray   msaa_cp_gray.txt
        --binary msaa_cp_binary.txt
        --poly   msaa_cp_polygon.txt
        --save   msaa_cp_preview.png
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.collections import PatchCollection


def load_matrix(path: str) -> np.ndarray:
    return np.loadtxt(path, comments="#")


def load_polygons(path: str):
    """读 demo_msaa_cp 输出的多边形文件，返回 list[np.ndarray(N,2)]，每行 (y, x)。"""
    polys = []
    cur = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                if cur:
                    polys.append(np.array(cur))
                    cur = []
                continue
            parts = line.split()
            if len(parts) >= 2:
                cur.append([float(parts[0]), float(parts[1])])
    if cur:
        polys.append(np.array(cur))
    return polys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gray",   default="msaa_cp_gray.txt")
    ap.add_argument("--binary", default="msaa_cp_binary.txt")
    ap.add_argument("--poly",   default="msaa_cp_polygon.txt")
    ap.add_argument("--save",   default="msaa_cp_preview.png")
    ap.add_argument("--dpi",    type=int, default=200)
    args = ap.parse_args()

    gray = load_matrix(args.gray)
    binary = load_matrix(args.binary)
    polys = load_polygons(args.poly)
    print(f"[plot] gray   shape={gray.shape}   max={gray.max():.4f}")
    print(f"[plot] binary shape={binary.shape} sum={binary.sum():.0f}")
    print(f"[plot] polys  : {len(polys)} contour(s)")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, mat, title, cmap in [
        (axes[0], gray,   "MSAA gray",   "hot"),
        (axes[1], binary, "MSAA binary", "gray"),
    ]:
        ax.imshow(mat, cmap=cmap, origin="upper",
                  extent=[0, mat.shape[1], mat.shape[0], 0])
        # 叠加多边形边界：(y, x) → 画图用 (x, y)
        for poly in polys:
            pts = poly[:, [1, 0]]  # swap to (x, y)
            patch = MplPolygon(pts, closed=True, fill=False,
                               edgecolor="cyan", linewidth=1.0)
            ax.add_patch(patch)
        ax.set_title(title)
        ax.set_axis_off()

    fig.tight_layout()
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.save, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] saved: {args.save}")


if __name__ == "__main__":
    main()
