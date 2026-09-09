#!/usr/bin/env python3
"""复现 JOLT-D-26-04401 论文第 2.2 节的边界积分光栅化。

算法对应论文公式 (7)-(8)：

1. 将闭合轮廓分割成长度约为 h/4 的小线段；
2. 在每段中点计算外法向量；
3. 用紧支撑三角核近似二维 Dirac delta，将边界梯度散布到网格；
4. 分别沿 x、y 方向积分梯度，恢复指示函数；
5. 将两个方向的结果平均并裁剪到 [0, 1]。

默认示例是由四段三次 Bezier 曲线构成的椭圆。也可以通过 --contour
传入两列为 x、y 的文本文件。轮廓既可以顺时针，也可以逆时针；程序会根据
有向面积自动确定外法向。

示例：
    ./.venv/bin/python scripts/reproduce_dirac_rasterization.py

    ./.venv/bin/python scripts/reproduce_dirac_rasterization.py \
        --contour path/to/contour_xy.txt \
        --width 256 --height 256 --output outputs/paper_rasterization
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path as MplPath


@dataclass(frozen=True)
class BoundarySamples:
    """离散边界段的中点、外法向量和长度。"""

    points: np.ndarray
    normals: np.ndarray
    lengths: np.ndarray


def cubic_bezier(
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    samples: int,
) -> np.ndarray:
    """采样一段三次 Bezier 曲线，不包含终点以避免相邻曲线重复。"""

    t = np.linspace(0.0, 1.0, samples, endpoint=False)[:, None]
    omt = 1.0 - t
    return (
        omt**3 * p0
        + 3.0 * omt**2 * t * p1
        + 3.0 * omt * t**2 * p2
        + t**3 * p3
    )


def make_demo_bezier_contour(width: int, height: int) -> np.ndarray:
    """生成由四段三次 Bezier 曲线近似的逆时针椭圆。"""

    cx = 0.50 * (width - 1)
    cy = 0.50 * (height - 1)
    rx = 0.31 * (width - 1)
    ry = 0.27 * (height - 1)
    kappa = 4.0 * (np.sqrt(2.0) - 1.0) / 3.0

    right = np.array([cx + rx, cy])
    top = np.array([cx, cy + ry])
    left = np.array([cx - rx, cy])
    bottom = np.array([cx, cy - ry])

    segments = [
        (right, right + [0.0, kappa * ry], top + [kappa * rx, 0.0], top),
        (top, top + [-kappa * rx, 0.0], left + [0.0, kappa * ry], left),
        (left, left + [0.0, -kappa * ry], bottom + [-kappa * rx, 0.0], bottom),
        (bottom, bottom + [kappa * rx, 0.0], right + [0.0, -kappa * ry], right),
    ]

    # 这里只负责把解析 Bezier 曲线变成足够密的折线；论文规定的 h/4
    # 边界分段在 sample_boundary_segments() 中再次执行。
    contour = np.vstack(
        [cubic_bezier(*segment, samples=64) for segment in segments]
    )
    return np.vstack([contour, contour[0]])


def close_contour(contour: np.ndarray) -> np.ndarray:
    """检查输入并保证轮廓首尾闭合。"""

    contour = np.asarray(contour, dtype=np.float64)
    if contour.ndim != 2 or contour.shape[1] != 2 or contour.shape[0] < 3:
        raise ValueError("contour 必须是形状为 (N, 2) 且 N >= 3 的数组")
    if not np.all(np.isfinite(contour)):
        raise ValueError("contour 中存在 NaN 或无穷大")
    if not np.allclose(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])
    return contour


def signed_area(contour: np.ndarray) -> float:
    """计算闭合轮廓的有向面积；正值表示逆时针。"""

    x0, y0 = contour[:-1].T
    x1, y1 = contour[1:].T
    return 0.5 * float(np.sum(x0 * y1 - x1 * y0))


def sample_boundary_segments(
    contour: np.ndarray,
    target_length: float,
) -> BoundarySamples:
    """按目标长度细分轮廓，并在每段中点计算外法向。

    参数：
        contour: 首尾闭合的折线点，形状为 (N, 2)。
        target_length: 论文中取 h/4。

    返回：
        每个边界小段的中点、外法向量和实际段长。
    """

    if target_length <= 0.0:
        raise ValueError("target_length 必须大于 0")

    contour = close_contour(contour)
    area = signed_area(contour)
    if abs(area) < 1e-12:
        raise ValueError("轮廓有向面积接近 0，可能退化或没有正确闭合")
    ccw = area > 0.0

    all_points: list[np.ndarray] = []
    all_normals: list[np.ndarray] = []
    all_lengths: list[np.ndarray] = []

    for p0, p1 in zip(contour[:-1], contour[1:]):
        delta = p1 - p0
        edge_length = float(np.linalg.norm(delta))
        if edge_length <= 1e-14:
            continue

        count = max(1, int(np.ceil(edge_length / target_length)))
        fraction = (np.arange(count, dtype=np.float64) + 0.5) / count
        midpoints = p0[None, :] + fraction[:, None] * delta[None, :]
        tangent = delta / edge_length

        # 逆时针轮廓的内部位于切线左侧，因此外法向是 (ty, -tx)。
        if ccw:
            outward = np.array([tangent[1], -tangent[0]])
        else:
            outward = np.array([-tangent[1], tangent[0]])

        all_points.append(midpoints)
        all_normals.append(np.repeat(outward[None, :], count, axis=0))
        all_lengths.append(np.full(count, edge_length / count))

    if not all_points:
        raise ValueError("轮廓中没有有效线段")

    return BoundarySamples(
        points=np.vstack(all_points),
        normals=np.vstack(all_normals),
        lengths=np.concatenate(all_lengths),
    )


def tent(r: float) -> float:
    """论文公式 (8) 的一维近似 Dirac 核。"""

    absolute = abs(r)
    return 1.0 - absolute if absolute <= 1.0 else 0.0


def scatter_boundary_gradient(
    samples: BoundarySamples,
    width: int,
    height: int,
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """按照论文公式 (7) 将边界梯度散布到二维网格。"""

    gx = np.zeros((height, width), dtype=np.float64)
    gy = np.zeros_like(gx)

    for point, normal, segment_length in zip(
        samples.points, samples.normals, samples.lengths
    ):
        ux = point[0] / h
        uy = point[1] / h
        ix0 = int(np.floor(ux))
        iy0 = int(np.floor(uy))
        scale = -segment_length / (h * h)

        # 三角核的支撑范围为 [-1, 1]，所以一个边界点最多影响四个网格点。
        for j in (iy0, iy0 + 1):
            if not 0 <= j < height:
                continue
            wy = tent(uy - j)
            if wy == 0.0:
                continue
            for i in (ix0, ix0 + 1):
                if not 0 <= i < width:
                    continue
                weight = tent(ux - i) * wy
                gx[j, i] += scale * normal[0] * weight
                gy[j, i] += scale * normal[1] * weight

    return gx, gy


def reconstruct_indicator(
    gx: np.ndarray,
    gy: np.ndarray,
    h: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """沿 x/y 积分梯度，平均并裁剪到 [0, 1]。"""

    # 计算域左边界和下边界位于掩模之外，因此指示函数初值为 0。
    chi_x_raw = h * np.cumsum(gx, axis=1)
    chi_y_raw = h * np.cumsum(gy, axis=0)
    chi_raw = 0.5 * (chi_x_raw + chi_y_raw)
    chi = np.clip(chi_raw, 0.0, 1.0)
    return chi_x_raw, chi_y_raw, chi_raw, chi


def supersampled_reference(
    contour: np.ndarray,
    width: int,
    height: int,
    h: float,
    samples_per_axis: int,
) -> np.ndarray:
    """用独立的规则网格超采样生成像素覆盖率参考结果。"""

    contour = close_contour(contour)
    if samples_per_axis <= 0:
        raise ValueError("samples_per_axis 必须大于 0")

    offsets = (
        (np.arange(samples_per_axis, dtype=np.float64) + 0.5)
        / samples_per_axis
        - 0.5
    )
    x_subpixels = (
        np.arange(width, dtype=np.float64)[:, None] + offsets[None, :]
    ).reshape(-1) * h
    path = MplPath(contour)
    coverage = np.empty((height, width), dtype=np.float64)

    # 逐个输出行计算，避免在较大网格上一次分配全部超采样点。
    for j in range(height):
        y_subpixels = (j + offsets) * h
        xx, yy = np.meshgrid(x_subpixels, y_subpixels)
        points = np.column_stack([xx.ravel(), yy.ravel()])
        inside = path.contains_points(points).reshape(
            samples_per_axis, width, samples_per_axis
        )
        coverage[j] = np.mean(inside, axis=(0, 2))

    return coverage


def compute_metrics(chi: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """计算连续误差以及 0.5 阈值后的二值一致性。"""

    prediction = chi >= 0.5
    target = reference >= 0.5
    intersection = int(np.count_nonzero(prediction & target))
    union = int(np.count_nonzero(prediction | target))
    return {
        "mae": float(np.mean(np.abs(chi - reference))),
        "rmse": float(np.sqrt(np.mean((chi - reference) ** 2))),
        "iou_at_0.5": intersection / union if union else 1.0,
        "pixel_accuracy_at_0.5": float(np.mean(prediction == target)),
    }


def save_figure(
    output_path: Path,
    contour: np.ndarray,
    samples: BoundarySamples,
    gx: np.ndarray,
    gy: np.ndarray,
    chi_x: np.ndarray,
    chi_y: np.ndarray,
    chi: np.ndarray,
    reference: np.ndarray,
    h: float,
) -> None:
    """保存算法中间量和参考结果的可视化。"""

    height, width = chi.shape
    extent = (0.0, (width - 1) * h, 0.0, (height - 1) * h)
    difference = chi - reference

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(contour[:, 0], contour[:, 1], color="tab:blue", linewidth=2)
    stride = max(1, samples.points.shape[0] // 36)
    selected = slice(None, None, stride)
    ax.quiver(
        samples.points[selected, 0],
        samples.points[selected, 1],
        samples.normals[selected, 0],
        samples.normals[selected, 1],
        color="tab:red",
        angles="xy",
        scale_units="xy",
        scale=0.16,
        width=0.004,
    )
    ax.set_title("Bezier contour and outward normals")
    ax.set_aspect("equal")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    panels = [
        (axes[0, 1], gx, r"$\widetilde{G}_x$ from Eq. (7)", "coolwarm", None),
        (axes[0, 2], gy, r"$\widetilde{G}_y$ from Eq. (7)", "coolwarm", None),
        (axes[0, 3], chi, r"final $\widetilde{\chi}$", "gray", (0.0, 1.0)),
        (axes[1, 0], chi_x, r"integral of $\widetilde{G}_x$", "gray", (0.0, 1.0)),
        (axes[1, 1], chi_y, r"integral of $\widetilde{G}_y$", "gray", (0.0, 1.0)),
        (axes[1, 2], reference, "supersampling reference", "gray", (0.0, 1.0)),
    ]
    for panel_ax, image, title, cmap, limits in panels:
        kwargs = {}
        if limits is not None:
            kwargs.update(vmin=limits[0], vmax=limits[1])
        handle = panel_ax.imshow(
            image,
            origin="lower",
            extent=extent,
            cmap=cmap,
            interpolation="nearest",
            **kwargs,
        )
        panel_ax.set_title(title)
        fig.colorbar(handle, ax=panel_ax, fraction=0.046, pad=0.04)

    limit = max(float(np.max(np.abs(difference))), 1e-12)
    handle = axes[1, 3].imshow(
        difference,
        origin="lower",
        extent=extent,
        cmap="coolwarm",
        interpolation="nearest",
        vmin=-limit,
        vmax=limit,
    )
    axes[1, 3].set_title(r"$\widetilde{\chi}$ - reference")
    fig.colorbar(handle, ax=axes[1, 3], fraction=0.046, pad=0.04)

    for panel_ax in axes.flat:
        panel_ax.set_xlabel("x")
        panel_ax.set_ylabel("y")

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="复现论文公式 (7)-(8) 的 Dirac 边界积分光栅化"
    )
    parser.add_argument("--contour", type=Path, help="可选的两列 x/y 闭合轮廓文本")
    parser.add_argument("--width", type=int, default=128, help="网格列数")
    parser.add_argument("--height", type=int, default=128, help="网格行数")
    parser.add_argument("--h", type=float, default=1.0, help="网格间距")
    parser.add_argument(
        "--segment-fraction",
        type=float,
        default=0.25,
        help="边界段长度与 h 的比例；论文取 0.25",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/paper_rasterization"),
        help="输出目录",
    )
    parser.add_argument(
        "--reference-samples",
        type=int,
        default=8,
        help="独立参考光栅化每个方向的子像素数，默认 8 即 8x8",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.width < 3 or args.height < 3:
        raise ValueError("width 和 height 必须至少为 3")
    if args.h <= 0.0:
        raise ValueError("h 必须大于 0")
    if args.segment_fraction <= 0.0:
        raise ValueError("segment-fraction 必须大于 0")
    if args.reference_samples <= 0:
        raise ValueError("reference-samples 必须大于 0")

    if args.contour is None:
        contour = make_demo_bezier_contour(args.width, args.height) * args.h
        contour_source = "built-in four-segment cubic Bezier ellipse"
    else:
        contour = close_contour(np.loadtxt(args.contour, dtype=np.float64))
        contour_source = str(args.contour)

    x_limit = (args.width - 1) * args.h
    y_limit = (args.height - 1) * args.h
    if (
        np.min(contour[:, 0]) <= 0.0
        or np.max(contour[:, 0]) >= x_limit
        or np.min(contour[:, 1]) <= 0.0
        or np.max(contour[:, 1]) >= y_limit
    ):
        raise ValueError("轮廓必须完整位于计算域内部，不能接触计算域边界")

    samples = sample_boundary_segments(
        contour,
        target_length=args.segment_fraction * args.h,
    )
    gx, gy = scatter_boundary_gradient(
        samples,
        width=args.width,
        height=args.height,
        h=args.h,
    )
    chi_x, chi_y, chi_raw, chi = reconstruct_indicator(gx, gy, args.h)
    reference = supersampled_reference(
        contour,
        width=args.width,
        height=args.height,
        h=args.h,
        samples_per_axis=args.reference_samples,
    )
    metrics = compute_metrics(chi, reference)

    args.output.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output / "mask_dirac.txt", chi, fmt="%.9f")
    np.savetxt(args.output / "mask_reference.txt", reference, fmt="%.9f")
    np.savez_compressed(
        args.output / "rasterization_data.npz",
        contour=contour,
        boundary_points=samples.points,
        boundary_normals=samples.normals,
        boundary_segment_lengths=samples.lengths,
        gx=gx,
        gy=gy,
        chi_x=chi_x,
        chi_y=chi_y,
        chi_raw=chi_raw,
        chi=chi,
        reference=reference,
    )
    save_figure(
        args.output / "paper_rasterization.png",
        contour,
        samples,
        gx,
        gy,
        chi_x,
        chi_y,
        chi,
        reference,
        args.h,
    )

    contour_area = abs(signed_area(close_contour(contour)))
    reconstructed_area = float(np.sum(chi) * args.h * args.h)
    print(f"contour source            : {contour_source}")
    print(f"grid                      : {args.width} x {args.height}, h={args.h:g}")
    print(f"boundary samples          : {samples.points.shape[0]}")
    print(f"mean boundary segment     : {np.mean(samples.lengths):.6f}")
    print(f"requested segment length  : {args.segment_fraction * args.h:.6f}")
    print(f"reference supersampling   : {args.reference_samples} x {args.reference_samples}")
    print(f"raw chi range             : [{chi_raw.min():.6f}, {chi_raw.max():.6f}]")
    print(f"contour area              : {contour_area:.6f}")
    print(f"reconstructed grid area   : {reconstructed_area:.6f}")
    print(f"MAE                       : {metrics['mae']:.8f}")
    print(f"RMSE                      : {metrics['rmse']:.8f}")
    print(f"IoU @ 0.5                 : {metrics['iou_at_0.5']:.8f}")
    print(f"pixel accuracy @ 0.5      : {metrics['pixel_accuracy_at_0.5']:.8f}")
    print(f"outputs                   : {args.output.resolve()}")


if __name__ == "__main__":
    main()
