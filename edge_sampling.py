from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch

from .common import validate_real_tensor


EdgeType = Literal["vertical", "horizontal"]


@dataclass
class EdgeSampleBatch:
    """
    基于目标版图生成的边界采样信息。

    统一约定：
    - points                : [N, M, 2], 连续像素坐标 (x, y)
    - normals               : [N, M, 2], 单位法向
    - valid_mask            : [N, M], bool
    - segment_ids           : [N, M], long，padding 位置为 -1
    - edge_types            : [N, M], long，0=vertical, 1=horizontal, padding 为 -1
    - num_points_per_pattern: [N]
    """

    points: torch.Tensor
    normals: torch.Tensor
    valid_mask: torch.Tensor
    segment_ids: torch.Tensor
    edge_types: torch.Tensor
    num_points_per_pattern: torch.Tensor

    @property
    def num_points_per_sample(self) -> torch.Tensor:
        return self.num_points_per_pattern


TargetEdgeSamples = EdgeSampleBatch


@dataclass
class _BoundarySegment:
    axis: EdgeType
    fixed_coord: float
    start_coord: float
    end_coord: float
    normal_x: float
    normal_y: float
    segment_id: int


def _prepare_target_nhw(
    target: torch.Tensor,
    threshold: float,
) -> tuple[torch.Tensor, bool]:
    validate_real_tensor(target, name="target")

    if target.ndim == 2:
        target_nhw = target.unsqueeze(0)
        squeeze_input = True
    elif target.ndim == 3:
        target_nhw = target
        squeeze_input = False
    else:
        raise ValueError(
            f"target 必须为 [H, W] 或 [N, H, W]，当前 shape={tuple(target.shape)}"
        )

    target_bin = target_nhw >= float(threshold)
    return target_bin, squeeze_input


def _extract_boundary_segments_single(
    target_hw: torch.Tensor,
    include_image_boundary: bool = True,
) -> list[_BoundarySegment]:
    if target_hw.ndim != 2:
        raise ValueError(f"target_hw 必须为 [H, W]，当前 shape={tuple(target_hw.shape)}")
    if target_hw.dtype != torch.bool:
        raise TypeError("target_hw 必须为 bool 张量")

    h, w = target_hw.shape
    segments: list[_BoundarySegment] = []
    segment_id = 0

    for j in range(w + 1):
        if j == 0:
            if not include_image_boundary:
                continue
            left_inside = torch.zeros(h, dtype=torch.bool)
            right_inside = target_hw[:, 0]
        elif j == w:
            if not include_image_boundary:
                continue
            left_inside = target_hw[:, w - 1]
            right_inside = torch.zeros(h, dtype=torch.bool)
        else:
            left_inside = target_hw[:, j - 1]
            right_inside = target_hw[:, j]

        sign = torch.zeros(h, dtype=torch.int8)
        sign[left_inside & (~right_inside)] = 1
        sign[(~left_inside) & right_inside] = -1

        i = 0
        while i < h:
            s = int(sign[i].item())
            if s == 0:
                i += 1
                continue

            start_i = i
            while i + 1 < h and int(sign[i + 1].item()) == s:
                i += 1
            end_i = i

            segments.append(
                _BoundarySegment(
                    axis="vertical",
                    fixed_coord=float(j) - 0.5,
                    start_coord=float(start_i) - 0.5,
                    end_coord=float(end_i) + 0.5,
                    normal_x=float(s),
                    normal_y=0.0,
                    segment_id=segment_id,
                )
            )
            segment_id += 1
            i += 1

    for i in range(h + 1):
        if i == 0:
            if not include_image_boundary:
                continue
            top_inside = torch.zeros(w, dtype=torch.bool)
            bottom_inside = target_hw[0, :]
        elif i == h:
            if not include_image_boundary:
                continue
            top_inside = target_hw[h - 1, :]
            bottom_inside = torch.zeros(w, dtype=torch.bool)
        else:
            top_inside = target_hw[i - 1, :]
            bottom_inside = target_hw[i, :]

        sign = torch.zeros(w, dtype=torch.int8)
        sign[top_inside & (~bottom_inside)] = 1
        sign[(~top_inside) & bottom_inside] = -1

        j = 0
        while j < w:
            s = int(sign[j].item())
            if s == 0:
                j += 1
                continue

            start_j = j
            while j + 1 < w and int(sign[j + 1].item()) == s:
                j += 1
            end_j = j

            segments.append(
                _BoundarySegment(
                    axis="horizontal",
                    fixed_coord=float(i) - 0.5,
                    start_coord=float(start_j) - 0.5,
                    end_coord=float(end_j) + 0.5,
                    normal_x=0.0,
                    normal_y=float(s),
                    segment_id=segment_id,
                )
            )
            segment_id += 1
            j += 1

    return segments


def _sample_one_segment(
    segment: _BoundarySegment,
    sample_step_px: float,
    corner_margin_px: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if sample_step_px <= 0:
        raise ValueError(f"sample_step_px 必须 > 0，当前为 {sample_step_px}")
    if corner_margin_px < 0:
        raise ValueError(f"corner_margin_px 必须 >= 0，当前为 {corner_margin_px}")

    a = float(segment.start_coord) + float(corner_margin_px)
    b = float(segment.end_coord) - float(corner_margin_px)

    if b < a:
        empty_2 = torch.empty((0, 2), dtype=dtype)
        empty_1 = torch.empty((0,), dtype=torch.long)
        return empty_2, empty_2.clone(), empty_1, empty_1.clone()

    length = b - a
    if length <= 1e-12:
        coords = torch.tensor([(a + b) * 0.5], dtype=dtype)
    else:
        num = max(int(math.floor(length / float(sample_step_px) + 1e-12)) + 1, 1)
        if num == 1:
            coords = torch.tensor([(a + b) * 0.5], dtype=dtype)
        else:
            coords = torch.linspace(a, b, steps=num, dtype=dtype)

    k = int(coords.numel())
    normals = torch.empty((k, 2), dtype=dtype)
    normals[:, 0] = float(segment.normal_x)
    normals[:, 1] = float(segment.normal_y)

    if segment.axis == "vertical":
        points = torch.empty((k, 2), dtype=dtype)
        points[:, 0] = float(segment.fixed_coord)
        points[:, 1] = coords
        edge_type_id = 0
    elif segment.axis == "horizontal":
        points = torch.empty((k, 2), dtype=dtype)
        points[:, 0] = coords
        points[:, 1] = float(segment.fixed_coord)
        edge_type_id = 1
    else:
        raise ValueError(f"未知 segment.axis: {segment.axis}")

    segment_ids = torch.full((k,), int(segment.segment_id), dtype=torch.long)
    edge_types = torch.full((k,), int(edge_type_id), dtype=torch.long)
    return points, normals, segment_ids, edge_types


def _sample_segments_single(
    segments: list[_BoundarySegment],
    sample_step_px: float,
    corner_margin_px: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pts_list = []
    nrm_list = []
    seg_id_list = []
    edge_type_list = []

    for seg in segments:
        pts, nrms, seg_ids, edge_types = _sample_one_segment(
            segment=seg,
            sample_step_px=sample_step_px,
            corner_margin_px=corner_margin_px,
            dtype=dtype,
        )
        if pts.numel() == 0:
            continue
        pts_list.append(pts)
        nrm_list.append(nrms)
        seg_id_list.append(seg_ids)
        edge_type_list.append(edge_types)

    if len(pts_list) == 0:
        raise ValueError(
            "当前 target 未生成任何边界采样点。"
            " 请检查 target 是否为空、corner_margin_px 是否过大，"
            "或 sample_step_px 是否设置异常。"
        )

    return (
        torch.cat(pts_list, dim=0),
        torch.cat(nrm_list, dim=0),
        torch.cat(seg_id_list, dim=0),
        torch.cat(edge_type_list, dim=0),
    )


def sample_target_edge_points(
    target: torch.Tensor,
    threshold: float = 0.5,
    sample_step_px: float = 1.0,
    corner_margin_px: float = 0.5,
    include_image_boundary: bool = True,
) -> EdgeSampleBatch:
    """
    基于目标版图生成边界采样点。

    支持输入：
    - target: [H, W]
    - target: [N, H, W]
    """
    target_bin, _ = _prepare_target_nhw(target=target, threshold=threshold)
    device = target.device

    out_dtype = target.dtype if target.dtype in (torch.float64, torch.float32) else torch.float32
    target_bin_cpu = target_bin.detach().to(device="cpu", dtype=torch.bool)

    per_sample_points = []
    per_sample_normals = []
    per_sample_seg_ids = []
    per_sample_edge_types = []
    counts = []

    n, _, _ = target_bin_cpu.shape
    for idx in range(n):
        segments = _extract_boundary_segments_single(
            target_hw=target_bin_cpu[idx],
            include_image_boundary=include_image_boundary,
        )

        pts, nrms, seg_ids, edge_types = _sample_segments_single(
            segments=segments,
            sample_step_px=sample_step_px,
            corner_margin_px=corner_margin_px,
            dtype=out_dtype,
        )

        per_sample_points.append(pts)
        per_sample_normals.append(nrms)
        per_sample_seg_ids.append(seg_ids)
        per_sample_edge_types.append(edge_types)
        counts.append(int(pts.shape[0]))

    max_points = max(counts)
    if max_points <= 0:
        raise ValueError("batch 内没有任何有效边界采样点。")

    points = torch.zeros((n, max_points, 2), device=device, dtype=out_dtype)
    normals = torch.zeros((n, max_points, 2), device=device, dtype=out_dtype)
    valid_mask = torch.zeros((n, max_points), device=device, dtype=torch.bool)
    segment_ids = torch.full((n, max_points), -1, device=device, dtype=torch.long)
    edge_types = torch.full((n, max_points), -1, device=device, dtype=torch.long)

    for idx in range(n):
        k = counts[idx]
        points[idx, :k] = per_sample_points[idx].to(device=device, dtype=out_dtype)
        normals[idx, :k] = per_sample_normals[idx].to(device=device, dtype=out_dtype)
        valid_mask[idx, :k] = True
        segment_ids[idx, :k] = per_sample_seg_ids[idx].to(device=device, dtype=torch.long)
        edge_types[idx, :k] = per_sample_edge_types[idx].to(device=device, dtype=torch.long)

    return EdgeSampleBatch(
        points=points,
        normals=normals,
        valid_mask=valid_mask,
        segment_ids=segment_ids,
        edge_types=edge_types,
        num_points_per_pattern=torch.tensor(counts, device=device, dtype=torch.long),
    )


def sample_target_epe_points(
    target: torch.Tensor,
    threshold: float = 0.5,
    sample_step_px: float = 1.0,
    corner_margin_px: float = 0.5,
    include_image_boundary: bool = True,
) -> EdgeSampleBatch:
    """
    兼容旧接口名。建议新代码统一改用 sample_target_edge_points。
    """
    return sample_target_edge_points(
        target=target,
        threshold=threshold,
        sample_step_px=sample_step_px,
        corner_margin_px=corner_margin_px,
        include_image_boundary=include_image_boundary,
    )
