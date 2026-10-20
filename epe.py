from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .common import (
    BatchReductionType,
    ConditionReductionType,
    extract_nominal_condition,
    reduce_condition_values,
    reduce_per_pattern_values,
    validate_condition_stack,
    validate_conditions,
    validate_real_tensor,
)


@dataclass
class EPEMaps:
    """
    单条件边缘放置误差 (Edge Placement Error, EPE) 空间映射容器。
    包含评估点 (Gauges) 处的光强、梯度幅值及最终物理位移量。
    """
    pred_aerial: torch.Tensor
    target_points: torch.Tensor      # [N, M, 2] 评估点坐标
    valid_mask: torch.Tensor         # [N, M] 有效点掩码
    sampled_intensity: torch.Tensor  # [N, M] 采样光强
    sampled_grad_mag: torch.Tensor   # [N, M] 采样梯度幅值 (Image Slope)
    denominator: torch.Tensor        # [N, M] 加上 epsilon 后的分母
    delta_nm: torch.Tensor           # [N, M] 预测边缘与目标边缘的法向物理偏差 (nm)

    @property
    def point_valid_mask(self) -> torch.Tensor:
        return self.valid_mask


@dataclass
class EPEResult:
    """单条件 EPE 评估结果。"""
    per_pattern_value: torch.Tensor  # [N] 单个 Pattern 内所有有效点 EPE 的平方和
    maps: EPEMaps

    @property
    def per_sample_value(self) -> torch.Tensor:
        return self.per_pattern_value


@dataclass
class ProcessWindowEPEMaps:
    """过程窗口 (Process Window) 下的 EPE 空间映射容器。"""
    nominal_aerial: torch.Tensor
    target_points: torch.Tensor
    valid_mask: torch.Tensor
    nominal_delta_nm: torch.Tensor   # [N, M] 名义条件下的 EPE 偏差
    delta_nm_stack: torch.Tensor     # [C, N, M] 跨全窗口条件的 EPE 偏差

    @property
    def point_valid_mask(self) -> torch.Tensor:
        return self.valid_mask


@dataclass
class ProcessWindowEPEResult:
    """
    过程窗口 EPE 统计结果容器。
    用于评估散焦 (Defocus) 与曝光宽容度 (Exposure Latitude) 对图案边缘保真度的联合影响。
    """
    per_condition_per_pattern: torch.Tensor  # [C, N]
    mean_over_conditions: torch.Tensor       # [N]
    max_over_conditions: torch.Tensor        # [N], 常用作鲁棒优化的最坏情况惩罚项
    nominal_value: torch.Tensor              # [N]
    maps: ProcessWindowEPEMaps

    @property
    def per_condition_value(self) -> torch.Tensor:
        return self.per_condition_per_pattern

    @property
    def mean_per_sample_value(self) -> torch.Tensor:
        return self.mean_over_conditions

    @property
    def max_per_sample_value(self) -> torch.Tensor:
        return self.max_over_conditions

    @property
    def nominal_per_sample_value(self) -> torch.Tensor:
        return self.nominal_value


def _prepare_nhw(x: torch.Tensor, name: str) -> tuple[torch.Tensor, bool]:
    """标准化张量为 [N, H, W] 格式，供空间像处理。"""
    validate_real_tensor(x, name=name)

    if x.ndim == 2:
        return x.unsqueeze(0), True
    if x.ndim == 3:
        return x, False

    raise ValueError(f"{name} 必须为 [H, W] 或 [N, H, W]，当前 shape={tuple(x.shape)}")


def _prepare_points(
    target_points: torch.Tensor,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """标准化评估点坐标张量为 [N, M, 2] 格式。"""
    validate_real_tensor(target_points, name="target_points")

    if target_points.ndim == 2:
        if target_points.shape[-1] != 2:
            raise ValueError(f"target_points 若为 [M, 2]，最后一维必须为 2，当前 shape={tuple(target_points.shape)}")
        points = target_points.unsqueeze(0).expand(batch_size, -1, -1)
    elif target_points.ndim == 3:
        if target_points.shape[0] != batch_size or target_points.shape[-1] != 2:
            raise ValueError(f"target_points 若为 [N, M, 2]，必须与 batch size 一致且最后一维为 2")
        points = target_points
    else:
        raise ValueError(f"target_points 必须为 [M, 2] 或 [N, M, 2]，当前 shape={tuple(target_points.shape)}")

    return points.to(device=device, dtype=dtype)


def _prepare_valid_mask(
    valid_mask: torch.Tensor | None,
    batch_size: int,
    num_points: int,
    device: torch.device,
) -> torch.Tensor:
    """标准化掩码为 [N, M] 格式，用于屏蔽无关点 (如 SRAF 区域或不关心的非关键特征)。"""
    if valid_mask is None:
        return torch.ones((batch_size, num_points), device=device, dtype=torch.bool)

    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError("valid_mask 必须为 torch.Tensor 或 None")

    if valid_mask.ndim == 1:
        if valid_mask.shape[0] != num_points:
            raise ValueError(f"valid_mask 长度必须与点数一致")
        out = valid_mask.unsqueeze(0).expand(batch_size, -1)
    elif valid_mask.ndim == 2:
        if valid_mask.shape != (batch_size, num_points):
            raise ValueError(f"valid_mask shape 必须与 batch / 点数一致")
        out = valid_mask
    else:
        raise ValueError(f"valid_mask 必须为 [M]、[N, M] 或 None")

    return out.to(device=device, dtype=torch.bool)


def _build_sobel_kernels(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """构造用于计算空间梯度的 3x3 Sobel 算子。"""
    kx = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device, dtype=dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=device, dtype=dtype,
    ).view(1, 1, 3, 3)
    return kx, ky


def _gradient_magnitude_map(aerial_nhw: torch.Tensor) -> torch.Tensor:
    """
    计算空间像的梯度幅值 (Image Slope)。

    物理意义上，梯度幅值代表了成像边缘的锐利程度。高斜率 (高 NILS) 意味着
    光强波动转化为空间 CD 误差的灵敏度更低，抗工艺干扰能力更强。
    """
    if aerial_nhw.ndim != 3:
        raise ValueError(f"aerial_nhw 必须为 [N, H, W]，当前 shape={tuple(aerial_nhw.shape)}")

    x4 = aerial_nhw.unsqueeze(1)
    kx, ky = _build_sobel_kernels(device=aerial_nhw.device, dtype=aerial_nhw.dtype)
    grad_x = F.conv2d(x4, kx, padding=1)
    grad_y = F.conv2d(x4, ky, padding=1)
    return torch.sqrt(grad_x.square() + grad_y.square()).squeeze(1)


def _points_to_grid(target_points: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """将绝对像素坐标映射至 [-1, 1] 归一化网格坐标，供 F.grid_sample 使用。"""
    if target_points.ndim != 3 or target_points.shape[-1] != 2:
        raise ValueError(f"target_points 必须为 [N, M, 2]")

    x = target_points[..., 0]
    y = target_points[..., 1]

    if width <= 1 or height <= 1:
        raise ValueError(f"height 与 width 必须都大于 1")

    grid_x = 2.0 * x / float(width - 1) - 1.0
    grid_y = 2.0 * y / float(height - 1) - 1.0
    return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(2)


def _sample_map_bilinear(map_nhw: torch.Tensor, target_points: torch.Tensor) -> torch.Tensor:
    """
    通过双线性插值提取指定测点 (Gauges) 的值。
    保证 EPE 评估具备亚像素精度且完全可导。
    """
    if map_nhw.ndim != 3:
        raise ValueError(f"map_nhw 必须为 [N, H, W]")

    n, h, w = map_nhw.shape
    if target_points.shape[0] != n:
        raise ValueError("target_points 的 batch 维必须与 map_nhw 一致")

    grid = _points_to_grid(target_points=target_points, height=h, width=w)
    sampled = F.grid_sample(
        input=map_nhw.unsqueeze(1),
        grid=grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(1).squeeze(-1)


def _compute_delta_nm(
    aerial_nhw: torch.Tensor,
    target_points_n: torch.Tensor,
    valid_mask_n: torch.Tensor,
    threshold: float,
    pixel_size_nm: float,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    EPE 核心物理计算引擎：基于一阶泰勒展开近似边缘偏移。

    Args:
        aerial_nhw: 空间像张量 [N, H, W]。
        target_points_n: 目标轮廓上的离散测点。
        threshold: 显影阈值强度 (Intensity Threshold)。
        pixel_size_nm: 网格像素物理尺寸 (nm)。
        epsilon: 防止除零的数值稳定性常数。

    Returns:
        包含采样光强、采样梯度、分母及最终位移量 delta_nm 的元组。

    Notes:
        - **物理公式**: 假设在测点附近光强呈线性变化，位移量可近似为 
          $ \Delta x = (I_{sampled} - I_{th}) / |\nabla I| \times \text{pixel\_size\_nm} $
        - **数值稳定性**: 由于光强平坦区域梯度趋于零，显式加上 epsilon 可避免梯度爆炸 (NaN)。
    """
    if pixel_size_nm <= 0:
        raise ValueError(f"pixel_size_nm 必须 > 0，当前为 {pixel_size_nm}")
    if epsilon <= 0:
        raise ValueError(f"epsilon 必须 > 0，当前为 {epsilon}")

    grad_mag_map = _gradient_magnitude_map(aerial_nhw)

    sampled_intensity = _sample_map_bilinear(aerial_nhw, target_points_n)
    sampled_grad_mag = _sample_map_bilinear(grad_mag_map, target_points_n)

    threshold_t = torch.as_tensor(threshold, device=aerial_nhw.device, dtype=aerial_nhw.dtype)
    epsilon_t = torch.as_tensor(epsilon, device=aerial_nhw.device, dtype=aerial_nhw.dtype)

    denominator = sampled_grad_mag + epsilon_t
    delta_nm = ((sampled_intensity - threshold_t) / denominator) * float(pixel_size_nm)

    # 掩码过滤，冻结无效点的梯度传递
    sampled_intensity = torch.where(valid_mask_n, sampled_intensity, torch.zeros_like(sampled_intensity))
    sampled_grad_mag = torch.where(valid_mask_n, sampled_grad_mag, torch.zeros_like(sampled_grad_mag))
    denominator = torch.where(valid_mask_n, denominator, torch.zeros_like(denominator))
    delta_nm = torch.where(valid_mask_n, delta_nm, torch.zeros_like(delta_nm))

    return sampled_intensity, sampled_grad_mag, denominator, delta_nm


def epe_result(
    pred_aerial: torch.Tensor,
    target_points: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    threshold: float = 0.25,
    pixel_size_nm: float = 1.0,
    epsilon: float = 1e-7,
) -> EPEResult:
    """计算单条件的 Edge Placement Error 映射结果。"""
    aerial_nhw, _ = _prepare_nhw(pred_aerial, name="pred_aerial")
    batch_size, _, _ = aerial_nhw.shape

    points_n = _prepare_points(target_points=target_points, batch_size=batch_size, device=aerial_nhw.device, dtype=aerial_nhw.dtype)
    valid_n = _prepare_valid_mask(valid_mask=valid_mask, batch_size=batch_size, num_points=points_n.shape[1], device=aerial_nhw.device)

    sampled_intensity, sampled_grad_mag, denominator, delta_nm = _compute_delta_nm(
        aerial_nhw=aerial_nhw,
        target_points_n=points_n,
        valid_mask_n=valid_n,
        threshold=threshold,
        pixel_size_nm=pixel_size_nm,
        epsilon=epsilon,
    )

    per_pattern_value = delta_nm.square().sum(dim=1)

    return EPEResult(
        per_pattern_value=per_pattern_value,
        maps=EPEMaps(
            pred_aerial=aerial_nhw, target_points=points_n, valid_mask=valid_n,
            sampled_intensity=sampled_intensity, sampled_grad_mag=sampled_grad_mag,
            denominator=denominator, delta_nm=delta_nm,
        ),
    )


def epe_loss(
    pred_aerial: torch.Tensor,
    target_points: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    threshold: float = 0.25,
    pixel_size_nm: float = 1.0,
    epsilon: float = 1e-7,
    batch_reduction: BatchReductionType = "mean",
    topk: int | None = None,
) -> torch.Tensor:
    """标准 EPE 损失函数，用于驱动掩模图案轮廓收敛。"""
    result = epe_result(
        pred_aerial=pred_aerial, target_points=target_points, valid_mask=valid_mask,
        threshold=threshold, pixel_size_nm=pixel_size_nm, epsilon=epsilon,
    )
    return reduce_per_pattern_values(
        result.per_pattern_value,
        reduction=batch_reduction,
        topk=topk,
    )


def process_window_epe_result(
    aerial_stack: torch.Tensor,
    conditions: list[dict],
    target_points: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    threshold: float = 0.25,
    pixel_size_nm: float = 1.0,
    epsilon: float = 1e-7,
    nominal_defocus_nm: float = 0.0,
    nominal_dose: float = 1.0,
) -> ProcessWindowEPEResult:
    """计算包含散焦 (Defocus) 堆叠的过程窗口 EPE 分布。"""
    validate_condition_stack(aerial_stack, name="aerial_stack")
    validate_conditions(conditions, aerial_stack.shape[0], stack_name="aerial_stack")

    c, n, _, _ = aerial_stack.shape

    points_n = _prepare_points(target_points=target_points, batch_size=n, device=aerial_stack.device, dtype=aerial_stack.dtype)
    valid_n = _prepare_valid_mask(valid_mask=valid_mask, batch_size=n, num_points=points_n.shape[1], device=aerial_stack.device)

    per_condition_list = []
    delta_nm_list = []

    # 遍历 PW 中所有焦面/剂量组合
    for idx in range(c):
        result_c = epe_result(
            pred_aerial=aerial_stack[idx], target_points=points_n, valid_mask=valid_n,
            threshold=threshold, pixel_size_nm=pixel_size_nm, epsilon=epsilon,
        )
        per_condition_list.append(result_c.per_pattern_value)
        delta_nm_list.append(result_c.maps.delta_nm)

    per_condition_per_pattern = torch.stack(per_condition_list, dim=0)
    delta_nm_stack = torch.stack(delta_nm_list, dim=0)

    mean_over_conditions = per_condition_per_pattern.mean(dim=0)
    max_over_conditions = per_condition_per_pattern.max(dim=0).values

    nominal_aerial = extract_nominal_condition(
        stack=aerial_stack, conditions=conditions,
        nominal_defocus_nm=nominal_defocus_nm, nominal_dose=nominal_dose,
        stack_name="aerial_stack",
    )
    nominal_result = epe_result(
        pred_aerial=nominal_aerial, target_points=points_n, valid_mask=valid_n,
        threshold=threshold, pixel_size_nm=pixel_size_nm, epsilon=epsilon,
    )

    return ProcessWindowEPEResult(
        per_condition_per_pattern=per_condition_per_pattern,
        mean_over_conditions=mean_over_conditions,
        max_over_conditions=max_over_conditions,
        nominal_value=nominal_result.per_pattern_value,
        maps=ProcessWindowEPEMaps(
            nominal_aerial=nominal_aerial, target_points=points_n, valid_mask=valid_n,
            nominal_delta_nm=nominal_result.maps.delta_nm, delta_nm_stack=delta_nm_stack,
        ),
    )


def process_window_epe_loss(
    aerial_stack: torch.Tensor,
    conditions: list[dict],
    target_points: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    threshold: float = 0.25,
    pixel_size_nm: float = 1.0,
    epsilon: float = 1e-7,
    nominal_defocus_nm: float = 0.0,
    nominal_dose: float = 1.0,
    condition_reduction: ConditionReductionType = "mean",
    batch_reduction: BatchReductionType = "mean",
    topk: int | None = None,
) -> torch.Tensor:
    """
    面向鲁棒性掩模优化 (Robust SMO) 的过程窗口 EPE 损失函数。
    
    Notes:
        - 配合 condition_reduction="max" 可强制网络优先收敛 PW 中表现最差的边角条件，
          这是提高重叠过程窗口 (Overlapping Process Window) 的标准工程手段。
    """
    result = process_window_epe_result(
        aerial_stack=aerial_stack, conditions=conditions, target_points=target_points,
        valid_mask=valid_mask, threshold=threshold, pixel_size_nm=pixel_size_nm,
        epsilon=epsilon, nominal_defocus_nm=nominal_defocus_nm, nominal_dose=nominal_dose,
    )

    per_pattern_value = reduce_condition_values(
        result.per_condition_per_pattern,
        reduction=condition_reduction,
    )

    if per_pattern_value.ndim != 1:
        raise ValueError(
            "condition_reduction='none' 时 process_window_epe_loss 不返回标量。"
            " 若要保留 [C, N]，请直接调用 process_window_epe_result()."
        )

    return reduce_per_pattern_values(
        per_pattern_value,
        reduction=batch_reduction,
        topk=topk,
    )


__all__ = [
    "EPEMaps",
    "EPEResult",
    "ProcessWindowEPEMaps",
    "ProcessWindowEPEResult",
    "epe_result",
    "epe_loss",
    "process_window_epe_result",
    "process_window_epe_loss",
]