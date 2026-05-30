"""Metric utilities for CAIP."""

from __future__ import annotations

from typing import Tuple

import torch


def expected_calibration_error(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 15,
) -> float:
    """Computes Expected Calibration Error (ECE) for classification outputs.

    Args:
        probabilities: Tensor of shape ``[N, C]`` containing class probabilities.
        labels: Tensor of shape ``[N]`` containing ground-truth class indices.
        num_bins: Number of confidence bins used to estimate calibration error.

    Returns:
        A scalar float ECE value in ``[0, 1]``.
    """
    # ECE 用于衡量“模型自信程度（置信度）”与“真实正确率”是否匹配：
    # - 理想情况下：当模型给出 0.8 的置信度时，它在这类样本上的准确率也应接近 0.8
    # - 实现方式：把置信度区间分成 num_bins 个桶，在每个桶里计算平均置信度与平均准确率的差，再按桶内样本占比加权求和

    if probabilities.ndim != 2:
        raise ValueError(f"`probabilities` must have shape [N, C], got {tuple(probabilities.shape)}")
    if labels.ndim != 1:
        raise ValueError(f"`labels` must have shape [N], got {tuple(labels.shape)}")
    if probabilities.size(0) != labels.size(0):
        raise ValueError("`probabilities` and `labels` must contain the same number of samples.")
    if num_bins <= 0:
        raise ValueError("`num_bins` must be a positive integer.")

    if probabilities.numel() == 0:
        return 0.0

    # 预测类别与对应置信度（最大类别概率）
    confidences, predictions = probabilities.max(dim=1)
    # correctness: 1 表示预测正确，0 表示预测错误
    correctness = predictions.eq(labels).float()
    # [0, 1] 等分成 num_bins 个区间，bin_boundaries 长度为 num_bins + 1
    bin_boundaries = torch.linspace(0.0, 1.0, steps=num_bins + 1, device=probabilities.device)

    ece = torch.zeros(1, device=probabilities.device)
    total_count = float(probabilities.size(0))

    # Each bin estimates the gap between confidence and empirical accuracy.
    # 每个置信度分桶衡量“模型自信程度”和“真实准确率”之间的偏差。
    for bin_idx in range(num_bins):
        left = bin_boundaries[bin_idx]
        right = bin_boundaries[bin_idx + 1]
        # 最后一个桶右边界闭区间，避免把置信度=1.0 的样本漏掉
        if bin_idx == num_bins - 1:
            in_bin = (confidences >= left) & (confidences <= right)
        else:
            in_bin = (confidences >= left) & (confidences < right)

        if not in_bin.any():
            continue

        # 桶内平均置信度与平均准确率
        bin_confidence = confidences[in_bin].mean()
        bin_accuracy = correctness[in_bin].mean()
        # 桶权重：桶内样本数 / 总样本数
        bin_weight = in_bin.float().sum() / total_count
        ece += torch.abs(bin_accuracy - bin_confidence) * bin_weight

    return float(ece.item())


def calibration_summary(
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 15,
) -> Tuple[float, int]:
    """Returns ECE together with the number of bins used."""
    # 统一返回 (ece, num_bins)，便于训练/评估阶段记录配置
    # 或在日志中展示“计算 ECE 时使用了多少个分桶”。

    return expected_calibration_error(probabilities, labels, num_bins=num_bins), num_bins
