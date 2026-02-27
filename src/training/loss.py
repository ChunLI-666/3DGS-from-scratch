"""
Loss Functions for 3D Gaussian Splatting Training

This module provides differentiable loss functions for optimizing
3D Gaussian parameters from image supervision.

Three loss functions are provided:
1. L1 loss — pixel-wise absolute difference
2. SSIM loss — structural similarity index measure
3. Combined loss — weighted combination of L1 and SSIM

Reference:
    3D Gaussian Splatting for Real-Time Radiance Field Rendering
    (Kerbl et al., 2023) — Section 5, Equation 7:
        L = (1 - λ) * L_1 + λ * L_SSIM
"""

import torch
import torch.nn.functional as F
from typing import Optional


def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Compute L1 (mean absolute error) loss between predicted and target images.

    L1 loss measures the average pixel-wise absolute difference:
        L_1 = (1/N) * Σ |pred_i - target_i|

    This is a simple but effective loss for image reconstruction.
    Compared to L2 (MSE), L1 is more robust to outliers and produces
    sharper results because it does not over-penalize large errors.

    Args:
        pred: Predicted image, any shape (e.g., [H, W, 3] or [B, C, H, W])
        target: Ground truth image, same shape as pred

    Returns:
        Scalar loss value (mean of absolute differences)

    Example:
        >>> pred = torch.rand(64, 64, 3)
        >>> target = torch.rand(64, 64, 3)
        >>> loss = l1_loss(pred, target)
    """
    return torch.abs(pred - target).mean()


def ssim_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
) -> torch.Tensor:
    """
    Compute SSIM (Structural Similarity Index Measure) loss.

    SSIM measures perceptual similarity between two images by comparing
    local patterns of luminance, contrast, and structure:

        SSIM(x, y) = (2*μ_x*μ_y + C1) * (2*σ_xy + C2)
                     / (μ_x² + μ_y² + C1) * (σ_x² + σ_y² + C2)

    where:
        - μ_x, μ_y are local means (computed via Gaussian window)
        - σ_x², σ_y² are local variances
        - σ_xy is local covariance
        - C1 = (0.01 * L)², C2 = (0.03 * L)² are stability constants
        - L is the dynamic range (1.0 for normalized images)

    The loss is: L_SSIM = 1 - SSIM (so 0 = perfect match)

    Args:
        pred: Predicted image [H, W, C] or [B, C, H, W]
        target: Ground truth image, same shape as pred
        window_size: Size of the Gaussian smoothing window (default: 11)

    Returns:
        Scalar SSIM loss value in [0, 1] range (lower is better)

    Example:
        >>> pred = torch.rand(64, 64, 3)
        >>> target = torch.rand(64, 64, 3)
        >>> loss = ssim_loss(pred, target)
    """
    # 将输入统一为 [B, C, H, W] 格式，方便卷积操作
    if pred.dim() == 3:
        # [H, W, C] -> [1, C, H, W]
        pred = pred.permute(2, 0, 1).unsqueeze(0)
        target = target.permute(2, 0, 1).unsqueeze(0)
    elif pred.dim() == 4 and pred.shape[-1] <= 4:
        # [B, H, W, C] -> [B, C, H, W]
        pred = pred.permute(0, 3, 1, 2)
        target = target.permute(0, 3, 1, 2)

    C = pred.shape[1]  # 通道数

    # 创建 1D Gaussian kernel，然后展成 2D
    gauss_1d = _create_gaussian_kernel_1d(window_size, sigma=1.5, device=pred.device)
    kernel_2d = gauss_1d.unsqueeze(-1) @ gauss_1d.unsqueeze(0)  # [W, W]
    kernel_2d = kernel_2d.expand(C, 1, window_size, window_size)  # [C, 1, W, W]

    pad = window_size // 2

    # 局部均值 μ_x, μ_y（每通道独立卷积, groups=C）
    mu_x = F.conv2d(pred, kernel_2d, padding=pad, groups=C)
    mu_y = F.conv2d(target, kernel_2d, padding=pad, groups=C)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    # 局部方差 σ² 和协方差 σ_xy
    sigma_x_sq = F.conv2d(pred * pred, kernel_2d, padding=pad, groups=C) - mu_x_sq
    sigma_y_sq = F.conv2d(target * target, kernel_2d, padding=pad, groups=C) - mu_y_sq
    sigma_xy = F.conv2d(pred * target, kernel_2d, padding=pad, groups=C) - mu_xy

    # 稳定性常数（假设像素值范围为 [0, 1]）
    C1 = 0.01 ** 2  # = 0.0001
    C2 = 0.03 ** 2  # = 0.0009

    # SSIM 公式
    numerator = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    ssim_map = numerator / denominator  # 每像素的 SSIM 值

    # 返回 1 - mean(SSIM)，使之成为需要最小化的损失
    return 1.0 - ssim_map.mean()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_l1: float = 0.8,
    lambda_ssim: float = 0.2,
) -> torch.Tensor:
    """
    Compute the combined loss used in the original 3DGS paper.

    The combined loss is a weighted sum of L1 and SSIM losses:
        L = λ_l1 * L_1 + λ_ssim * L_SSIM

    The original paper uses λ_l1 = 0.8 and λ_ssim = 0.2:
        L = 0.8 * L_1 + 0.2 * L_SSIM

    This combination leverages L1 for pixel-accurate reconstruction
    and SSIM for perceptually consistent structure.

    Args:
        pred: Predicted image [H, W, C] or [B, C, H, W]
        target: Ground truth image, same shape as pred
        lambda_l1: Weight for L1 loss (default: 0.8)
        lambda_ssim: Weight for SSIM loss (default: 0.2)

    Returns:
        Scalar combined loss value

    Example:
        >>> pred = torch.rand(64, 64, 3)
        >>> target = torch.rand(64, 64, 3)
        >>> loss = combined_loss(pred, target)
    """
    loss_l1 = l1_loss(pred, target)
    loss_ssim = ssim_loss(pred, target)
    return lambda_l1 * loss_l1 + lambda_ssim * loss_ssim


def _create_gaussian_kernel_1d(
    size: int,
    sigma: float,
    device: torch.device,
) -> torch.Tensor:
    """
    Create a 1D Gaussian kernel for SSIM window.

    The kernel is defined as:
        g(x) = exp(-x² / (2σ²))

    then normalized so all values sum to 1.

    Args:
        size: Kernel size (should be odd)
        sigma: Standard deviation of the Gaussian
        device: Torch device for the tensor

    Returns:
        Normalized 1D Gaussian kernel [size]
    """
    coords = torch.arange(size, dtype=torch.float32, device=device)
    coords -= size // 2  # 中心化: [-size//2, ..., 0, ..., size//2]

    kernel = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()  # 归一化

    return kernel
