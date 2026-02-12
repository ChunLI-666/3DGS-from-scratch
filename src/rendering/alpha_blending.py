"""
Alpha Blending for 3D Gaussian Splatting

This module provides functions for alpha compositing of Gaussians,
following the standard front-to-back blending formula used in 3DGS.

Alpha Blending Formula:
    C = Σ (T_i * α_i * c_i)

where:
    - T_i = Π_{j<i} (1 - α_j) is the transmittance
    - α_i is the alpha (opacity * gaussian value) of Gaussian i
    - c_i is the color of Gaussian i
"""

import torch
from typing import Tuple, Optional


def compute_transmittance(alphas: torch.Tensor) -> torch.Tensor:
    """
    Compute transmittance for each Gaussian.

    Transmittance T_i is the product of (1 - α_j) for all j < i.
    This represents how much light can pass through all previous Gaussians.

    Args:
        alphas: Alpha values [N] or [H, W, N] sorted front-to-back

    Returns:
        Transmittance values, same shape as input
    """
    if alphas.dim() == 1:
        # [N] case - single pixel
        one_minus_alpha = 1 - alphas
        # T_i = prod_{j<i}(1 - α_j)
        # T_0 = 1, T_1 = 1-α_0, T_2 = (1-α_0)(1-α_1), ...
        transmittance = torch.ones_like(alphas)
        transmittance[1:] = torch.cumprod(one_minus_alpha[:-1], dim=0)
        return transmittance

    elif alphas.dim() == 3:
        # [H, W, N] case - full image
        one_minus_alpha = 1 - alphas
        transmittance = torch.ones_like(alphas)
        transmittance[:, :, 1:] = torch.cumprod(
            one_minus_alpha[:, :, :-1], dim=-1
        )
        return transmittance

    else:
        raise ValueError(f"Expected 1D or 3D tensor, got {alphas.dim()}D")


def alpha_blend(
    colors: torch.Tensor,
    alphas: torch.Tensor,
    background: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Perform alpha blending of Gaussians (front-to-back).

    Args:
        colors: Color values [N, 3] or [H, W, N, 3] sorted front-to-back
        alphas: Alpha values [N] or [H, W, N] sorted front-to-back
        background: Background color [3] (default: white)

    Returns:
        Blended color [3] or [H, W, 3]
    """
    if background is None:
        background = torch.ones(3, dtype=colors.dtype, device=colors.device)

    if alphas.dim() == 1:
        # Single pixel case: [N] alphas, [N, 3] colors
        transmittance = compute_transmittance(alphas)  # [N]

        # Weighted sum: C = Σ T_i * α_i * c_i
        weights = transmittance * alphas  # [N]
        blended = (weights.unsqueeze(-1) * colors).sum(dim=0)  # [3]

        # Add background: final_T = Π(1-α_i)
        final_transmittance = (1 - alphas).prod()
        blended = blended + final_transmittance * background

        return blended

    elif alphas.dim() == 3:
        # Full image case: [H, W, N] alphas, [H, W, N, 3] colors
        H, W, N = alphas.shape
        transmittance = compute_transmittance(alphas)  # [H, W, N]

        # Weighted sum
        weights = transmittance * alphas  # [H, W, N]
        blended = (weights.unsqueeze(-1) * colors).sum(dim=2)  # [H, W, 3]

        # Add background
        final_transmittance = (1 - alphas).prod(dim=-1)  # [H, W]
        blended = blended + final_transmittance.unsqueeze(-1) * background

        return blended

    else:
        raise ValueError(f"Expected 1D or 3D alphas, got {alphas.dim()}D")


def alpha_blend_backward(
    grad_output: torch.Tensor,
    colors: torch.Tensor,
    alphas: torch.Tensor,
    transmittance: torch.Tensor,
    background: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Backward pass for alpha blending.

    Computes gradients with respect to colors and alphas.

    Args:
        grad_output: Gradient of loss with respect to blended color [3] or [H, W, 3]
        colors: Forward pass colors [N, 3] or [H, W, N, 3]
        alphas: Forward pass alphas [N] or [H, W, N]
        transmittance: Forward pass transmittance [N] or [H, W, N]
        background: Background color [3]

    Returns:
        Tuple of (grad_colors, grad_alphas)
    """
    if background is None:
        background = torch.ones(3, dtype=colors.dtype, device=colors.device)

    if alphas.dim() == 1:
        N = alphas.shape[0]

        # Gradient w.r.t. colors: dL/dc_i = dL/dC * T_i * α_i
        weights = transmittance * alphas  # [N]
        grad_colors = grad_output.unsqueeze(0) * weights.unsqueeze(-1)  # [N, 3]

        # Gradient w.r.t. alphas is more complex
        # dL/dα_i = dL/dC · (T_i * c_i - Σ_{j>i} T_j * α_j * c_j / (1-α_i))
        #         - dL/dC · final_T * background / (1-α_i)  (if α_i affects final_T)

        grad_alphas = torch.zeros(N, device=alphas.device, dtype=alphas.dtype)

        # Simple gradient: contribution from direct term
        for i in range(N):
            # Direct contribution: T_i * c_i
            direct = transmittance[i] * (grad_output * colors[i]).sum()

            # Contribution through transmittance of later terms
            indirect = 0.0
            for j in range(i + 1, N):
                # T_j depends on (1-α_i), so dT_j/dα_i = -T_j/(1-α_i)
                if alphas[i] < 1:
                    weight_j = transmittance[j] * alphas[j]
                    indirect -= (weight_j / (1 - alphas[i])) * (
                        grad_output * colors[j]
                    ).sum()

            # Background contribution
            final_T = (1 - alphas).prod()
            if alphas[i] < 1:
                bg_contrib = -(final_T / (1 - alphas[i])) * (
                    grad_output * background
                ).sum()
            else:
                bg_contrib = 0.0

            grad_alphas[i] = direct + indirect + bg_contrib

        return grad_colors, grad_alphas

    else:
        # Batched version - similar logic
        # For efficiency, we use vectorized operations
        H, W, N = alphas.shape

        # Gradient w.r.t. colors
        weights = transmittance * alphas  # [H, W, N]
        grad_colors = grad_output.unsqueeze(2) * weights.unsqueeze(-1)  # [H, W, N, 3]

        # Gradient w.r.t. alphas (simplified version)
        # This computes the direct gradient, not the full chain rule
        grad_alphas = (
            transmittance * (grad_output.unsqueeze(2) * colors).sum(dim=-1)
        )

        return grad_colors, grad_alphas


class AlphaBlendFunction(torch.autograd.Function):
    """
    Autograd function for alpha blending.

    This enables gradient computation through the blending operation.
    """

    @staticmethod
    def forward(
        ctx,
        colors: torch.Tensor,
        alphas: torch.Tensor,
        background: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass."""
        transmittance = compute_transmittance(alphas)
        result = alpha_blend(colors, alphas, background)

        # Save for backward
        ctx.save_for_backward(colors, alphas, transmittance, background)

        return result

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass."""
        colors, alphas, transmittance, background = ctx.saved_tensors

        grad_colors, grad_alphas = alpha_blend_backward(
            grad_output, colors, alphas, transmittance, background
        )

        return grad_colors, grad_alphas, None


def differentiable_alpha_blend(
    colors: torch.Tensor,
    alphas: torch.Tensor,
    background: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Differentiable alpha blending with autograd support.

    Args:
        colors: Color values [N, 3] or [H, W, N, 3]
        alphas: Alpha values [N] or [H, W, N]
        background: Background color [3]

    Returns:
        Blended color with gradient support
    """
    if background is None:
        background = torch.ones(3, dtype=colors.dtype, device=colors.device)

    # Use simple implementation for now (torch's autograd handles gradients)
    return alpha_blend(colors, alphas, background)


def blend_ordered_gaussians(
    gaussian_colors: torch.Tensor,
    gaussian_alphas: torch.Tensor,
    pixel_values: torch.Tensor,
    pixel_opacities: torch.Tensor,
    background: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Blend Gaussians at a pixel given their evaluated values.

    This is a higher-level function that combines:
    1. Alpha computation: α = gaussian_value * opacity
    2. Alpha blending

    Args:
        gaussian_colors: Colors of Gaussians [N, 3]
        gaussian_alphas: Base opacities [N]
        pixel_values: Gaussian values at this pixel [N] (0-1)
        pixel_opacities: Per-Gaussian opacity [N]
        background: Background color [3]

    Returns:
        Final pixel color [3]
    """
    # Compute effective alpha
    alphas = pixel_values * pixel_opacities

    return alpha_blend(gaussian_colors, alphas, background)
