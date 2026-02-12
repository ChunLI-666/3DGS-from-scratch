"""
Spherical Harmonics Utilities for 3D Gaussian Splatting

This module provides:
1. Spherical Harmonics (SH) coefficient constants
2. SH evaluation functions for view-dependent color
3. RGB to SH and SH to RGB conversion utilities

Spherical Harmonics enable view-dependent appearance in 3DGS,
allowing surfaces to exhibit specular highlights, reflections,
and other view-dependent effects.
"""

import torch
import math
from typing import Optional


# ============================================================================
# Spherical Harmonics Constants
# ============================================================================

# Degree 0 (1 coefficient) - Constant/DC component
SH_C0 = 0.28209479177387814  # 1 / (2 * sqrt(pi))

# Degree 1 (3 coefficients) - Linear terms
SH_C1 = 0.4886025119029199  # sqrt(3) / (2 * sqrt(pi))

# Degree 2 (5 coefficients) - Quadratic terms
SH_C2 = [
    1.0925484305920792,   # sqrt(15) / (2 * sqrt(pi))
    -1.0925484305920792,  # -sqrt(15) / (2 * sqrt(pi))
    0.31539156525252005,  # sqrt(5) / (4 * sqrt(pi))
    -1.0925484305920792,  # -sqrt(15) / (2 * sqrt(pi))
    0.5462742152960396,   # sqrt(15) / (4 * sqrt(pi))
]

# Degree 3 (7 coefficients) - Cubic terms
SH_C3 = [
    -0.5900435899266435,   # -sqrt(70) / (8 * sqrt(pi))
    2.890611442640554,     # sqrt(105) / (2 * sqrt(pi))
    -0.4570457994644658,   # -sqrt(42) / (8 * sqrt(pi))
    0.3731763325901154,    # sqrt(7) / (4 * sqrt(pi))
    -0.4570457994644658,   # -sqrt(42) / (8 * sqrt(pi))
    1.445305721320277,     # sqrt(105) / (4 * sqrt(pi))
    -0.5900435899266435,   # -sqrt(70) / (8 * sqrt(pi))
]

# Total number of coefficients per degree
SH_DEGREES = {
    0: 1,
    1: 4,   # 1 + 3 = 4
    2: 9,   # 1 + 3 + 5 = 9
    3: 16,  # 1 + 3 + 5 + 7 = 16
}


# ============================================================================
# Core Evaluation Functions
# ============================================================================

def eval_sh_degree_0(sh: torch.Tensor) -> torch.Tensor:
    """
    Evaluate degree 0 (constant) spherical harmonics.

    Args:
        sh: SH coefficients [N, 1, 3] or [N, 3]

    Returns:
        Color [N, 3]
    """
    if sh.dim() == 2:
        return SH_C0 * sh
    return SH_C0 * sh[:, 0]


def eval_sh_degree_1(
    sh: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate up to degree 1 spherical harmonics.

    Degree 1 terms represent linear variation:
    Y_1^{-1} = sqrt(3)/(2*sqrt(pi)) * y
    Y_1^0 = sqrt(3)/(2*sqrt(pi)) * z
    Y_1^1 = sqrt(3)/(2*sqrt(pi)) * x

    Args:
        sh: SH coefficients [N, 4, 3] for 4 coefficients per color
        direction: View directions [N, 3] (normalized)

    Returns:
        Color [N, 3]
    """
    x, y, z = direction[:, 0:1], direction[:, 1:2], direction[:, 2:3]

    # Degree 0
    result = SH_C0 * sh[:, 0]

    # Degree 1
    result = result + SH_C1 * (-y * sh[:, 1] + z * sh[:, 2] - x * sh[:, 3])

    return result


def eval_sh_degree_2(
    sh: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate up to degree 2 spherical harmonics.

    Args:
        sh: SH coefficients [N, 9, 3] for 9 coefficients per color
        direction: View directions [N, 3] (normalized)

    Returns:
        Color [N, 3]
    """
    x, y, z = direction[:, 0:1], direction[:, 1:2], direction[:, 2:3]
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    # Degree 0
    result = SH_C0 * sh[:, 0]

    # Degree 1
    result = result + SH_C1 * (-y * sh[:, 1] + z * sh[:, 2] - x * sh[:, 3])

    # Degree 2
    result = result + SH_C2[0] * xy * sh[:, 4]
    result = result + SH_C2[1] * yz * sh[:, 5]
    result = result + SH_C2[2] * (2.0 * zz - xx - yy) * sh[:, 6]
    result = result + SH_C2[3] * xz * sh[:, 7]
    result = result + SH_C2[4] * (xx - yy) * sh[:, 8]

    return result


def eval_sh_degree_3(
    sh: torch.Tensor,
    direction: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate up to degree 3 spherical harmonics.

    Args:
        sh: SH coefficients [N, 16, 3] for 16 coefficients per color
        direction: View directions [N, 3] (normalized)

    Returns:
        Color [N, 3]
    """
    x, y, z = direction[:, 0:1], direction[:, 1:2], direction[:, 2:3]
    xx, yy, zz = x * x, y * y, z * z
    xy, yz, xz = x * y, y * z, x * z

    # Degree 0
    result = SH_C0 * sh[:, 0]

    # Degree 1
    result = result + SH_C1 * (-y * sh[:, 1] + z * sh[:, 2] - x * sh[:, 3])

    # Degree 2
    result = result + SH_C2[0] * xy * sh[:, 4]
    result = result + SH_C2[1] * yz * sh[:, 5]
    result = result + SH_C2[2] * (2.0 * zz - xx - yy) * sh[:, 6]
    result = result + SH_C2[3] * xz * sh[:, 7]
    result = result + SH_C2[4] * (xx - yy) * sh[:, 8]

    # Degree 3
    result = result + SH_C3[0] * y * (3.0 * xx - yy) * sh[:, 9]
    result = result + SH_C3[1] * xy * z * sh[:, 10]
    result = result + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh[:, 11]
    result = result + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh[:, 12]
    result = result + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh[:, 13]
    result = result + SH_C3[5] * z * (xx - yy) * sh[:, 14]
    result = result + SH_C3[6] * x * (xx - 3.0 * yy) * sh[:, 15]

    return result


def eval_sh(
    sh: torch.Tensor,
    direction: torch.Tensor,
    degree: int = 3,
) -> torch.Tensor:
    """
    Evaluate spherical harmonics at given view directions.

    This is the main entry point for SH evaluation in 3DGS.

    Args:
        sh: SH coefficients [N, K, 3] where K = (degree+1)^2
        direction: View directions [N, 3] (should be normalized)
        degree: Maximum SH degree to evaluate (0, 1, 2, or 3)

    Returns:
        RGB color values [N, 3]

    Example:
        >>> sh = torch.randn(100, 16, 3)  # 100 Gaussians, degree 3
        >>> directions = torch.randn(100, 3)
        >>> directions = directions / directions.norm(dim=-1, keepdim=True)
        >>> colors = eval_sh(sh, directions, degree=3)
    """
    if degree == 0:
        return eval_sh_degree_0(sh)
    elif degree == 1:
        return eval_sh_degree_1(sh, direction)
    elif degree == 2:
        return eval_sh_degree_2(sh, direction)
    elif degree == 3:
        return eval_sh_degree_3(sh, direction)
    else:
        raise ValueError(f"SH degree {degree} not supported. Use 0, 1, 2, or 3.")


# ============================================================================
# Utility Functions
# ============================================================================

def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB color to zeroth-order SH coefficient (DC component).

    This allows initializing SH coefficients from RGB colors.
    The inverse operation is: rgb = SH_C0 * sh_dc

    Args:
        rgb: RGB colors [N, 3] in range [0, 1]

    Returns:
        SH coefficients [N, 3] for the DC term
    """
    return (rgb - 0.5) / SH_C0


def sh_to_rgb(sh_dc: torch.Tensor) -> torch.Tensor:
    """
    Convert zeroth-order SH coefficient back to RGB.

    Args:
        sh_dc: DC SH coefficients [N, 3]

    Returns:
        RGB colors [N, 3]
    """
    return SH_C0 * sh_dc + 0.5


def initialize_sh_from_rgb(
    rgb: torch.Tensor,
    degree: int = 3,
) -> torch.Tensor:
    """
    Initialize SH coefficients from RGB colors.

    The DC term is set from the RGB value, and all higher-order
    terms are initialized to zero (diffuse assumption).

    Args:
        rgb: RGB colors [N, 3]
        degree: Maximum SH degree

    Returns:
        SH coefficients [N, (degree+1)^2, 3]
    """
    N = rgb.shape[0]
    n_coeffs = (degree + 1) ** 2

    sh = torch.zeros(N, n_coeffs, 3, device=rgb.device, dtype=rgb.dtype)
    sh[:, 0, :] = rgb_to_sh(rgb)

    return sh


def get_direction_from_camera(
    gaussian_means: torch.Tensor,
    camera_center: torch.Tensor,
) -> torch.Tensor:
    """
    Compute view directions from Gaussian positions to camera.

    Args:
        gaussian_means: 3D positions of Gaussians [N, 3]
        camera_center: Camera position in world coordinates [3]

    Returns:
        Normalized view directions [N, 3]
    """
    direction = camera_center.unsqueeze(0) - gaussian_means
    direction = direction / (direction.norm(dim=-1, keepdim=True) + 1e-8)
    return direction


# ============================================================================
# Visualization Helpers
# ============================================================================

def visualize_sh_basis(degree: int = 2, resolution: int = 100):
    """
    Generate visualization data for SH basis functions.

    Creates a spherical sampling and evaluates each basis function.
    Useful for understanding what each SH coefficient represents.

    Args:
        degree: Maximum SH degree
        resolution: Number of samples in each angular direction

    Returns:
        Dict with 'theta', 'phi', 'x', 'y', 'z', and 'basis_values'
    """
    # Create spherical grid
    theta = torch.linspace(0, math.pi, resolution)
    phi = torch.linspace(0, 2 * math.pi, resolution)
    theta_grid, phi_grid = torch.meshgrid(theta, phi, indexing='ij')

    # Convert to Cartesian
    x = torch.sin(theta_grid) * torch.cos(phi_grid)
    y = torch.sin(theta_grid) * torch.sin(phi_grid)
    z = torch.cos(theta_grid)

    # Stack directions
    directions = torch.stack([
        x.flatten(), y.flatten(), z.flatten()
    ], dim=-1)

    N = directions.shape[0]
    n_coeffs = (degree + 1) ** 2

    # Evaluate each basis function
    basis_values = []
    for i in range(n_coeffs):
        # Create one-hot SH coefficients
        sh = torch.zeros(N, n_coeffs, 3)
        sh[:, i, :] = 1.0

        # Evaluate (for RGB, average channels)
        value = eval_sh(sh, directions, degree=degree)
        value = value.mean(dim=-1).reshape(resolution, resolution)
        basis_values.append(value)

    return {
        'theta': theta_grid,
        'phi': phi_grid,
        'x': x,
        'y': y,
        'z': z,
        'basis_values': basis_values,
        'resolution': resolution,
    }


class SphericalHarmonicsEncoder(torch.nn.Module):
    """
    Spherical Harmonics encoder for view-dependent appearance.

    This module wraps SH evaluation in an nn.Module for use in
    neural network pipelines.

    Args:
        degree: Maximum SH degree (0-3)

    Example:
        >>> encoder = SphericalHarmonicsEncoder(degree=3)
        >>> sh_coeffs = torch.randn(100, 16, 3)  # Learnable parameters
        >>> directions = torch.randn(100, 3)
        >>> directions = directions / directions.norm(dim=-1, keepdim=True)
        >>> colors = encoder(sh_coeffs, directions)
    """

    def __init__(self, degree: int = 3):
        super().__init__()
        self.degree = degree
        self.n_coefficients = (degree + 1) ** 2

    def forward(
        self,
        sh_coefficients: torch.Tensor,
        directions: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate SH at given directions.

        Args:
            sh_coefficients: [N, K, 3] where K = (degree+1)^2
            directions: [N, 3] normalized view directions

        Returns:
            [N, 3] RGB colors
        """
        return eval_sh(sh_coefficients, directions, self.degree)

    def extra_repr(self) -> str:
        return f'degree={self.degree}, n_coefficients={self.n_coefficients}'
