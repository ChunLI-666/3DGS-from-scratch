"""
Pixel-aligned Gaussian Representation for Feed-forward 3DGS.

In feed-forward methods (MVSplat, pixelSplat), each pixel in the input
image corresponds to exactly one 3D Gaussian. The Gaussian center is
obtained by back-projecting the pixel coordinate to 3D using the
predicted depth. Other Gaussian properties (scale, rotation, opacity)
are predicted by neural network heads.

This is fundamentally different from original 3DGS where Gaussians
are freely positioned and optimized via gradient descent.

Key concepts:
- Back-projection: pixel (u,v) + depth d → 3D point (X, Y, Z)
- Structured layout: Gaussians follow image grid structure
- No densification: Fixed number of Gaussians = H × W per view

Reference:
- MVSplat (ECCV 2024): https://arxiv.org/abs/2403.14627
- pixelSplat (CVPR 2024): https://arxiv.org/abs/2312.12337
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


def create_pixel_grid(
    height: int,
    width: int,
    device: torch.device = torch.device('cpu'),
    homogeneous: bool = True,
) -> torch.Tensor:
    """
    Create a pixel coordinate grid.

    Args:
        height: image height
        width: image width
        device: torch device
        homogeneous: if True, return [3, H, W] with ones; else [2, H, W]

    Returns:
        grid: [2, H, W] or [3, H, W] pixel coordinates
    """
    u = torch.arange(width, device=device, dtype=torch.float32)
    v = torch.arange(height, device=device, dtype=torch.float32)
    v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')

    if homogeneous:
        ones = torch.ones_like(u_grid)
        grid = torch.stack([u_grid, v_grid, ones], dim=0)  # [3, H, W]
    else:
        grid = torch.stack([u_grid, v_grid], dim=0)  # [2, H, W]

    return grid


def unproject_depth_to_3d(
    depth: torch.Tensor,
    K: torch.Tensor,
    pose: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Back-project a depth map to 3D points.

    This is the core operation for creating pixel-aligned Gaussians:
    each pixel (u, v) with predicted depth d is mapped to 3D point:
        X = (u - cx) * d / fx
        Y = (v - cy) * d / fy
        Z = d

    Args:
        depth: [B, 1, H, W] depth map
        K: [B, 3, 3] camera intrinsics
        pose: [B, 4, 4] optional camera-to-world pose
              If provided, transforms points to world coordinates.

    Returns:
        points_3d: [B, H, W, 3] 3D point coordinates
    """
    B, _, H, W = depth.shape
    device = depth.device

    # Create pixel grid
    pixel_grid = create_pixel_grid(H, W, device=device)  # [3, H, W]
    pixel_grid = pixel_grid.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 3, H, W]

    # Flatten for matrix multiplication
    pixels_flat = pixel_grid.reshape(B, 3, -1)  # [B, 3, H*W]

    # Back-project: X_cam = K^{-1} @ [u, v, 1]^T * depth
    K_inv = torch.inverse(K)  # [B, 3, 3]
    rays = torch.bmm(K_inv, pixels_flat)  # [B, 3, H*W]

    # Scale by depth
    depth_flat = depth.reshape(B, 1, -1)  # [B, 1, H*W]
    points_cam = rays * depth_flat  # [B, 3, H*W]

    # Optionally transform to world coordinates
    if pose is not None:
        R = pose[:, :3, :3]  # [B, 3, 3]
        t = pose[:, :3, 3:]  # [B, 3, 1]
        points_world = torch.bmm(R, points_cam) + t  # [B, 3, H*W]
        points_3d = points_world.reshape(B, 3, H, W).permute(0, 2, 3, 1)
    else:
        points_3d = points_cam.reshape(B, 3, H, W).permute(0, 2, 3, 1)

    return points_3d  # [B, H, W, 3]


class PixelAlignedGaussians:
    """
    Pixel-aligned Gaussian representation.

    Stores Gaussians that are organized in a grid corresponding to
    input image pixels. Each pixel contributes one Gaussian.

    Attributes:
        positions: [B, N, 3] 3D positions (from back-projection)
        scales: [B, N, 3] or [B, N, 2] Gaussian scales
        rotations: [B, N, 4] quaternion rotations (w, x, y, z)
        opacities: [B, N, 1] opacity values in [0, 1]
        colors: [B, N, 3] RGB colors or [B, N, K*3] SH coefficients
        height: original image height
        width: original image width
    """

    def __init__(
        self,
        positions: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        height: int,
        width: int,
    ):
        self.positions = positions
        self.scales = scales
        self.rotations = rotations
        self.opacities = opacities
        self.colors = colors
        self.height = height
        self.width = width

    @property
    def num_gaussians(self) -> int:
        return self.positions.shape[1]

    @property
    def batch_size(self) -> int:
        return self.positions.shape[0]

    @classmethod
    def from_depth_and_features(
        cls,
        depth: torch.Tensor,
        features: Dict[str, torch.Tensor],
        K: torch.Tensor,
        pose: Optional[torch.Tensor] = None,
        image_colors: Optional[torch.Tensor] = None,
    ) -> 'PixelAlignedGaussians':
        """
        Create pixel-aligned Gaussians from predicted depth and features.

        This is the main factory method used during inference:
        1. Back-project depth to get 3D positions
        2. Use network-predicted features for scale, rotation, opacity
        3. Use image colors directly (or predicted SH coefficients)

        Args:
            depth: [B, 1, H, W] predicted depth map
            features: dict with keys:
                'scales': [B, 3, H, W] or [B, 2, H, W]
                'rotations': [B, 4, H, W] quaternion
                'opacities': [B, 1, H, W]
            K: [B, 3, 3] camera intrinsics
            pose: [B, 4, 4] optional camera-to-world pose
            image_colors: [B, 3, H, W] optional image RGB for color

        Returns:
            PixelAlignedGaussians instance
        """
        B, _, H, W = depth.shape
        N = H * W

        # Step 1: Back-project to 3D
        positions = unproject_depth_to_3d(depth, K, pose)  # [B, H, W, 3]
        positions = positions.reshape(B, N, 3)

        # Step 2: Reshape predictions
        scales = features['scales'].permute(0, 2, 3, 1).reshape(B, N, -1)
        rotations = features['rotations'].permute(0, 2, 3, 1).reshape(B, N, 4)
        opacities = features['opacities'].permute(0, 2, 3, 1).reshape(B, N, 1)

        # Normalize quaternion
        rotations = rotations / (rotations.norm(dim=-1, keepdim=True) + 1e-8)

        # Step 3: Colors
        if image_colors is not None:
            colors = image_colors.permute(0, 2, 3, 1).reshape(B, N, 3)
        else:
            colors = torch.ones(B, N, 3, device=depth.device) * 0.5

        return cls(
            positions=positions,
            scales=scales,
            rotations=rotations,
            opacities=opacities,
            colors=colors,
            height=H,
            width=W,
        )

    def to_dict(self) -> Dict[str, torch.Tensor]:
        """Convert to dictionary format for rendering."""
        return {
            'positions': self.positions,
            'scales': self.scales,
            'rotations': self.rotations,
            'opacities': self.opacities,
            'colors': self.colors,
        }

    def get_depth_map(self) -> torch.Tensor:
        """
        Extract depth map from Gaussian positions.

        Returns:
            depth_map: [B, 1, H, W] depth values (Z coordinate)
        """
        B = self.batch_size
        z = self.positions[:, :, 2]  # [B, N]
        return z.reshape(B, 1, self.height, self.width)

    def merge(self, other: 'PixelAlignedGaussians') -> 'PixelAlignedGaussians':
        """
        Merge Gaussians from another view.

        In multi-view feed-forward methods, Gaussians from each input
        view are combined into a single set for rendering.

        Args:
            other: PixelAlignedGaussians from another view

        Returns:
            merged: PixelAlignedGaussians with concatenated Gaussians
        """
        return PixelAlignedGaussians(
            positions=torch.cat([self.positions, other.positions], dim=1),
            scales=torch.cat([self.scales, other.scales], dim=1),
            rotations=torch.cat([self.rotations, other.rotations], dim=1),
            opacities=torch.cat([self.opacities, other.opacities], dim=1),
            colors=torch.cat([self.colors, other.colors], dim=1),
            height=self.height,
            width=self.width,
        )

    def __repr__(self) -> str:
        return (
            f"PixelAlignedGaussians("
            f"batch_size={self.batch_size}, "
            f"num_gaussians={self.num_gaussians}, "
            f"resolution=({self.height}, {self.width}))"
        )
