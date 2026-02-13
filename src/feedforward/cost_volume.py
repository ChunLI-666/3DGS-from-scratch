"""
Cost Volume and Plane Sweeping for Feed-forward 3DGS.

This module implements the core geometric reasoning component used in
MVSplat and similar methods. The Cost Volume encodes multi-view matching
costs at different depth hypotheses, enabling the network to reason
about scene geometry.

Key concepts:
- Plane Sweeping: Sample depth hypotheses and warp features
- Cost Volume: Store matching costs at each (pixel, depth) location
- Soft Argmin: Differentiable depth regression from cost volume

Reference:
- MVSplat (ECCV 2024): https://arxiv.org/abs/2403.14627
- MVSNet (ECCV 2018): https://arxiv.org/abs/1804.02505
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def create_depth_planes(
    num_depths: int = 64,
    min_depth: float = 0.5,
    max_depth: float = 100.0,
    sampling: str = 'log_uniform',
) -> torch.Tensor:
    """
    Create depth hypothesis planes for plane sweeping.

    Args:
        num_depths: Number of depth planes to sample
        min_depth: Minimum depth value
        max_depth: Maximum depth value
        sampling: Sampling strategy - 'uniform' or 'log_uniform'

    Returns:
        depth_planes: [D] tensor of depth values
    """
    if sampling == 'uniform':
        depth_planes = torch.linspace(min_depth, max_depth, num_depths)
    elif sampling == 'log_uniform':
        # Log-uniform sampling: more planes near camera (where precision matters)
        depth_planes = torch.exp(
            torch.linspace(math.log(min_depth), math.log(max_depth), num_depths)
        )
    else:
        raise ValueError(f"Unknown sampling strategy: {sampling}")

    return depth_planes


def homography_warp(
    feat_src: torch.Tensor,
    depth: float,
    K_ref: torch.Tensor,
    K_src: torch.Tensor,
    T_src_ref: torch.Tensor,
) -> torch.Tensor:
    """
    Warp source features to reference view at a given depth via homography.

    For a depth hypothesis d, the homography from reference to source is:
        H = K_src @ (R - t @ n^T / d) @ K_ref^{-1}

    For a fronto-parallel plane (n = [0, 0, 1]), this simplifies to
    standard plane-induced homography.

    Args:
        feat_src: [B, C, H, W] source feature map
        depth: scalar depth value for this plane
        K_ref: [B, 3, 3] reference camera intrinsics
        K_src: [B, 3, 3] source camera intrinsics
        T_src_ref: [B, 4, 4] transformation from reference to source frame

    Returns:
        warped_feat: [B, C, H, W] source features warped to reference view
    """
    B, C, H, W = feat_src.shape
    device = feat_src.device

    # Extract rotation and translation
    R = T_src_ref[:, :3, :3]  # [B, 3, 3]
    t = T_src_ref[:, :3, 3:]  # [B, 3, 1]

    # Normal vector for fronto-parallel plane (in reference frame)
    n = torch.tensor([0.0, 0.0, 1.0], device=device).reshape(1, 3, 1)
    n = n.expand(B, -1, -1)

    # Homography: H = K_src @ (R - t @ n^T / d) @ K_ref^{-1}
    # Simplified: H = K_src @ (R + t @ n^T / d) @ K_ref^{-1}
    # (sign depends on convention)
    tn = torch.bmm(t, n.transpose(1, 2))  # [B, 3, 3]
    H = R + tn / depth  # [B, 3, 3]
    H = torch.bmm(K_src, torch.bmm(H, torch.inverse(K_ref)))  # [B, 3, 3]

    # Create pixel coordinate grid for reference image
    u = torch.arange(W, device=device, dtype=torch.float32)
    v = torch.arange(H, device=device, dtype=torch.float32)
    v_grid, u_grid = torch.meshgrid(v, u, indexing='ij')
    ones = torch.ones_like(u_grid)
    pixel_coords = torch.stack([u_grid, v_grid, ones], dim=0)  # [3, H, W]
    pixel_coords = pixel_coords.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 3, H, W]
    pixel_coords_flat = pixel_coords.reshape(B, 3, -1)  # [B, 3, H*W]

    # Warp: source_coords = H @ ref_coords
    src_coords = torch.bmm(H, pixel_coords_flat)  # [B, 3, H*W]
    src_coords = src_coords[:, :2] / (src_coords[:, 2:3] + 1e-8)  # [B, 2, H*W]
    src_coords = src_coords.reshape(B, 2, H, W)

    # Normalize to [-1, 1] for grid_sample
    src_coords_norm = torch.zeros_like(src_coords)
    src_coords_norm[:, 0] = 2.0 * src_coords[:, 0] / (W - 1) - 1.0
    src_coords_norm[:, 1] = 2.0 * src_coords[:, 1] / (H - 1) - 1.0
    grid = src_coords_norm.permute(0, 2, 3, 1)  # [B, H, W, 2]

    # Sample source features
    warped_feat = F.grid_sample(
        feat_src, grid, mode='bilinear', padding_mode='zeros', align_corners=True
    )

    return warped_feat


class PlaneSweeper(nn.Module):
    """
    Plane Sweeping module for cost volume construction.

    Sweeps through depth hypotheses and warps source features to
    the reference view at each depth, building a 4D cost volume.
    """

    def __init__(
        self,
        num_depths: int = 64,
        min_depth: float = 0.5,
        max_depth: float = 100.0,
        sampling: str = 'log_uniform',
    ):
        super().__init__()
        self.num_depths = num_depths
        self.min_depth = min_depth
        self.max_depth = max_depth

        depth_planes = create_depth_planes(num_depths, min_depth, max_depth, sampling)
        self.register_buffer('depth_planes', depth_planes)

    def forward(
        self,
        feat_ref: torch.Tensor,
        feat_src: torch.Tensor,
        K_ref: torch.Tensor,
        K_src: torch.Tensor,
        T_src_ref: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build raw cost volume by warping and computing matching cost.

        Args:
            feat_ref: [B, C, H, W] reference feature map
            feat_src: [B, C, H, W] source feature map
            K_ref: [B, 3, 3] reference camera intrinsics
            K_src: [B, 3, 3] source camera intrinsics
            T_src_ref: [B, 4, 4] ref-to-source transformation

        Returns:
            cost_volume: [B, C, D, H, W] matching cost at each depth
        """
        B, C, H, W = feat_ref.shape
        D = self.num_depths

        cost_slices = []
        for i in range(D):
            depth = self.depth_planes[i].item()

            # Warp source features to reference at this depth
            warped = homography_warp(feat_src, depth, K_ref, K_src, T_src_ref)

            # Compute matching cost: variance (lower = better match)
            cost = (feat_ref - warped).pow(2)  # [B, C, H, W]
            cost_slices.append(cost)

        # Stack along depth dimension
        cost_volume = torch.stack(cost_slices, dim=2)  # [B, C, D, H, W]

        return cost_volume


class CostVolumeBuilder(nn.Module):
    """
    Complete Cost Volume builder supporting multiple source views.

    For N source views, builds individual cost volumes and aggregates
    them via mean variance (as in MVSNet/MVSplat).

    This is the core geometric reasoning component:
    - Low cost at depth d → features match well → likely correct depth
    - High cost at depth d → features don't match → wrong depth
    """

    def __init__(
        self,
        num_depths: int = 64,
        min_depth: float = 0.5,
        max_depth: float = 100.0,
        sampling: str = 'log_uniform',
    ):
        super().__init__()
        self.plane_sweeper = PlaneSweeper(
            num_depths, min_depth, max_depth, sampling
        )

    @property
    def depth_planes(self) -> torch.Tensor:
        return self.plane_sweeper.depth_planes

    def forward(
        self,
        feat_ref: torch.Tensor,
        feat_srcs: list,
        K_ref: torch.Tensor,
        K_srcs: list,
        T_srcs_ref: list,
    ) -> torch.Tensor:
        """
        Build aggregated cost volume from multiple source views.

        Args:
            feat_ref: [B, C, H, W] reference feature map
            feat_srcs: list of [B, C, H, W] source feature maps
            K_ref: [B, 3, 3] reference intrinsics
            K_srcs: list of [B, 3, 3] source intrinsics
            T_srcs_ref: list of [B, 4, 4] ref-to-source transforms

        Returns:
            cost_volume: [B, C, D, H, W] aggregated cost volume
        """
        cost_volumes = []
        for feat_src, K_src, T_src_ref in zip(feat_srcs, K_srcs, T_srcs_ref):
            cv = self.plane_sweeper(feat_ref, feat_src, K_ref, K_src, T_src_ref)
            cost_volumes.append(cv)

        # Aggregate: mean variance across source views
        if len(cost_volumes) == 1:
            return cost_volumes[0]

        cost_volume = torch.stack(cost_volumes, dim=0).mean(dim=0)
        return cost_volume


def depth_regression_softargmin(
    cost_volume: torch.Tensor,
    depth_planes: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable depth regression via soft argmin.

    Converts the cost volume into a depth probability distribution
    and computes the expected depth (soft argmin).

    Args:
        cost_volume: [B, C, D, H, W] or [B, 1, D, H, W] cost volume
        depth_planes: [D] depth hypothesis values
        temperature: softmax temperature (lower = sharper)

    Returns:
        depth: [B, 1, H, W] regressed depth map
        prob: [B, D, H, W] depth probability distribution
    """
    # Average over feature channels if needed
    if cost_volume.shape[1] > 1:
        cost_volume = cost_volume.mean(dim=1, keepdim=True)  # [B, 1, D, H, W]

    cost_volume = cost_volume.squeeze(1)  # [B, D, H, W]

    # Convert cost to probability: low cost = high probability
    # Negate cost and apply softmax along depth dimension
    prob = F.softmax(-cost_volume / temperature, dim=1)  # [B, D, H, W]

    # Expected depth (soft argmin)
    D = depth_planes.shape[0]
    depth_planes_view = depth_planes.reshape(1, D, 1, 1)  # broadcastable
    depth = (prob * depth_planes_view).sum(dim=1, keepdim=True)  # [B, 1, H, W]

    return depth, prob
