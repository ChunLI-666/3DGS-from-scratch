"""
Gaussian Prediction Heads for Feed-forward 3DGS.

Neural network heads that predict per-pixel Gaussian parameters
from processed features (e.g., from Cost Volume or cross-attention).

Each head predicts one Gaussian property:
- DepthHead: predicts per-pixel depth (positive via Softplus)
- CovarianceHead: predicts scale and rotation (covariance parameters)
- OpacityHead: predicts transparency (sigmoid to [0, 1])

These heads are applied to the output of the feature processing
stage (3D U-Net for MVSplat, Transformer decoder for pixelSplat).

Reference:
- MVSplat (ECCV 2024): https://arxiv.org/abs/2403.14627
"""

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthHead(nn.Module):
    """
    Predicts per-pixel depth from features.

    Two modes:
    - 'regression': Direct depth prediction via convolutions
    - 'cost_volume': Soft argmin from cost volume probability

    The depth is used to back-project pixels to 3D for Gaussian centers.
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 32,
        mode: str = 'regression',
    ):
        super().__init__()
        self.mode = mode

        if mode == 'regression':
            self.layers = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, 1, 1),
                nn.Softplus(),  # Ensures positive depth
            )

    def forward(
        self,
        features: torch.Tensor,
        cost_volume: Optional[torch.Tensor] = None,
        depth_planes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict depth map.

        Args:
            features: [B, C, H, W] processed features
            cost_volume: [B, 1, D, H, W] optional cost volume
            depth_planes: [D] depth hypothesis values

        Returns:
            depth: [B, 1, H, W] predicted depth map
        """
        if self.mode == 'regression':
            return self.layers(features)
        elif self.mode == 'cost_volume':
            assert cost_volume is not None and depth_planes is not None
            from .cost_volume import depth_regression_softargmin
            depth, _ = depth_regression_softargmin(cost_volume, depth_planes)
            return depth
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


class CovarianceHead(nn.Module):
    """
    Predicts per-pixel Gaussian covariance parameters.

    Depending on configuration, predicts:
    - '2d': 2D scales (sx, sy) for screen-space Gaussians
    - '3d': 3D scales (sx, sy, sz) + rotation quaternion (w, x, y, z)

    The 2D mode is simpler and used in some methods; the 3D mode
    gives full control over Gaussian shape in world space.
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 32,
        mode: str = '3d',
        min_scale: float = 1e-4,
    ):
        super().__init__()
        self.mode = mode
        self.min_scale = min_scale

        if mode == '2d':
            # Predict 2D scale only
            self.scale_layers = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, 2, 1),
            )
        elif mode == '3d':
            # Predict 3D scale + rotation quaternion
            self.scale_layers = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, 3, 1),
            )
            self.rotation_layers = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden_channels, 4, 1),
            )
        else:
            raise ValueError(f"Unknown covariance mode: {mode}")

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Predict covariance parameters.

        Args:
            features: [B, C, H, W] processed features

        Returns:
            dict with 'scales' and optionally 'rotations':
                scales: [B, 2 or 3, H, W]
                rotations: [B, 4, H, W] (only for '3d' mode)
        """
        # Predict scales (exp to ensure positive + minimum)
        raw_scales = self.scale_layers(features)
        scales = torch.exp(raw_scales) + self.min_scale

        result = {'scales': scales}

        if self.mode == '3d':
            # Predict rotation quaternion and normalize
            raw_rotation = self.rotation_layers(features)  # [B, 4, H, W]
            rotation = F.normalize(raw_rotation, p=2, dim=1)
            result['rotations'] = rotation

        return result


class OpacityHead(nn.Module):
    """
    Predicts per-pixel Gaussian opacity.

    Output is passed through sigmoid to get values in [0, 1].
    High opacity = opaque surface, low opacity = transparent/uncertain.
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 32,
    ):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict opacity map.

        Args:
            features: [B, C, H, W] processed features

        Returns:
            opacity: [B, 1, H, W] in range [0, 1]
        """
        return torch.sigmoid(self.layers(features))


class GaussianPredictionHeads(nn.Module):
    """
    Combined Gaussian prediction heads.

    Groups depth, covariance, and opacity heads into a single module
    that predicts all Gaussian parameters from features.

    This represents the final stage of both MVSplat and pixelSplat
    architectures, where processed features are converted to
    per-pixel Gaussian parameters.
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 32,
        depth_mode: str = 'regression',
        covariance_mode: str = '3d',
    ):
        super().__init__()
        self.depth_head = DepthHead(in_channels, hidden_channels, depth_mode)
        self.covariance_head = CovarianceHead(
            in_channels, hidden_channels, covariance_mode
        )
        self.opacity_head = OpacityHead(in_channels, hidden_channels)

    def forward(
        self,
        features: torch.Tensor,
        cost_volume: Optional[torch.Tensor] = None,
        depth_planes: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict all Gaussian parameters.

        Args:
            features: [B, C, H, W] processed features
            cost_volume: optional cost volume for depth prediction
            depth_planes: optional depth planes for soft argmin

        Returns:
            dict with keys:
                'depth': [B, 1, H, W]
                'scales': [B, 2 or 3, H, W]
                'rotations': [B, 4, H, W] (if 3D covariance mode)
                'opacities': [B, 1, H, W]
        """
        # Predict depth
        depth = self.depth_head(features, cost_volume, depth_planes)

        # Predict covariance (scale + rotation)
        cov_params = self.covariance_head(features)

        # Predict opacity
        opacity = self.opacity_head(features)

        result = {
            'depth': depth,
            'opacities': opacity,
            **cov_params,
        }

        return result
