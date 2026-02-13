"""
Feed-forward Gaussian Splatting Module.

This module provides educational implementations of feed-forward
3D Gaussian Splatting methods (MVSplat, pixelSplat, etc.).

Key components:
- CostVolumeBuilder: Plane sweeping cost volume construction
- PixelAlignedGaussians: Pixel-aligned Gaussian representation
- GaussianPredictor: Neural network heads for Gaussian prediction
- CrossViewInteraction: Cross-view feature aggregation
"""

from .cost_volume import (
    CostVolumeBuilder,
    PlaneSweeper,
    depth_regression_softargmin,
)
from .pixel_aligned import (
    PixelAlignedGaussians,
    unproject_depth_to_3d,
    create_pixel_grid,
)
from .gaussian_predictor import (
    GaussianPredictionHeads,
    DepthHead,
    CovarianceHead,
    OpacityHead,
)

__all__ = [
    'CostVolumeBuilder',
    'PlaneSweeper',
    'depth_regression_softargmin',
    'PixelAlignedGaussians',
    'unproject_depth_to_3d',
    'create_pixel_grid',
    'GaussianPredictionHeads',
    'DepthHead',
    'CovarianceHead',
    'OpacityHead',
]
