"""
Simple Rasterizer for 3D Gaussian Splatting

This module provides an educational, pure-Python/PyTorch implementation
of Gaussian splatting rasterization. It is designed for understanding,
not performance.

The rasterizer takes projected 2D Gaussians and renders them to an image
using proper depth ordering and alpha blending.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Gaussian2D:
    """
    A 2D Gaussian for rendering.

    Attributes:
        mean: 2D center in image coordinates [2]
        covariance: 2D covariance matrix [2, 2]
        depth: Distance from camera (for sorting)
        opacity: Base opacity (0-1)
        color: RGB color [3]
    """
    mean: torch.Tensor
    covariance: torch.Tensor
    depth: float
    opacity: float
    color: torch.Tensor


class SimpleRasterizer:
    """
    Simple tile-based Gaussian rasterizer.

    This is a pure PyTorch implementation for educational purposes.
    For production use, see the CUDA-accelerated diff-gaussian-rasterization.

    The rasterizer:
    1. Sorts Gaussians by depth (front-to-back)
    2. For each pixel, evaluates all Gaussians
    3. Applies alpha blending with proper transmittance

    Args:
        image_width: Output image width
        image_height: Output image height
        tile_size: Tile size for tile-based rendering (default: 16)
        background: Background color [3] (default: white)
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        tile_size: int = 16,
        background: Optional[torch.Tensor] = None,
    ):
        self.width = image_width
        self.height = image_height
        self.tile_size = tile_size

        if background is None:
            self.background = torch.ones(3)
        else:
            self.background = background

        # Precompute pixel coordinates
        self._create_pixel_grid()

    def _create_pixel_grid(self):
        """Create meshgrid of pixel coordinates."""
        y = torch.arange(self.height, dtype=torch.float32)
        x = torch.arange(self.width, dtype=torch.float32)
        self.y_grid, self.x_grid = torch.meshgrid(y, x, indexing='ij')

    def to(self, device: torch.device) -> 'SimpleRasterizer':
        """Move rasterizer to device."""
        self.background = self.background.to(device)
        self.x_grid = self.x_grid.to(device)
        self.y_grid = self.y_grid.to(device)
        return self

    def _compute_gaussian_bbox(
        self,
        gaussian: Gaussian2D,
        n_sigma: float = 3.0,
    ) -> Tuple[int, int, int, int]:
        """
        Compute bounding box for a 2D Gaussian.

        Returns:
            Tuple of (x_min, y_min, x_max, y_max) in pixel coordinates
        """
        # Get eigenvalues for extent
        eigenvalues = torch.linalg.eigvalsh(gaussian.covariance)
        max_radius = n_sigma * torch.sqrt(eigenvalues.max()).item()

        # Compute bounds
        cx, cy = gaussian.mean[0].item(), gaussian.mean[1].item()
        x_min = max(0, int(cx - max_radius))
        x_max = min(self.width, int(cx + max_radius) + 1)
        y_min = max(0, int(cy - max_radius))
        y_max = min(self.height, int(cy + max_radius) + 1)

        return x_min, y_min, x_max, y_max

    def _evaluate_gaussian(
        self,
        gaussian: Gaussian2D,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate 2D Gaussian at given coordinates.

        Args:
            gaussian: Gaussian2D to evaluate
            x, y: Coordinate grids

        Returns:
            Gaussian values (0 to 1) at each coordinate
        """
        # Compute inverse covariance
        cov_inv = torch.linalg.inv(gaussian.covariance)

        # Offset from mean
        dx = x - gaussian.mean[0]
        dy = y - gaussian.mean[1]

        # Mahalanobis distance: d = [dx, dy]^T @ cov_inv @ [dx, dy]
        mahal = (
            cov_inv[0, 0] * dx * dx +
            (cov_inv[0, 1] + cov_inv[1, 0]) * dx * dy +
            cov_inv[1, 1] * dy * dy
        )

        return torch.exp(-0.5 * mahal)

    def render_naive(
        self,
        gaussians: List[Gaussian2D],
    ) -> torch.Tensor:
        """
        Naive rendering: evaluate all Gaussians at all pixels.

        This is O(N * W * H) where N is number of Gaussians.
        Simple but slow for many Gaussians.

        Args:
            gaussians: List of Gaussian2D to render

        Returns:
            Rendered image [H, W, 3]
        """
        device = self.x_grid.device

        # Sort by depth (front to back)
        sorted_gaussians = sorted(gaussians, key=lambda g: g.depth)

        # Initialize output
        image = torch.zeros(self.height, self.width, 3, device=device)
        transmittance = torch.ones(self.height, self.width, device=device)

        for gaussian in sorted_gaussians:
            # Evaluate Gaussian at all pixels
            gaussian_value = self._evaluate_gaussian(
                gaussian, self.x_grid, self.y_grid
            )

            # Compute alpha
            alpha = gaussian_value * gaussian.opacity

            # Alpha blending: C = C + T * alpha * color
            # where T is transmittance (product of (1 - alpha) for previous Gaussians)
            weight = transmittance * alpha

            for c in range(3):
                image[:, :, c] += weight * gaussian.color[c]

            # Update transmittance: T = T * (1 - alpha)
            transmittance = transmittance * (1 - alpha)

        # Add background
        for c in range(3):
            image[:, :, c] += transmittance * self.background[c]

        return torch.clamp(image, 0, 1)

    def render_with_bbox(
        self,
        gaussians: List[Gaussian2D],
        n_sigma: float = 3.0,
    ) -> torch.Tensor:
        """
        Render with bounding box culling.

        Only evaluates Gaussians within their bounding boxes.
        More efficient than naive rendering.

        Args:
            gaussians: List of Gaussian2D to render
            n_sigma: Number of standard deviations for bounding box

        Returns:
            Rendered image [H, W, 3]
        """
        device = self.x_grid.device

        # Sort by depth (front to back)
        sorted_gaussians = sorted(gaussians, key=lambda g: g.depth)

        # Initialize output
        image = torch.zeros(self.height, self.width, 3, device=device)
        transmittance = torch.ones(self.height, self.width, device=device)

        for gaussian in sorted_gaussians:
            # Get bounding box
            x_min, y_min, x_max, y_max = self._compute_gaussian_bbox(
                gaussian, n_sigma
            )

            # Skip if completely outside image
            if x_min >= x_max or y_min >= y_max:
                continue

            # Extract region
            x_region = self.x_grid[y_min:y_max, x_min:x_max]
            y_region = self.y_grid[y_min:y_max, x_min:x_max]

            # Evaluate Gaussian in region
            gaussian_value = self._evaluate_gaussian(gaussian, x_region, y_region)

            # Compute alpha
            alpha = gaussian_value * gaussian.opacity

            # Get transmittance in region
            T_region = transmittance[y_min:y_max, x_min:x_max]
            weight = T_region * alpha

            # Blend color
            for c in range(3):
                image[y_min:y_max, x_min:x_max, c] += weight * gaussian.color[c]

            # Update transmittance
            transmittance[y_min:y_max, x_min:x_max] = T_region * (1 - alpha)

        # Add background
        for c in range(3):
            image[:, :, c] += transmittance * self.background[c]

        return torch.clamp(image, 0, 1)

    def render(
        self,
        gaussians: List[Gaussian2D],
        method: str = 'bbox',
    ) -> torch.Tensor:
        """
        Render Gaussians to image.

        Args:
            gaussians: List of Gaussian2D to render
            method: Rendering method ('naive' or 'bbox')

        Returns:
            Rendered image [H, W, 3]
        """
        if len(gaussians) == 0:
            # Return background
            image = self.background.unsqueeze(0).unsqueeze(0)
            return image.expand(self.height, self.width, 3).clone()

        if method == 'naive':
            return self.render_naive(gaussians)
        elif method == 'bbox':
            return self.render_with_bbox(gaussians)
        else:
            raise ValueError(f"Unknown method: {method}")


class TileBasedRasterizer(SimpleRasterizer):
    """
    Tile-based rasterizer for improved performance.

    Divides the image into tiles and processes each tile independently.
    For each tile, only evaluates Gaussians that overlap with the tile.

    This is closer to the actual 3DGS CUDA implementation.
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        tile_size: int = 16,
        background: Optional[torch.Tensor] = None,
    ):
        super().__init__(image_width, image_height, tile_size, background)

        # Compute number of tiles
        self.n_tiles_x = (image_width + tile_size - 1) // tile_size
        self.n_tiles_y = (image_height + tile_size - 1) // tile_size

    def _get_tile_bounds(
        self,
        tile_x: int,
        tile_y: int,
    ) -> Tuple[int, int, int, int]:
        """Get pixel bounds for a tile."""
        x_min = tile_x * self.tile_size
        y_min = tile_y * self.tile_size
        x_max = min(x_min + self.tile_size, self.width)
        y_max = min(y_min + self.tile_size, self.height)
        return x_min, y_min, x_max, y_max

    def _gaussian_overlaps_tile(
        self,
        gaussian: Gaussian2D,
        tile_bounds: Tuple[int, int, int, int],
        n_sigma: float = 3.0,
    ) -> bool:
        """Check if Gaussian overlaps with tile."""
        tx_min, ty_min, tx_max, ty_max = tile_bounds
        gx_min, gy_min, gx_max, gy_max = self._compute_gaussian_bbox(
            gaussian, n_sigma
        )

        # Check for intersection
        return not (gx_max <= tx_min or gx_min >= tx_max or
                   gy_max <= ty_min or gy_min >= ty_max)

    def render_tiled(
        self,
        gaussians: List[Gaussian2D],
        n_sigma: float = 3.0,
    ) -> torch.Tensor:
        """
        Tile-based rendering.

        Args:
            gaussians: List of Gaussian2D to render
            n_sigma: Number of standard deviations for culling

        Returns:
            Rendered image [H, W, 3]
        """
        device = self.x_grid.device

        # Sort by depth (front to back)
        sorted_gaussians = sorted(gaussians, key=lambda g: g.depth)

        # Initialize output
        image = torch.zeros(self.height, self.width, 3, device=device)
        transmittance = torch.ones(self.height, self.width, device=device)

        # Process each tile
        for ty in range(self.n_tiles_y):
            for tx in range(self.n_tiles_x):
                tile_bounds = self._get_tile_bounds(tx, ty)
                x_min, y_min, x_max, y_max = tile_bounds

                # Get coordinate grids for this tile
                x_tile = self.x_grid[y_min:y_max, x_min:x_max]
                y_tile = self.y_grid[y_min:y_max, x_min:x_max]

                # Get transmittance for this tile
                T_tile = transmittance[y_min:y_max, x_min:x_max]

                # Find Gaussians that overlap this tile
                for gaussian in sorted_gaussians:
                    if not self._gaussian_overlaps_tile(
                        gaussian, tile_bounds, n_sigma
                    ):
                        continue

                    # Evaluate Gaussian in tile
                    gaussian_value = self._evaluate_gaussian(
                        gaussian, x_tile, y_tile
                    )

                    # Compute alpha and blend
                    alpha = gaussian_value * gaussian.opacity
                    weight = T_tile * alpha

                    for c in range(3):
                        image[y_min:y_max, x_min:x_max, c] += (
                            weight * gaussian.color[c]
                        )

                    # Update transmittance
                    T_tile = T_tile * (1 - alpha)

                # Store updated transmittance
                transmittance[y_min:y_max, x_min:x_max] = T_tile

        # Add background
        for c in range(3):
            image[:, :, c] += transmittance * self.background[c]

        return torch.clamp(image, 0, 1)

    def render(
        self,
        gaussians: List[Gaussian2D],
        method: str = 'tiled',
    ) -> torch.Tensor:
        """
        Render Gaussians to image.

        Args:
            gaussians: List of Gaussian2D to render
            method: Rendering method ('naive', 'bbox', or 'tiled')

        Returns:
            Rendered image [H, W, 3]
        """
        if len(gaussians) == 0:
            image = self.background.unsqueeze(0).unsqueeze(0)
            return image.expand(self.height, self.width, 3).clone()

        if method == 'tiled':
            return self.render_tiled(gaussians)
        else:
            return super().render(gaussians, method)


def create_test_gaussians(n_gaussians: int = 10) -> List[Gaussian2D]:
    """
    Create random test Gaussians for visualization.

    Args:
        n_gaussians: Number of Gaussians to create

    Returns:
        List of Gaussian2D
    """
    import numpy as np

    gaussians = []
    for _ in range(n_gaussians):
        # Random position
        mean = torch.tensor([
            np.random.uniform(100, 540),
            np.random.uniform(100, 380),
        ])

        # Random covariance
        angle = np.random.uniform(0, np.pi)
        sx = np.random.uniform(10, 50)
        sy = np.random.uniform(10, 50)

        c, s = np.cos(angle), np.sin(angle)
        R = torch.tensor([[c, -s], [s, c]])
        S = torch.diag(torch.tensor([sx, sy]))
        cov = R @ S @ S.T @ R.T

        # Random properties
        depth = np.random.uniform(1, 10)
        opacity = np.random.uniform(0.5, 1.0)
        color = torch.tensor([
            np.random.uniform(0, 1),
            np.random.uniform(0, 1),
            np.random.uniform(0, 1),
        ])

        gaussians.append(Gaussian2D(
            mean=mean,
            covariance=cov,
            depth=depth,
            opacity=opacity,
            color=color,
        ))

    return gaussians
