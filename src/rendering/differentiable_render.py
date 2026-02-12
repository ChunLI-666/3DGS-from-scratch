"""
Differentiable Renderer for 3D Gaussian Splatting

This module provides a complete differentiable rendering pipeline
that combines:
1. 3D to 2D projection
2. Gaussian splatting
3. Alpha blending

The renderer is fully differentiable, enabling gradient-based optimization
of Gaussian parameters from image supervision.
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass

from .rasterizer import Gaussian2D, SimpleRasterizer, TileBasedRasterizer
from .alpha_blending import compute_transmittance


@dataclass
class RenderOutput:
    """
    Output from the differentiable renderer.

    Attributes:
        image: Rendered RGB image [H, W, 3]
        depth: Depth map [H, W] (optional)
        alpha: Alpha/opacity map [H, W]
        n_visible: Number of visible Gaussians
    """
    image: torch.Tensor
    depth: Optional[torch.Tensor] = None
    alpha: Optional[torch.Tensor] = None
    n_visible: int = 0


class DifferentiableRenderer(nn.Module):
    """
    Fully differentiable Gaussian splatting renderer.

    This renderer:
    1. Projects 3D Gaussians to 2D using camera parameters
    2. Sorts Gaussians by depth
    3. Rasterizes using alpha blending
    4. Supports gradient computation for all Gaussian parameters

    Args:
        image_width: Output image width
        image_height: Output image height
        background: Background color [3]
        near_plane: Near clipping plane
        far_plane: Far clipping plane
    """

    def __init__(
        self,
        image_width: int,
        image_height: int,
        background: Optional[torch.Tensor] = None,
        near_plane: float = 0.1,
        far_plane: float = 100.0,
    ):
        super().__init__()

        self.width = image_width
        self.height = image_height
        self.near_plane = near_plane
        self.far_plane = far_plane

        if background is None:
            self.register_buffer('background', torch.ones(3))
        else:
            self.register_buffer('background', background)

        # Create pixel coordinate grid
        y = torch.arange(image_height, dtype=torch.float32)
        x = torch.arange(image_width, dtype=torch.float32)
        y_grid, x_grid = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('x_grid', x_grid)
        self.register_buffer('y_grid', y_grid)

    def _project_point(
        self,
        point: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project 3D point to 2D.

        Args:
            point: 3D point [3]
            R: Camera rotation [3, 3]
            t: Camera translation [3]
            fx, fy, cx, cy: Camera intrinsics

        Returns:
            Tuple of (2D point [2], depth)
        """
        # World to camera
        point_cam = R @ point + t

        # Perspective projection
        x = point_cam[0] / point_cam[2]
        y = point_cam[1] / point_cam[2]

        u = fx * x + cx
        v = fy * y + cy

        return torch.stack([u, v]), point_cam[2]

    def _compute_jacobian(
        self,
        point_cam: torch.Tensor,
        fx: float,
        fy: float,
    ) -> torch.Tensor:
        """
        Compute projection Jacobian.

        Args:
            point_cam: Point in camera coordinates [3]
            fx, fy: Focal lengths

        Returns:
            Jacobian [2, 3]
        """
        x, y, z = point_cam[0], point_cam[1], point_cam[2]
        z_sq = z * z

        J = torch.zeros(2, 3, device=point_cam.device, dtype=point_cam.dtype)
        J[0, 0] = fx / z
        J[0, 2] = -fx * x / z_sq
        J[1, 1] = fy / z
        J[1, 2] = -fy * y / z_sq

        return J

    def _project_covariance(
        self,
        cov_3d: torch.Tensor,
        R: torch.Tensor,
        J: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project 3D covariance to 2D.

        Args:
            cov_3d: 3D covariance in world space [3, 3]
            R: Camera rotation [3, 3]
            J: Projection Jacobian [2, 3]

        Returns:
            2D covariance [2, 2]
        """
        # Transform to camera space
        cov_cam = R @ cov_3d @ R.T

        # Project to 2D
        cov_2d = J @ cov_cam @ J.T

        # Add small value for numerical stability
        cov_2d = cov_2d + torch.eye(2, device=cov_2d.device) * 0.3

        return cov_2d

    def _evaluate_gaussian_2d(
        self,
        mean: torch.Tensor,
        cov_inv: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate 2D Gaussian at coordinates.

        Args:
            mean: Gaussian center [2]
            cov_inv: Inverse covariance [2, 2]
            x, y: Coordinate grids

        Returns:
            Gaussian values
        """
        dx = x - mean[0]
        dy = y - mean[1]

        mahal = (
            cov_inv[0, 0] * dx * dx +
            (cov_inv[0, 1] + cov_inv[1, 0]) * dx * dy +
            cov_inv[1, 1] * dy * dy
        )

        return torch.exp(-0.5 * mahal)

    def _get_gaussian_bbox(
        self,
        mean: torch.Tensor,
        cov: torch.Tensor,
        n_sigma: float = 3.0,
    ) -> Tuple[int, int, int, int]:
        """
        Compute bounding box for 2D Gaussian.

        Returns:
            (x_min, y_min, x_max, y_max)
        """
        eigenvalues = torch.linalg.eigvalsh(cov)
        max_radius = n_sigma * torch.sqrt(eigenvalues.max()).item()

        cx, cy = mean[0].item(), mean[1].item()
        x_min = max(0, int(cx - max_radius))
        x_max = min(self.width, int(cx + max_radius) + 1)
        y_min = max(0, int(cy - max_radius))
        y_max = min(self.height, int(cy + max_radius) + 1)

        return x_min, y_min, x_max, y_max

    def render(
        self,
        means_3d: torch.Tensor,
        covariances_3d: torch.Tensor,
        opacities: torch.Tensor,
        colors: torch.Tensor,
        R: torch.Tensor,
        t: torch.Tensor,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        return_depth: bool = False,
    ) -> RenderOutput:
        """
        Render Gaussians to image.

        Args:
            means_3d: Gaussian centers [N, 3]
            covariances_3d: 3D covariances [N, 3, 3]
            opacities: Gaussian opacities [N]
            colors: Gaussian colors [N, 3]
            R: Camera rotation [3, 3]
            t: Camera translation [3]
            fx, fy, cx, cy: Camera intrinsics
            return_depth: Whether to compute depth map

        Returns:
            RenderOutput with rendered image and optional depth/alpha
        """
        device = means_3d.device
        N = means_3d.shape[0]

        # Project all Gaussians
        means_2d = []
        covs_2d = []
        depths = []
        visible_mask = []

        for i in range(N):
            # Transform to camera space
            mean_cam = R @ means_3d[i] + t

            # Visibility check
            if mean_cam[2] <= self.near_plane or mean_cam[2] > self.far_plane:
                visible_mask.append(False)
                continue

            # Project center
            mean_2d, depth = self._project_point(
                means_3d[i], R, t, fx, fy, cx, cy
            )

            # Check if in image bounds
            margin = 100
            if (mean_2d[0] < -margin or mean_2d[0] > self.width + margin or
                mean_2d[1] < -margin or mean_2d[1] > self.height + margin):
                visible_mask.append(False)
                continue

            # Project covariance
            J = self._compute_jacobian(mean_cam, fx, fy)
            cov_2d = self._project_covariance(covariances_3d[i], R, J)

            means_2d.append(mean_2d)
            covs_2d.append(cov_2d)
            depths.append(depth)
            visible_mask.append(True)

        # Handle empty case
        n_visible = len(means_2d)
        if n_visible == 0:
            image = self.background.unsqueeze(0).unsqueeze(0)
            image = image.expand(self.height, self.width, 3)
            return RenderOutput(
                image=image,
                depth=torch.zeros(self.height, self.width, device=device) if return_depth else None,
                alpha=torch.zeros(self.height, self.width, device=device),
                n_visible=0,
            )

        # Stack visible Gaussians
        means_2d = torch.stack(means_2d)
        covs_2d = torch.stack(covs_2d)
        depths = torch.stack(depths)

        # Get visible indices
        visible_indices = [i for i, v in enumerate(visible_mask) if v]
        visible_opacities = opacities[visible_indices]
        visible_colors = colors[visible_indices]

        # Sort by depth (front to back)
        depth_order = torch.argsort(depths)
        means_2d = means_2d[depth_order]
        covs_2d = covs_2d[depth_order]
        depths = depths[depth_order]
        visible_opacities = visible_opacities[depth_order]
        visible_colors = visible_colors[depth_order]

        # Render with alpha blending
        image = torch.zeros(self.height, self.width, 3, device=device)
        alpha_map = torch.zeros(self.height, self.width, device=device)
        depth_map = torch.zeros(self.height, self.width, device=device) if return_depth else None
        transmittance = torch.ones(self.height, self.width, device=device)

        for i in range(n_visible):
            mean = means_2d[i]
            cov = covs_2d[i]
            cov_inv = torch.linalg.inv(cov)
            depth = depths[i]
            opacity = visible_opacities[i]
            color = visible_colors[i]

            # Get bounding box
            x_min, y_min, x_max, y_max = self._get_gaussian_bbox(mean, cov)
            if x_min >= x_max or y_min >= y_max:
                continue

            # Get coordinates in region
            x_region = self.x_grid[y_min:y_max, x_min:x_max]
            y_region = self.y_grid[y_min:y_max, x_min:x_max]

            # Evaluate Gaussian
            gaussian_value = self._evaluate_gaussian_2d(
                mean, cov_inv, x_region, y_region
            )

            # Compute alpha
            alpha = gaussian_value * opacity

            # Get transmittance in region
            T_region = transmittance[y_min:y_max, x_min:x_max]
            weight = T_region * alpha

            # Blend color
            for c in range(3):
                image[y_min:y_max, x_min:x_max, c] += weight * color[c]

            # Update alpha map
            alpha_map[y_min:y_max, x_min:x_max] += weight

            # Update depth map
            if return_depth:
                depth_map[y_min:y_max, x_min:x_max] += weight * depth

            # Update transmittance
            transmittance[y_min:y_max, x_min:x_max] = T_region * (1 - alpha)

        # Add background
        for c in range(3):
            image[:, :, c] += transmittance * self.background[c]

        # Normalize depth by alpha
        if return_depth:
            valid_alpha = alpha_map > 1e-6
            depth_map[valid_alpha] /= alpha_map[valid_alpha]

        return RenderOutput(
            image=torch.clamp(image, 0, 1),
            depth=depth_map,
            alpha=alpha_map,
            n_visible=n_visible,
        )

    def forward(
        self,
        gaussian_params: dict,
        camera_params: dict,
        return_depth: bool = False,
    ) -> RenderOutput:
        """
        Forward pass for training.

        Args:
            gaussian_params: Dict with 'means', 'covariances', 'opacities', 'colors'
            camera_params: Dict with 'R', 't', 'fx', 'fy', 'cx', 'cy'
            return_depth: Whether to compute depth map

        Returns:
            RenderOutput
        """
        return self.render(
            means_3d=gaussian_params['means'],
            covariances_3d=gaussian_params['covariances'],
            opacities=gaussian_params['opacities'],
            colors=gaussian_params['colors'],
            R=camera_params['R'],
            t=camera_params['t'],
            fx=camera_params['fx'],
            fy=camera_params['fy'],
            cx=camera_params['cx'],
            cy=camera_params['cy'],
            return_depth=return_depth,
        )


class GaussianRenderer(nn.Module):
    """
    High-level Gaussian renderer with parameter management.

    This class integrates:
    - Gaussian parameter storage (as nn.Parameters)
    - Covariance construction from scale/rotation
    - Differentiable rendering

    Suitable for optimization-based reconstruction.
    """

    def __init__(
        self,
        n_gaussians: int,
        image_width: int,
        image_height: int,
        background: Optional[torch.Tensor] = None,
    ):
        super().__init__()

        self.n_gaussians = n_gaussians
        self.renderer = DifferentiableRenderer(
            image_width, image_height, background
        )

        # Initialize Gaussian parameters
        self.means = nn.Parameter(torch.zeros(n_gaussians, 3))
        self.scales_raw = nn.Parameter(torch.zeros(n_gaussians, 3))  # log scale
        self.rotations_raw = nn.Parameter(
            torch.zeros(n_gaussians, 4)
        )  # quaternion
        self.rotations_raw.data[:, 0] = 1.0  # Identity rotation
        self.opacities_raw = nn.Parameter(torch.zeros(n_gaussians))  # logit
        self.colors = nn.Parameter(torch.ones(n_gaussians, 3) * 0.5)

    @property
    def scales(self) -> torch.Tensor:
        """Get activated scales."""
        return torch.exp(self.scales_raw)

    @property
    def rotations(self) -> torch.Tensor:
        """Get normalized quaternions."""
        return self.rotations_raw / (
            self.rotations_raw.norm(dim=-1, keepdim=True) + 1e-8
        )

    @property
    def opacities(self) -> torch.Tensor:
        """Get activated opacities."""
        return torch.sigmoid(self.opacities_raw)

    def _quaternion_to_rotation_matrix(self, q: torch.Tensor) -> torch.Tensor:
        """Convert quaternion to rotation matrix."""
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)

        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

        N = q.shape[0]
        R = torch.zeros(N, 3, 3, device=q.device, dtype=q.dtype)

        R[:, 0, 0] = 1 - 2*(y*y + z*z)
        R[:, 0, 1] = 2*(x*y - w*z)
        R[:, 0, 2] = 2*(x*z + w*y)
        R[:, 1, 0] = 2*(x*y + w*z)
        R[:, 1, 1] = 1 - 2*(x*x + z*z)
        R[:, 1, 2] = 2*(y*z - w*x)
        R[:, 2, 0] = 2*(x*z - w*y)
        R[:, 2, 1] = 2*(y*z + w*x)
        R[:, 2, 2] = 1 - 2*(x*x + y*y)

        return R

    def get_covariances(self) -> torch.Tensor:
        """Compute 3D covariances from scale and rotation."""
        scales = self.scales
        R = self._quaternion_to_rotation_matrix(self.rotations)

        # Build diagonal scaling matrix
        S = torch.diag_embed(scales)  # [N, 3, 3]

        # Compute covariance: Σ = R @ S @ S.T @ R.T
        S_squared = S @ S.transpose(-1, -2)
        covariances = R @ S_squared @ R.transpose(-1, -2)

        return covariances

    def forward(
        self,
        camera_params: dict,
        return_depth: bool = False,
    ) -> RenderOutput:
        """
        Render current Gaussians.

        Args:
            camera_params: Dict with 'R', 't', 'fx', 'fy', 'cx', 'cy'
            return_depth: Whether to compute depth map

        Returns:
            RenderOutput
        """
        gaussian_params = {
            'means': self.means,
            'covariances': self.get_covariances(),
            'opacities': self.opacities,
            'colors': self.colors,
        }

        return self.renderer(gaussian_params, camera_params, return_depth)

    def initialize_from_points(
        self,
        points: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
        scale: float = 0.1,
    ):
        """
        Initialize Gaussians from 3D points.

        Args:
            points: 3D points [N, 3]
            colors: RGB colors [N, 3] (optional)
            scale: Initial scale for all Gaussians
        """
        assert points.shape[0] == self.n_gaussians

        with torch.no_grad():
            self.means.data = points.clone()
            self.scales_raw.data = torch.log(
                torch.ones_like(points) * scale
            )
            self.rotations_raw.data = torch.zeros(self.n_gaussians, 4)
            self.rotations_raw.data[:, 0] = 1.0
            self.opacities_raw.data = torch.zeros(self.n_gaussians)

            if colors is not None:
                self.colors.data = colors.clone()
