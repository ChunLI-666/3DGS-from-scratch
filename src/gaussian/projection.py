"""
Projection Utilities for 3D Gaussian Splatting

This module provides functions to project 3D Gaussians to 2D image space.
"""

import torch
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from .covariance import build_covariance_matrix, project_covariance_2d


@dataclass
class Camera:
    """
    Camera model for projection.

    Attributes:
        fx, fy: Focal lengths
        cx, cy: Principal point
        width, height: Image dimensions
        R: Rotation matrix [3, 3] (world to camera)
        t: Translation vector [3] (world to camera)
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    R: torch.Tensor
    t: torch.Tensor

    @property
    def K(self) -> torch.Tensor:
        """Get intrinsic matrix [3, 3]."""
        K = torch.tensor([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=self.R.dtype, device=self.R.device)
        return K

    @property
    def extrinsic(self) -> torch.Tensor:
        """Get extrinsic matrix [4, 4]."""
        T = torch.eye(4, dtype=self.R.dtype, device=self.R.device)
        T[:3, :3] = self.R
        T[:3, 3] = self.t
        return T

    def transform_point(self, point_world: torch.Tensor) -> torch.Tensor:
        """Transform point from world to camera coordinates."""
        return self.R @ point_world + self.t

    def project_point(self, point_world: torch.Tensor) -> torch.Tensor:
        """Project 3D world point to 2D image coordinates."""
        point_cam = self.transform_point(point_world)
        x = point_cam[0] / point_cam[2]
        y = point_cam[1] / point_cam[2]
        u = self.fx * x + self.cx
        v = self.fy * y + self.cy
        return torch.stack([u, v])


@dataclass
class Gaussian2D:
    """
    A 2D Gaussian resulting from projection.

    Attributes:
        mean: 2D center [u, v]
        covariance: 2D covariance matrix [2, 2]
        depth: Distance from camera
        opacity: Opacity value
        color: RGB color [3]
    """
    mean: torch.Tensor
    covariance: torch.Tensor
    depth: float
    opacity: float
    color: torch.Tensor


def compute_projection_jacobian(
    point_cam: torch.Tensor,
    fx: float,
    fy: float,
) -> torch.Tensor:
    """
    Compute the Jacobian of the projection function.

    The projection function is:
        u = fx * x/z + cx
        v = fy * y/z + cy

    The Jacobian is:
        J = [[fx/z, 0, -fx*x/z^2],
             [0, fy/z, -fy*y/z^2]]

    Args:
        point_cam: Point in camera coordinates [3] or [N, 3]
        fx, fy: Focal lengths

    Returns:
        Jacobian matrix [2, 3] or [N, 2, 3]
    """
    if point_cam.dim() == 1:
        point_cam = point_cam.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    x, y, z = point_cam[:, 0], point_cam[:, 1], point_cam[:, 2]
    z_sq = z * z

    J = torch.zeros(point_cam.shape[0], 2, 3, device=point_cam.device, dtype=point_cam.dtype)
    J[:, 0, 0] = fx / z
    J[:, 0, 2] = -fx * x / z_sq
    J[:, 1, 1] = fy / z
    J[:, 1, 2] = -fy * y / z_sq

    if squeeze_output:
        J = J.squeeze(0)

    return J


def project_gaussian(
    mean_3d: torch.Tensor,
    covariance_3d: torch.Tensor,
    opacity: float,
    color: torch.Tensor,
    camera: Camera,
) -> Optional[Gaussian2D]:
    """
    Project a 3D Gaussian to 2D image space.

    Args:
        mean_3d: 3D center position [3]
        covariance_3d: 3D covariance matrix [3, 3]
        opacity: Opacity value
        color: RGB color [3]
        camera: Camera parameters

    Returns:
        Gaussian2D or None if behind camera
    """
    # Transform to camera coordinates
    mean_cam = camera.transform_point(mean_3d)

    # Check if in front of camera
    if mean_cam[2] <= 0.1:
        return None

    # Project center to image
    mean_2d = camera.project_point(mean_3d)

    # Check if in image bounds (with margin)
    margin = 100
    if (mean_2d[0] < -margin or mean_2d[0] > camera.width + margin or
        mean_2d[1] < -margin or mean_2d[1] > camera.height + margin):
        return None

    # Compute projection Jacobian
    J = compute_projection_jacobian(mean_cam, camera.fx, camera.fy)

    # Project covariance
    # First transform covariance to camera space
    cov_cam = camera.R @ covariance_3d @ camera.R.T

    # Then project to 2D using Jacobian
    cov_2d = J @ cov_cam @ J.T

    # Add small value to diagonal for numerical stability
    cov_2d = cov_2d + torch.eye(2, device=cov_2d.device) * 0.3

    return Gaussian2D(
        mean=mean_2d,
        covariance=cov_2d,
        depth=mean_cam[2].item(),
        opacity=opacity,
        color=color,
    )


def compute_gaussian_2d_extent(
    covariance_2d: torch.Tensor,
    n_sigma: float = 3.0,
) -> Tuple[float, float]:
    """
    Compute the extent (bounding box) of a 2D Gaussian.

    Args:
        covariance_2d: 2D covariance matrix [2, 2]
        n_sigma: Number of standard deviations for extent

    Returns:
        Tuple of (width, height) of bounding box
    """
    # Eigendecomposition to get principal axes
    eigenvalues, _ = torch.linalg.eigh(covariance_2d)

    # Extent is n_sigma times the standard deviation
    extent = n_sigma * torch.sqrt(torch.clamp(eigenvalues, min=1e-8))

    return extent[1].item(), extent[0].item()  # width, height


def evaluate_gaussian_2d(
    x: torch.Tensor,
    y: torch.Tensor,
    mean: torch.Tensor,
    covariance: torch.Tensor,
) -> torch.Tensor:
    """
    Evaluate a 2D Gaussian at given coordinates.

    Args:
        x, y: Coordinates (can be meshgrid)
        mean: Gaussian center [2]
        covariance: Covariance matrix [2, 2]

    Returns:
        Gaussian values at coordinates
    """
    # Compute inverse covariance
    cov_inv = torch.linalg.inv(covariance)

    # Compute offset from mean
    dx = x - mean[0]
    dy = y - mean[1]

    # Compute Mahalanobis distance
    # d = [dx, dy]^T @ cov_inv @ [dx, dy]
    mahal = (
        cov_inv[0, 0] * dx * dx +
        (cov_inv[0, 1] + cov_inv[1, 0]) * dx * dy +
        cov_inv[1, 1] * dy * dy
    )

    # Gaussian value
    return torch.exp(-0.5 * mahal)
