"""
3D Gaussian Model Implementation

This module provides educational implementations of 3D Gaussian primitives
used in 3D Gaussian Splatting.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple

from .covariance import build_covariance_matrix, quaternion_to_rotation_matrix


class Gaussian3D:
    """
    A single 3D Gaussian primitive.

    Each 3D Gaussian is parameterized by:
    - mean (μ): 3D position [x, y, z]
    - scaling (s): Scale factors [sx, sy, sz]
    - rotation (q): Rotation quaternion [w, x, y, z]
    - opacity (α): Opacity value [0, 1]
    - sh_coeffs: Spherical harmonics coefficients for view-dependent color

    The covariance matrix is computed as: Σ = R @ S @ S.T @ R.T
    where R is the rotation matrix and S = diag(sx, sy, sz)
    """

    def __init__(
        self,
        mean: torch.Tensor,
        scaling: torch.Tensor,
        rotation: torch.Tensor,
        opacity: float = 1.0,
        color: Optional[torch.Tensor] = None,
        sh_coeffs: Optional[torch.Tensor] = None,
    ):
        """
        Initialize a 3D Gaussian.

        Args:
            mean: Center position [3]
            scaling: Scale factors [3]
            rotation: Rotation quaternion [4] (w, x, y, z)
            opacity: Opacity value (default: 1.0)
            color: RGB color [3] (optional, used if sh_coeffs not provided)
            sh_coeffs: Spherical harmonics coefficients [n_coeffs, 3]
        """
        self.mean = mean
        self.scaling = scaling
        self.rotation = rotation / (rotation.norm() + 1e-8)  # Normalize quaternion
        self.opacity = opacity

        if sh_coeffs is not None:
            self.sh_coeffs = sh_coeffs
        elif color is not None:
            # Convert color to DC component of SH
            self.sh_coeffs = color.unsqueeze(0)  # [1, 3]
        else:
            self.sh_coeffs = torch.ones(1, 3) * 0.5  # Default gray

    @property
    def device(self):
        return self.mean.device

    def get_covariance(self) -> torch.Tensor:
        """
        Compute the 3D covariance matrix.

        Returns:
            Covariance matrix [3, 3]
        """
        return build_covariance_matrix(self.scaling, self.rotation)

    def get_rotation_matrix(self) -> torch.Tensor:
        """
        Get the rotation matrix from quaternion.

        Returns:
            Rotation matrix [3, 3]
        """
        return quaternion_to_rotation_matrix(self.rotation)

    def to(self, device):
        """Move Gaussian to device."""
        self.mean = self.mean.to(device)
        self.scaling = self.scaling.to(device)
        self.rotation = self.rotation.to(device)
        self.sh_coeffs = self.sh_coeffs.to(device)
        return self

    def clone(self) -> 'Gaussian3D':
        """Create a copy of this Gaussian."""
        return Gaussian3D(
            mean=self.mean.clone(),
            scaling=self.scaling.clone(),
            rotation=self.rotation.clone(),
            opacity=self.opacity,
            sh_coeffs=self.sh_coeffs.clone(),
        )

    def __repr__(self):
        return (
            f"Gaussian3D(\n"
            f"  mean={self.mean.tolist()},\n"
            f"  scaling={self.scaling.tolist()},\n"
            f"  rotation={self.rotation.tolist()},\n"
            f"  opacity={self.opacity}\n"
            f")"
        )


class GaussianModel(nn.Module):
    """
    A collection of 3D Gaussians representing a scene.

    This is a simplified educational version of the GaussianModel
    from the official 3DGS implementation.
    """

    def __init__(self, sh_degree: int = 0):
        """
        Initialize the Gaussian model.

        Args:
            sh_degree: Maximum spherical harmonics degree (0-3)
        """
        super().__init__()
        self.sh_degree = sh_degree
        self.active_sh_degree = 0

        # Number of SH coefficients per channel
        self.n_sh_coeffs = (sh_degree + 1) ** 2

        # Learnable parameters (initialized empty)
        self._xyz = nn.Parameter(torch.empty(0, 3))
        self._scaling = nn.Parameter(torch.empty(0, 3))
        self._rotation = nn.Parameter(torch.empty(0, 4))
        self._opacity = nn.Parameter(torch.empty(0, 1))
        self._features_dc = nn.Parameter(torch.empty(0, 1, 3))
        self._features_rest = nn.Parameter(torch.empty(0, self.n_sh_coeffs - 1, 3))

    @property
    def num_gaussians(self) -> int:
        """Get the number of Gaussians."""
        return self._xyz.shape[0]

    @property
    def xyz(self) -> torch.Tensor:
        """Get positions."""
        return self._xyz

    @property
    def scaling(self) -> torch.Tensor:
        """Get scaling factors (activated)."""
        return torch.exp(self._scaling)

    @property
    def rotation(self) -> torch.Tensor:
        """Get rotation quaternions (normalized)."""
        return torch.nn.functional.normalize(self._rotation, dim=-1)

    @property
    def opacity(self) -> torch.Tensor:
        """Get opacity values (activated)."""
        return torch.sigmoid(self._opacity)

    @property
    def features(self) -> torch.Tensor:
        """Get all SH features."""
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        """
        Compute covariance matrices for all Gaussians.

        Args:
            scaling_modifier: Optional scaling modifier

        Returns:
            Covariance matrices [N, 3, 3]
        """
        scaling = self.scaling * scaling_modifier
        rotation = self.rotation

        # Build covariance matrices
        covariances = []
        for i in range(self.num_gaussians):
            cov = build_covariance_matrix(scaling[i], rotation[i])
            covariances.append(cov)

        return torch.stack(covariances)

    def initialize_from_points(
        self,
        points: torch.Tensor,
        colors: Optional[torch.Tensor] = None,
        initial_scale: float = 0.01,
    ):
        """
        Initialize Gaussians from a point cloud.

        Args:
            points: Point positions [N, 3]
            colors: Point colors [N, 3] (optional)
            initial_scale: Initial scale factor
        """
        n_points = points.shape[0]
        device = points.device

        # Initialize positions
        self._xyz = nn.Parameter(points.clone())

        # Initialize scaling (log space)
        self._scaling = nn.Parameter(
            torch.log(torch.ones(n_points, 3, device=device) * initial_scale)
        )

        # Initialize rotation (identity quaternion)
        self._rotation = nn.Parameter(
            torch.tensor([[1, 0, 0, 0]], device=device, dtype=torch.float32).repeat(n_points, 1)
        )

        # Initialize opacity (inverse sigmoid of 0.5)
        self._opacity = nn.Parameter(torch.zeros(n_points, 1, device=device))

        # Initialize SH features
        if colors is not None:
            # Convert RGB to SH DC component
            dc = (colors - 0.5) / 0.28209479177387814  # SH_C0
            self._features_dc = nn.Parameter(dc.unsqueeze(1))
        else:
            self._features_dc = nn.Parameter(torch.zeros(n_points, 1, 3, device=device))

        self._features_rest = nn.Parameter(
            torch.zeros(n_points, self.n_sh_coeffs - 1, 3, device=device)
        )

    def get_gaussians(self) -> list:
        """
        Get list of Gaussian3D objects.

        Returns:
            List of Gaussian3D instances
        """
        gaussians = []
        for i in range(self.num_gaussians):
            g = Gaussian3D(
                mean=self._xyz[i].detach(),
                scaling=self.scaling[i].detach(),
                rotation=self.rotation[i].detach(),
                opacity=self.opacity[i].item(),
                sh_coeffs=self.features[i].detach(),
            )
            gaussians.append(g)
        return gaussians

    def __len__(self):
        return self.num_gaussians

    def __repr__(self):
        return f"GaussianModel(n_gaussians={self.num_gaussians}, sh_degree={self.sh_degree})"
