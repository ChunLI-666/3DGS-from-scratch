"""
Gaussian model implementations for 3DGS
"""

from .gaussian_model import Gaussian3D, GaussianModel
from .covariance import build_covariance_matrix, quaternion_to_rotation_matrix
from .projection import project_gaussian, compute_projection_jacobian

__all__ = [
    'Gaussian3D',
    'GaussianModel',
    'build_covariance_matrix',
    'quaternion_to_rotation_matrix',
    'project_gaussian',
    'compute_projection_jacobian',
]
