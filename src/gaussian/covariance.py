"""
Covariance Matrix Utilities

This module provides functions to build covariance matrices from
scaling and rotation parameters.
"""

import torch
import numpy as np
from typing import Union


def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    Convert a quaternion to a rotation matrix.

    The quaternion format is [w, x, y, z] where w is the scalar component.

    Args:
        q: Quaternion [4] or batch of quaternions [N, 4]

    Returns:
        Rotation matrix [3, 3] or [N, 3, 3]
    """
    # Handle single quaternion
    if q.dim() == 1:
        q = q.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # Normalize quaternion
    q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)

    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # Build rotation matrix
    # Reference: https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/
    R = torch.zeros(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)

    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)

    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)

    if squeeze_output:
        R = R.squeeze(0)

    return R


def rotation_matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    Convert a rotation matrix to a quaternion.

    Args:
        R: Rotation matrix [3, 3] or [N, 3, 3]

    Returns:
        Quaternion [4] or [N, 4] in [w, x, y, z] format
    """
    if R.dim() == 2:
        R = R.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = R.shape[0]
    q = torch.zeros(batch_size, 4, device=R.device, dtype=R.dtype)

    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]

    # Case 1: trace > 0
    mask1 = trace > 0
    s1 = torch.sqrt(trace[mask1] + 1.0) * 2  # s = 4 * qw
    q[mask1, 0] = 0.25 * s1
    q[mask1, 1] = (R[mask1, 2, 1] - R[mask1, 1, 2]) / s1
    q[mask1, 2] = (R[mask1, 0, 2] - R[mask1, 2, 0]) / s1
    q[mask1, 3] = (R[mask1, 1, 0] - R[mask1, 0, 1]) / s1

    # Case 2: R[0,0] is largest diagonal
    mask2 = ~mask1 & (R[:, 0, 0] > R[:, 1, 1]) & (R[:, 0, 0] > R[:, 2, 2])
    s2 = torch.sqrt(1.0 + R[mask2, 0, 0] - R[mask2, 1, 1] - R[mask2, 2, 2]) * 2
    q[mask2, 0] = (R[mask2, 2, 1] - R[mask2, 1, 2]) / s2
    q[mask2, 1] = 0.25 * s2
    q[mask2, 2] = (R[mask2, 0, 1] + R[mask2, 1, 0]) / s2
    q[mask2, 3] = (R[mask2, 0, 2] + R[mask2, 2, 0]) / s2

    # Case 3: R[1,1] is largest diagonal
    mask3 = ~mask1 & ~mask2 & (R[:, 1, 1] > R[:, 2, 2])
    s3 = torch.sqrt(1.0 + R[mask3, 1, 1] - R[mask3, 0, 0] - R[mask3, 2, 2]) * 2
    q[mask3, 0] = (R[mask3, 0, 2] - R[mask3, 2, 0]) / s3
    q[mask3, 1] = (R[mask3, 0, 1] + R[mask3, 1, 0]) / s3
    q[mask3, 2] = 0.25 * s3
    q[mask3, 3] = (R[mask3, 1, 2] + R[mask3, 2, 1]) / s3

    # Case 4: R[2,2] is largest diagonal
    mask4 = ~mask1 & ~mask2 & ~mask3
    s4 = torch.sqrt(1.0 + R[mask4, 2, 2] - R[mask4, 0, 0] - R[mask4, 1, 1]) * 2
    q[mask4, 0] = (R[mask4, 1, 0] - R[mask4, 0, 1]) / s4
    q[mask4, 1] = (R[mask4, 0, 2] + R[mask4, 2, 0]) / s4
    q[mask4, 2] = (R[mask4, 1, 2] + R[mask4, 2, 1]) / s4
    q[mask4, 3] = 0.25 * s4

    if squeeze_output:
        q = q.squeeze(0)

    return q


def build_covariance_matrix(
    scaling: torch.Tensor,
    rotation: torch.Tensor,
) -> torch.Tensor:
    """
    Build a 3D covariance matrix from scaling and rotation.

    The covariance is computed as: Σ = R @ S @ S.T @ R.T
    where S = diag(scaling) and R is the rotation matrix.

    Args:
        scaling: Scale factors [3] or [N, 3]
        rotation: Quaternion [4] or [N, 4] in [w, x, y, z] format

    Returns:
        Covariance matrix [3, 3] or [N, 3, 3]
    """
    # Handle single input
    if scaling.dim() == 1:
        scaling = scaling.unsqueeze(0)
        rotation = rotation.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size = scaling.shape[0]

    # Get rotation matrix
    R = quaternion_to_rotation_matrix(rotation)  # [N, 3, 3]

    # Build scaling matrix
    S = torch.zeros(batch_size, 3, 3, device=scaling.device, dtype=scaling.dtype)
    S[:, 0, 0] = scaling[:, 0]
    S[:, 1, 1] = scaling[:, 1]
    S[:, 2, 2] = scaling[:, 2]

    # Compute covariance: Σ = R @ S @ S.T @ R.T
    # Since S is diagonal, S @ S.T = S^2
    S_squared = S @ S.transpose(-1, -2)
    covariance = R @ S_squared @ R.transpose(-1, -2)

    if squeeze_output:
        covariance = covariance.squeeze(0)

    return covariance


def decompose_covariance(covariance: torch.Tensor) -> tuple:
    """
    Decompose a covariance matrix into scaling and rotation.

    Uses eigendecomposition: Σ = V @ Λ @ V.T
    where scaling = sqrt(Λ) and rotation = V

    Args:
        covariance: Covariance matrix [3, 3] or [N, 3, 3]

    Returns:
        Tuple of (scaling [3], rotation quaternion [4])
        or ([N, 3], [N, 4]) for batched input
    """
    if covariance.dim() == 2:
        covariance = covariance.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)

    # Scaling is sqrt of eigenvalues
    scaling = torch.sqrt(torch.clamp(eigenvalues, min=1e-8))

    # Rotation matrix from eigenvectors
    # Ensure proper rotation (det = 1)
    det = torch.linalg.det(eigenvectors)
    eigenvectors = eigenvectors * det.unsqueeze(-1).unsqueeze(-1).sign()

    rotation = rotation_matrix_to_quaternion(eigenvectors)

    if squeeze_output:
        scaling = scaling.squeeze(0)
        rotation = rotation.squeeze(0)

    return scaling, rotation


def project_covariance_2d(
    covariance_3d: torch.Tensor,
    jacobian: torch.Tensor,
    view_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Project a 3D covariance matrix to 2D image space.

    The projection formula is: Σ' = J @ W @ Σ @ W.T @ J.T
    where W is the view transformation and J is the projection Jacobian.

    Args:
        covariance_3d: 3D covariance [3, 3] or [N, 3, 3]
        jacobian: Projection Jacobian [2, 3] or [N, 2, 3]
        view_matrix: View transformation [3, 3] or [N, 3, 3]

    Returns:
        2D covariance [2, 2] or [N, 2, 2]
    """
    if covariance_3d.dim() == 2:
        covariance_3d = covariance_3d.unsqueeze(0)
        jacobian = jacobian.unsqueeze(0)
        view_matrix = view_matrix.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    # Transform covariance to camera space
    cov_view = view_matrix @ covariance_3d @ view_matrix.transpose(-1, -2)

    # Project to 2D
    cov_2d = jacobian @ cov_view @ jacobian.transpose(-1, -2)

    if squeeze_output:
        cov_2d = cov_2d.squeeze(0)

    return cov_2d
