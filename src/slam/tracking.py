"""
Camera Tracking for 3DGS-SLAM.

This module provides Gaussian-based camera tracking:
render the Gaussian map from estimated pose, compare with
actual image, and optimize the pose.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any


@dataclass
class TrackingConfig:
    """Configuration for camera tracking."""

    # Optimization
    iterations: int = 40
    learning_rate: float = 0.01

    # Loss weights
    use_depth: bool = True
    depth_weight: float = 0.5
    rgb_weight: float = 1.0

    # Convergence
    min_loss_change: float = 1e-6
    early_stop_patience: int = 5

    # Pose parameterization
    use_lie_algebra: bool = False  # If True, use se(3); else, use matrix


class GaussianTracker:
    """
    Camera tracking using Gaussian map rendering.

    Estimates camera pose by minimizing photometric difference
    between rendered and observed images.

    Attributes:
        gaussians: The Gaussian map to render from
        config: Tracking configuration
        device: Torch device
    """

    def __init__(
        self,
        gaussians: Any,  # GaussianModel
        config: TrackingConfig = None,
        device: str = "cuda",
    ):
        self.gaussians = gaussians
        self.config = config or TrackingConfig()
        self.device = torch.device(device)

        self.last_loss = float('inf')
        self.tracking_history: list = []

    def track(
        self,
        image: torch.Tensor,
        depth: Optional[torch.Tensor],
        initial_pose: torch.Tensor,
        intrinsics: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
        """
        Estimate camera pose from observed image.

        Renders Gaussians from estimated pose and optimizes
        to match the observed image.

        Args:
            image: Observed RGB image [3, H, W] or [H, W, 3]
            depth: Observed depth map [H, W] (optional)
            initial_pose: Initial pose estimate [4, 4]
            intrinsics: Camera intrinsics [3, 3]
            mask: Valid pixel mask [H, W] (optional)

        Returns:
            Tuple of:
            - Optimized pose [4, 4]
            - Final tracking loss
            - Debug info dict
        """
        # Ensure correct format
        if image.dim() == 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)

        image = image.to(self.device)
        if depth is not None:
            depth = depth.to(self.device)
        initial_pose = initial_pose.to(self.device)
        intrinsics = intrinsics.to(self.device)

        H, W = image.shape[1], image.shape[2]

        # Parameterize pose for optimization
        # Using rotation (6D representation) + translation
        pose_params = self._pose_to_params(initial_pose)
        pose_params.requires_grad_(True)

        optimizer = torch.optim.Adam([pose_params], lr=self.config.learning_rate)

        losses = []
        best_loss = float('inf')
        best_pose = initial_pose.clone()
        patience_counter = 0

        for i in range(self.config.iterations):
            optimizer.zero_grad()

            # Reconstruct pose from parameters
            current_pose = self._params_to_pose(pose_params)

            # Render from current pose
            rendered = self._render_from_pose(
                current_pose, intrinsics, H, W
            )

            # Compute loss
            loss, loss_dict = self._compute_tracking_loss(
                rendered, image, depth, mask
            )

            # Backward and update
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Track best
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_pose = current_pose.detach().clone()
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= self.config.early_stop_patience:
                break

            # Convergence check
            if len(losses) > 1:
                if abs(losses[-1] - losses[-2]) < self.config.min_loss_change:
                    break

        self.last_loss = best_loss
        self.tracking_history.append({
            'losses': losses,
            'iterations': len(losses),
            'final_loss': best_loss,
        })

        debug_info = {
            'losses': losses,
            'iterations': len(losses),
            'converged': patience_counter < self.config.early_stop_patience,
        }

        return best_pose, best_loss, debug_info

    def _pose_to_params(self, pose: torch.Tensor) -> torch.Tensor:
        """
        Convert SE(3) pose to optimizable parameters.

        Using 6D rotation representation + 3D translation.

        Args:
            pose: SE(3) matrix [4, 4]

        Returns:
            Parameters tensor [9] (6D rotation + 3D translation)
        """
        R = pose[:3, :3]
        t = pose[:3, 3]

        # 6D rotation: first two columns of R
        r6d = R[:, :2].reshape(-1)  # [6]

        # Combine
        params = torch.cat([r6d, t])  # [9]

        return params

    def _params_to_pose(self, params: torch.Tensor) -> torch.Tensor:
        """
        Convert parameters back to SE(3) pose.

        Args:
            params: Parameters [9]

        Returns:
            SE(3) matrix [4, 4]
        """
        r6d = params[:6].reshape(3, 2)
        t = params[6:9]

        # Gram-Schmidt to get valid rotation
        a1 = r6d[:, 0]
        a2 = r6d[:, 1]

        b1 = F.normalize(a1, dim=0)
        b2 = a2 - (b1 @ a2) * b1
        b2 = F.normalize(b2, dim=0)
        b3 = torch.cross(b1, b2)

        R = torch.stack([b1, b2, b3], dim=1)

        # Build SE(3)
        pose = torch.eye(4, device=params.device)
        pose[:3, :3] = R
        pose[:3, 3] = t

        return pose

    def _render_from_pose(
        self,
        pose: torch.Tensor,
        intrinsics: torch.Tensor,
        height: int,
        width: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Render Gaussians from given camera pose.

        Args:
            pose: Camera pose [4, 4]
            intrinsics: Camera intrinsics [3, 3]
            height: Image height
            width: Image width

        Returns:
            Dict with 'rgb' [3, H, W] and optionally 'depth' [H, W]
        """
        # Stub implementation: generate plausible rendered output
        # In production, this uses diff-gaussian-rasterization CUDA kernel

        # Generate smooth gradient RGB to simulate rendered scene
        rendered_rgb = torch.rand(3, height, width, device=self.device) * 0.3 + 0.5

        # Generate depth map with some spatial variation
        y_coords = torch.linspace(0, 1, height, device=self.device)
        x_coords = torch.linspace(0, 1, width, device=self.device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        rendered_depth = 1.0 + 0.2 * (yy + xx) / 2.0

        return {
            'rgb': rendered_rgb,
            'depth': rendered_depth,
        }

    def _compute_tracking_loss(
        self,
        rendered: Dict[str, torch.Tensor],
        target_image: torch.Tensor,
        target_depth: Optional[torch.Tensor],
        mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute tracking loss between rendered and observed.

        Args:
            rendered: Rendered outputs (rgb, depth)
            target_image: Observed RGB [3, H, W]
            target_depth: Observed depth [H, W]
            mask: Valid pixel mask [H, W]

        Returns:
            Total loss and dict of individual losses
        """
        loss_dict = {}

        # RGB loss (L1)
        rgb_diff = torch.abs(rendered['rgb'] - target_image)
        if mask is not None:
            rgb_diff = rgb_diff * mask.unsqueeze(0)
        rgb_loss = rgb_diff.mean()
        loss_dict['rgb'] = rgb_loss.item()

        total_loss = self.config.rgb_weight * rgb_loss

        # Depth loss
        if self.config.use_depth and target_depth is not None:
            depth_diff = torch.abs(rendered['depth'] - target_depth)
            if mask is not None:
                depth_diff = depth_diff * mask
            depth_loss = depth_diff.mean()
            loss_dict['depth'] = depth_loss.item()

            total_loss = total_loss + self.config.depth_weight * depth_loss

        loss_dict['total'] = total_loss.item()

        return total_loss, loss_dict

    def reset_tracking_history(self):
        """Clear tracking history."""
        self.tracking_history = []

    def get_tracking_stats(self) -> Dict[str, Any]:
        """Get tracking statistics."""
        if not self.tracking_history:
            return {}

        losses = [h['final_loss'] for h in self.tracking_history]
        iterations = [h['iterations'] for h in self.tracking_history]

        return {
            'n_frames': len(self.tracking_history),
            'mean_loss': sum(losses) / len(losses),
            'mean_iterations': sum(iterations) / len(iterations),
            'total_iterations': sum(iterations),
        }
