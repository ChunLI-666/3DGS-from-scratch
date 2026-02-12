"""
Gaussian Mapping for 3DGS-SLAM.

This module provides Gaussian map management and optimization
for SLAM systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple


@dataclass
class MappingConfig:
    """Configuration for Gaussian mapping."""

    # Optimization
    iterations: int = 60
    learning_rate_position: float = 0.001
    learning_rate_features: float = 0.01
    learning_rate_opacity: float = 0.05
    learning_rate_scaling: float = 0.005
    learning_rate_rotation: float = 0.001

    # Densification
    densify_interval: int = 20
    densify_grad_threshold: float = 0.0002
    min_opacity: float = 0.005

    # Gaussian initialization
    sh_degree: int = 0  # Start low for speed
    initial_opacity: float = 0.5
    initial_scale: float = 0.01

    # Memory management
    max_gaussians: int = 500000


class GaussianMapper:
    """
    Gaussian map management and optimization.

    Handles:
    - Adding new Gaussians from depth observations
    - Optimizing Gaussians using keyframes
    - Densification and pruning
    - Memory management

    Attributes:
        config: Mapping configuration
        n_gaussians: Current number of Gaussians
    """

    def __init__(self, config: MappingConfig = None, device: str = "cuda"):
        self.config = config or MappingConfig()
        self.device = torch.device(device)

        # Gaussian parameters (will be nn.Parameters when initialized)
        self._xyz = None  # [N, 3]
        self._features_dc = None  # [N, 1, 3]
        self._features_rest = None  # [N, K, 3]
        self._scaling = None  # [N, 3]
        self._rotation = None  # [N, 4]
        self._opacity = None  # [N, 1]

        self.optimizer = None
        self._initialized = False

        # Statistics
        self.stats = {
            'total_added': 0,
            'total_pruned': 0,
            'optimization_steps': 0,
        }

    @property
    def n_gaussians(self) -> int:
        """Current number of Gaussians."""
        if self._xyz is None:
            return 0
        return self._xyz.shape[0]

    def initialize(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        scales: Optional[torch.Tensor] = None,
    ):
        """
        Initialize Gaussian map from points.

        Args:
            points: 3D points [N, 3]
            colors: RGB colors [N, 3] in [0, 1]
            scales: Optional initial scales [N, 3]
        """
        N = points.shape[0]

        # Positions
        self._xyz = nn.Parameter(points.to(self.device))

        # Colors -> SH DC
        SH_C0 = 0.28209479177387814
        fused_color = (colors - 0.5) / SH_C0
        self._features_dc = nn.Parameter(
            fused_color.unsqueeze(1).to(self.device)
        )

        # Higher-order SH (zeros for now)
        n_sh_rest = ((self.config.sh_degree + 1) ** 2 - 1)
        self._features_rest = nn.Parameter(
            torch.zeros(N, n_sh_rest, 3, device=self.device)
        )

        # Scales
        if scales is None:
            scales = torch.full((N, 3), self.config.initial_scale)
        self._scaling = nn.Parameter(
            torch.log(scales.to(self.device))
        )

        # Rotations (identity quaternions)
        rots = torch.zeros(N, 4, device=self.device)
        rots[:, 0] = 1.0
        self._rotation = nn.Parameter(rots)

        # Opacities
        opacity_logit = torch.log(
            torch.tensor(self.config.initial_opacity / (1 - self.config.initial_opacity))
        )
        self._opacity = nn.Parameter(
            torch.full((N, 1), opacity_logit, device=self.device)
        )

        # Setup optimizer
        self._setup_optimizer()

        self._initialized = True
        self.stats['total_added'] = N

    def add_gaussians_from_frame(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        pose: torch.Tensor,
        intrinsics: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        subsample: int = 4,
    ) -> int:
        """
        Add new Gaussians from an RGB-D frame.

        Projects depth to 3D points and creates Gaussians.

        Args:
            image: RGB image [3, H, W] or [H, W, 3]
            depth: Depth map [H, W]
            pose: Camera pose [4, 4] (world to camera)
            intrinsics: Camera intrinsics [3, 3]
            mask: Valid depth mask [H, W]
            subsample: Subsampling factor (1 = all pixels)

        Returns:
            Number of Gaussians added
        """
        if image.dim() == 3 and image.shape[-1] == 3:
            image = image.permute(2, 0, 1)

        H, W = depth.shape
        device = depth.device

        # Create pixel grid
        v, u = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing='ij'
        )

        # Subsample
        if subsample > 1:
            v = v[::subsample, ::subsample]
            u = u[::subsample, ::subsample]
            depth_sub = depth[::subsample, ::subsample]
            image_sub = image[:, ::subsample, ::subsample]
            if mask is not None:
                mask_sub = mask[::subsample, ::subsample]
            else:
                mask_sub = None
        else:
            depth_sub = depth
            image_sub = image
            mask_sub = mask

        # Flatten
        v = v.reshape(-1).float()
        u = u.reshape(-1).float()
        z = depth_sub.reshape(-1)
        colors = image_sub.reshape(3, -1).T  # [N, 3]

        # Filter by mask and valid depth
        if mask_sub is not None:
            valid = mask_sub.reshape(-1) & (z > 0)
        else:
            valid = z > 0

        v = v[valid]
        u = u[valid]
        z = z[valid]
        colors = colors[valid]

        if len(z) == 0:
            return 0

        # Unproject to camera coordinates
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        z_cam = z

        points_cam = torch.stack([x_cam, y_cam, z_cam], dim=1)  # [N, 3]

        # Transform to world coordinates
        # pose is world-to-camera, so world = pose^{-1} @ cam
        pose_inv = torch.linalg.inv(pose)
        R = pose_inv[:3, :3]
        t = pose_inv[:3, 3]

        points_world = (R @ points_cam.T).T + t  # [N, 3]

        # Estimate scales from depth gradient (simple approach)
        # More sophisticated: use local point density
        scales = torch.full(
            (len(points_world), 3),
            self.config.initial_scale,
            device=device
        )

        # Add to existing Gaussians
        n_added = self._add_gaussians(points_world, colors, scales)

        return n_added

    def _add_gaussians(
        self,
        points: torch.Tensor,
        colors: torch.Tensor,
        scales: torch.Tensor,
    ) -> int:
        """
        Add Gaussians to the map.

        Args:
            points: 3D positions [N, 3]
            colors: RGB colors [N, 3]
            scales: Initial scales [N, 3]

        Returns:
            Number of Gaussians added
        """
        N = points.shape[0]

        if not self._initialized:
            self.initialize(points, colors, scales)
            return N

        # Check memory limit
        if self.n_gaussians + N > self.config.max_gaussians:
            # Need to prune first or skip
            N = self.config.max_gaussians - self.n_gaussians
            if N <= 0:
                return 0
            points = points[:N]
            colors = colors[:N]
            scales = scales[:N]

        # Create new parameters
        SH_C0 = 0.28209479177387814
        new_xyz = points.to(self.device)
        new_features_dc = ((colors - 0.5) / SH_C0).unsqueeze(1).to(self.device)
        new_features_rest = torch.zeros(
            N, self._features_rest.shape[1], 3, device=self.device
        )
        new_scaling = torch.log(scales.to(self.device))
        new_rotation = torch.zeros(N, 4, device=self.device)
        new_rotation[:, 0] = 1.0
        opacity_logit = torch.log(
            torch.tensor(self.config.initial_opacity / (1 - self.config.initial_opacity))
        )
        new_opacity = torch.full((N, 1), opacity_logit, device=self.device)

        # Concatenate with existing
        self._xyz = nn.Parameter(torch.cat([self._xyz.data, new_xyz], dim=0))
        self._features_dc = nn.Parameter(
            torch.cat([self._features_dc.data, new_features_dc], dim=0)
        )
        self._features_rest = nn.Parameter(
            torch.cat([self._features_rest.data, new_features_rest], dim=0)
        )
        self._scaling = nn.Parameter(
            torch.cat([self._scaling.data, new_scaling], dim=0)
        )
        self._rotation = nn.Parameter(
            torch.cat([self._rotation.data, new_rotation], dim=0)
        )
        self._opacity = nn.Parameter(
            torch.cat([self._opacity.data, new_opacity], dim=0)
        )

        # Recreate optimizer
        self._setup_optimizer()

        self.stats['total_added'] += N

        return N

    def optimize(
        self,
        keyframes: List[Any],  # List[Keyframe]
        n_iterations: Optional[int] = None,
    ) -> float:
        """
        Optimize Gaussians using keyframes.

        Args:
            keyframes: List of keyframes for optimization
            n_iterations: Override config iterations

        Returns:
            Final loss value
        """
        if not self._initialized or len(keyframes) == 0:
            return float('inf')

        n_iters = n_iterations or self.config.iterations
        total_loss = 0.0

        for i in range(n_iters):
            self.optimizer.zero_grad()

            # Random keyframe
            import random
            kf = random.choice(keyframes)

            # Render from keyframe pose
            rendered = self._render_from_keyframe(kf)

            # Compute loss
            loss = self._compute_mapping_loss(rendered, kf)

            # Backward
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Densification
            if i > 0 and i % self.config.densify_interval == 0:
                self._densify_and_prune()

            self.stats['optimization_steps'] += 1

        return total_loss / n_iters

    def _setup_optimizer(self):
        """Setup optimizer with per-parameter learning rates."""
        param_groups = [
            {'params': [self._xyz], 'lr': self.config.learning_rate_position, 'name': 'xyz'},
            {'params': [self._features_dc], 'lr': self.config.learning_rate_features, 'name': 'f_dc'},
            {'params': [self._features_rest], 'lr': self.config.learning_rate_features / 20, 'name': 'f_rest'},
            {'params': [self._scaling], 'lr': self.config.learning_rate_scaling, 'name': 'scaling'},
            {'params': [self._rotation], 'lr': self.config.learning_rate_rotation, 'name': 'rotation'},
            {'params': [self._opacity], 'lr': self.config.learning_rate_opacity, 'name': 'opacity'},
        ]
        self.optimizer = torch.optim.Adam(param_groups)

    def _render_from_keyframe(self, keyframe) -> Dict[str, torch.Tensor]:
        """
        Render Gaussians from keyframe viewpoint.

        This is a placeholder - actual implementation uses CUDA rasterizer.
        """
        H, W = keyframe.image.shape[1], keyframe.image.shape[2]

        # Placeholder render
        return {
            'rgb': torch.zeros(3, H, W, device=self.device),
            'depth': torch.ones(H, W, device=self.device),
        }

    def _compute_mapping_loss(
        self,
        rendered: Dict[str, torch.Tensor],
        keyframe,
    ) -> torch.Tensor:
        """Compute mapping loss for optimization."""
        rgb_loss = F.l1_loss(rendered['rgb'], keyframe.image)

        if keyframe.depth is not None:
            depth_loss = F.l1_loss(rendered['depth'], keyframe.depth)
            return rgb_loss + 0.5 * depth_loss

        return rgb_loss

    def _densify_and_prune(self):
        """Densify and prune Gaussians based on gradients and opacity."""
        with torch.no_grad():
            # Prune low opacity
            opacity = torch.sigmoid(self._opacity)
            mask = (opacity > self.config.min_opacity).squeeze()

            if mask.sum() < self.n_gaussians:
                n_pruned = self.n_gaussians - mask.sum().item()

                self._xyz = nn.Parameter(self._xyz.data[mask])
                self._features_dc = nn.Parameter(self._features_dc.data[mask])
                self._features_rest = nn.Parameter(self._features_rest.data[mask])
                self._scaling = nn.Parameter(self._scaling.data[mask])
                self._rotation = nn.Parameter(self._rotation.data[mask])
                self._opacity = nn.Parameter(self._opacity.data[mask])

                self._setup_optimizer()
                self.stats['total_pruned'] += n_pruned

    def get_gaussian_params(self) -> Dict[str, torch.Tensor]:
        """Get current Gaussian parameters as dict."""
        return {
            'xyz': self._xyz.data,
            'features_dc': self._features_dc.data,
            'features_rest': self._features_rest.data,
            'scaling': self._scaling.data,
            'rotation': self._rotation.data,
            'opacity': self._opacity.data,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get mapping statistics."""
        return {
            **self.stats,
            'n_gaussians': self.n_gaussians,
        }
