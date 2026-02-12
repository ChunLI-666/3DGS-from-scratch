"""
Keyframe Management for 3DGS-SLAM.

This module provides keyframe selection, storage, and management
utilities for SLAM systems using 3D Gaussian Splatting.
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
import numpy as np


@dataclass
class KeyframeConfig:
    """Configuration for keyframe management."""

    # Motion thresholds for new keyframe
    min_translation: float = 0.05  # meters
    min_rotation: float = 5.0  # degrees

    # Overlap requirements
    min_overlap: float = 0.7  # 70% overlap with previous keyframe
    max_overlap: float = 0.95  # Don't add if too similar

    # Storage limits
    max_keyframes: int = 1000

    # Covisibility
    covisibility_threshold: float = 0.3  # Min overlap for covisibility


@dataclass
class Keyframe:
    """
    Represents a single keyframe in the SLAM system.

    Stores the camera pose, image data, and associated Gaussian indices.
    """

    frame_id: int
    timestamp: float

    # Camera pose (world to camera transform)
    pose: torch.Tensor  # [4, 4] SE(3) matrix

    # Image data
    image: torch.Tensor  # [3, H, W] RGB
    depth: Optional[torch.Tensor] = None  # [H, W] depth map

    # Camera intrinsics
    fx: float = 500.0
    fy: float = 500.0
    cx: float = 320.0
    cy: float = 240.0

    # Associated Gaussians (indices into main Gaussian array)
    gaussian_indices: List[int] = field(default_factory=list)

    # Covisible keyframes
    covisible_keyframes: Set[int] = field(default_factory=set)

    @property
    def intrinsics(self) -> torch.Tensor:
        """Camera intrinsic matrix [3, 3]."""
        K = torch.eye(3)
        K[0, 0] = self.fx
        K[1, 1] = self.fy
        K[0, 2] = self.cx
        K[1, 2] = self.cy
        return K

    @property
    def position(self) -> torch.Tensor:
        """Camera position in world coordinates [3]."""
        # Pose is world-to-camera, so camera position is -R^T @ t
        R = self.pose[:3, :3]
        t = self.pose[:3, 3]
        return -R.T @ t

    @property
    def rotation(self) -> torch.Tensor:
        """Camera rotation matrix [3, 3]."""
        return self.pose[:3, :3]


class KeyframeManager:
    """
    Manages keyframe selection, storage, and covisibility.

    Handles:
    - Keyframe selection based on motion criteria
    - Covisibility graph maintenance
    - Keyframe retrieval for optimization
    """

    def __init__(self, config: KeyframeConfig = None):
        self.config = config or KeyframeConfig()

        self.keyframes: Dict[int, Keyframe] = {}
        self.keyframe_order: List[int] = []  # Order of insertion

        # Covisibility graph: keyframe_id -> set of covisible keyframe_ids
        self.covisibility_graph: Dict[int, Set[int]] = {}

        self._next_id = 0

    def should_add_keyframe(
        self,
        current_pose: torch.Tensor,
        current_image: torch.Tensor,
        overlap_ratio: float = None,
    ) -> bool:
        """
        Decide if current frame should be added as keyframe.

        Criteria:
        1. Sufficient motion since last keyframe
        2. Not too much overlap (redundant)
        3. Enough overlap (can track)

        Args:
            current_pose: Current camera pose [4, 4]
            current_image: Current RGB image [3, H, W]
            overlap_ratio: Computed overlap with last keyframe (if available)

        Returns:
            True if should add as keyframe
        """
        if len(self.keyframes) == 0:
            return True  # Always add first frame

        # Get last keyframe
        last_kf = self.keyframes[self.keyframe_order[-1]]

        # Check motion
        translation, rotation_deg = self._compute_relative_motion(
            last_kf.pose, current_pose
        )

        motion_sufficient = (
            translation > self.config.min_translation or
            rotation_deg > self.config.min_rotation
        )

        if not motion_sufficient:
            return False

        # Check overlap if provided
        if overlap_ratio is not None:
            if overlap_ratio > self.config.max_overlap:
                return False  # Too similar
            if overlap_ratio < self.config.min_overlap:
                return False  # Too different (tracking may fail)

        return True

    def add_keyframe(
        self,
        frame_id: int,
        pose: torch.Tensor,
        image: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
        timestamp: float = 0.0,
        intrinsics: Tuple[float, float, float, float] = None,
    ) -> Keyframe:
        """
        Add a new keyframe.

        Args:
            frame_id: Original frame ID from sequence
            pose: Camera pose [4, 4]
            image: RGB image [3, H, W]
            depth: Optional depth map [H, W]
            timestamp: Frame timestamp
            intrinsics: (fx, fy, cx, cy) if not default

        Returns:
            Created Keyframe object
        """
        kf_id = self._next_id
        self._next_id += 1

        # Create keyframe
        kf = Keyframe(
            frame_id=frame_id,
            timestamp=timestamp,
            pose=pose.clone(),
            image=image.clone(),
            depth=depth.clone() if depth is not None else None,
        )

        if intrinsics is not None:
            kf.fx, kf.fy, kf.cx, kf.cy = intrinsics

        # Store
        self.keyframes[kf_id] = kf
        self.keyframe_order.append(kf_id)
        self.covisibility_graph[kf_id] = set()

        # Update covisibility with recent keyframes
        self._update_covisibility(kf_id)

        # Enforce max keyframes
        if len(self.keyframes) > self.config.max_keyframes:
            self._cull_keyframes()

        return kf

    def get_keyframe(self, kf_id: int) -> Optional[Keyframe]:
        """Get keyframe by ID."""
        return self.keyframes.get(kf_id)

    def get_recent_keyframes(self, n: int = 5) -> List[Keyframe]:
        """Get N most recent keyframes."""
        recent_ids = self.keyframe_order[-n:]
        return [self.keyframes[kf_id] for kf_id in recent_ids]

    def get_covisible_keyframes(
        self,
        kf_id: int,
        min_overlap: float = None,
    ) -> List[int]:
        """
        Get IDs of keyframes covisible with given keyframe.

        Args:
            kf_id: Query keyframe ID
            min_overlap: Minimum overlap threshold (uses config default if None)

        Returns:
            List of covisible keyframe IDs
        """
        if kf_id not in self.covisibility_graph:
            return []

        return list(self.covisibility_graph[kf_id])

    def get_keyframes_for_mapping(
        self,
        current_kf_id: int,
        window_size: int = 5,
    ) -> List[Keyframe]:
        """
        Get keyframes for mapping optimization.

        Returns covisible keyframes + recent keyframes.

        Args:
            current_kf_id: Current keyframe ID
            window_size: Number of recent keyframes to include

        Returns:
            List of keyframes for optimization
        """
        kf_ids = set()

        # Add covisible
        kf_ids.update(self.get_covisible_keyframes(current_kf_id))

        # Add recent
        recent_ids = self.keyframe_order[-window_size:]
        kf_ids.update(recent_ids)

        # Always include current
        kf_ids.add(current_kf_id)

        return [self.keyframes[kf_id] for kf_id in kf_ids if kf_id in self.keyframes]

    def _compute_relative_motion(
        self,
        pose1: torch.Tensor,
        pose2: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Compute relative translation (meters) and rotation (degrees).

        Args:
            pose1: First pose [4, 4]
            pose2: Second pose [4, 4]

        Returns:
            (translation_meters, rotation_degrees)
        """
        # Relative pose: pose2 @ pose1.inv()
        relative = pose2 @ torch.linalg.inv(pose1)

        # Translation
        translation = relative[:3, 3].norm().item()

        # Rotation (angle from rotation matrix)
        R = relative[:3, :3]
        trace = R.trace()
        cos_angle = (trace - 1) / 2
        cos_angle = torch.clamp(cos_angle, -1, 1)
        angle_rad = torch.acos(cos_angle)
        angle_deg = angle_rad.item() * 180 / np.pi

        return translation, angle_deg

    def _update_covisibility(self, new_kf_id: int):
        """
        Update covisibility graph for new keyframe.

        Simple implementation: Mark recent keyframes as covisible.
        Full implementation would use Gaussian visibility overlap.
        """
        new_kf = self.keyframes[new_kf_id]

        for other_id in self.keyframe_order[:-1]:  # Exclude self
            other_kf = self.keyframes[other_id]

            # Simple proximity check (should use actual visibility)
            distance = (new_kf.position - other_kf.position).norm().item()

            if distance < 2.0:  # Within 2 meters
                self.covisibility_graph[new_kf_id].add(other_id)
                self.covisibility_graph[other_id].add(new_kf_id)

                new_kf.covisible_keyframes.add(other_id)
                other_kf.covisible_keyframes.add(new_kf_id)

    def _cull_keyframes(self):
        """
        Remove redundant keyframes to stay under limit.

        Culling criteria:
        - Keep first and last keyframes
        - Remove keyframes with high covisibility (redundant)
        """
        if len(self.keyframes) <= self.config.max_keyframes:
            return

        # Simple: remove oldest non-essential keyframes
        # More sophisticated: analyze covisibility

        n_to_remove = len(self.keyframes) - self.config.max_keyframes

        # Don't remove first or last
        candidates = self.keyframe_order[1:-1]

        for i in range(min(n_to_remove, len(candidates))):
            kf_id = candidates[i]

            # Remove from structures
            del self.keyframes[kf_id]
            self.keyframe_order.remove(kf_id)

            # Clean covisibility
            if kf_id in self.covisibility_graph:
                for other_id in self.covisibility_graph[kf_id]:
                    if other_id in self.covisibility_graph:
                        self.covisibility_graph[other_id].discard(kf_id)
                del self.covisibility_graph[kf_id]

    def __len__(self) -> int:
        return len(self.keyframes)

    def __iter__(self):
        for kf_id in self.keyframe_order:
            yield self.keyframes[kf_id]
