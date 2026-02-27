"""
PointMap: Dense 3D point representation.

A PointMap stores a 3D point for each pixel in an image, along with
optional confidence scores. This is a core representation in DUSt3R.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PointMap:
    """
    Dense point map: (H, W, 3) array of 3D coordinates.

    Similar to a depth map, but directly stores 3D coordinates instead of
    scalar depth values. This avoids the need for camera intrinsics during
    unprojection.

    Attributes:
        points: [H, W, 3] array of 3D coordinates
        confidence: [H, W] array of per-pixel confidence scores (optional)
        image: [H, W, 3] RGB image for coloring (optional)
    """

    def __init__(self, points, confidence=None, image=None):
        """
        Args:
            points: [H, W, 3] 3D coordinates (numpy or torch)
            confidence: [H, W] confidence scores in [0, 1]
            image: [H, W, 3] RGB image for visualization
        """
        if isinstance(points, torch.Tensor):
            self.points = points.cpu().numpy()
        else:
            self.points = np.asarray(points)

        self.H, self.W = self.points.shape[:2]

        if confidence is not None:
            if isinstance(confidence, torch.Tensor):
                self.confidence = confidence.cpu().numpy()
            else:
                self.confidence = np.asarray(confidence)
        else:
            # Default: uniform confidence
            self.confidence = np.ones((self.H, self.W))

        if image is not None:
            if isinstance(image, torch.Tensor):
                self.image = image.cpu().numpy()
            else:
                self.image = np.asarray(image)
        else:
            self.image = None

    def to_depth(self, camera_center=None):
        """
        Convert PointMap to depth map.

        If camera_center is None, assumes camera is at origin and points are
        in camera coordinates. Then depth = ||p||.

        Args:
            camera_center: [3] camera position in world coordinates

        Returns:
            depth: [H, W] depth map
        """
        if camera_center is None:
            # Simple case: depth = Z coordinate (assumes camera at origin)
            depth = np.abs(self.points[..., 2])
        else:
            # Camera at arbitrary location: depth = distance from camera
            camera_center = np.asarray(camera_center)
            distances = np.linalg.norm(
                self.points - camera_center[None, None, :],
                axis=-1
            )
            depth = distances

        return depth

    def to_pointcloud(self, confidence_threshold=0.0):
        """
        Convert PointMap to unstructured point cloud.

        Filters points by confidence and reshapes from (H, W, 3) to (N, 3).

        Args:
            confidence_threshold: Only keep points with confidence >= this

        Returns:
            points_3d: [N, 3] array of 3D points
            colors: [N, 3] RGB colors (if image available), else None
        """
        # Mask by confidence
        mask = self.confidence >= confidence_threshold
        points_3d = self.points[mask]

        if self.image is not None:
            colors = self.image[mask]
        else:
            colors = None

        return points_3d, colors

    def visualize(self, title='PointMap', confidence_threshold=0.0,
                  figsize=(12, 5)):
        """
        Visualize PointMap with matplotlib.

        Shows depth map and 3D point cloud side by side.

        Args:
            title: Plot title
            confidence_threshold: Filter points by confidence
            figsize: Figure size
        """
        fig = plt.figure(figsize=figsize)

        # Left: Depth map
        ax1 = fig.add_subplot(121)
        depth = self.to_depth()
        im = ax1.imshow(depth, cmap='plasma')
        ax1.set_title(f'{title} - Depth Map')
        ax1.axis('off')
        plt.colorbar(im, ax=ax1, label='Depth')

        # Right: 3D point cloud
        ax2 = fig.add_subplot(122, projection='3d')

        points_3d, colors = self.to_pointcloud(confidence_threshold)

        if colors is not None:
            # Normalize colors to [0, 1] if needed
            if colors.max() > 1.0:
                colors = colors / 255.0
            ax2.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                       c=colors, s=1, alpha=0.5)
        else:
            # Color by depth
            depth_flat = depth.flatten()
            scatter = ax2.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                                 c=depth_flat[self.confidence.flatten() >= confidence_threshold],
                                 cmap='plasma', s=1, alpha=0.5)
            plt.colorbar(scatter, ax=ax2, label='Depth')

        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title(f'{title} - 3D Point Cloud')

        plt.tight_layout()
        return fig

    def filter_by_confidence(self, threshold=0.5):
        """
        Create a new PointMap with low-confidence points masked.

        Args:
            threshold: Confidence threshold

        Returns:
            New PointMap with low-confidence points set to NaN
        """
        points_filtered = self.points.copy()
        mask = self.confidence < threshold
        points_filtered[mask] = np.nan

        return PointMap(points_filtered, confidence=self.confidence,
                       image=self.image)

    def __repr__(self):
        return (f"PointMap(H={self.H}, W={self.W}, "
                f"confidence_range=[{self.confidence.min():.3f}, "
                f"{self.confidence.max():.3f}])")
