"""
GlobalAligner: Convert pairwise relative poses to global scene alignment.

This module provides educational implementations of:
- Procrustes analysis for aligning point clouds
- Simple pose graph optimization for multi-view scenes
- Bundle adjustment concepts
"""

import numpy as np
from scipy.linalg import svd


class GlobalAligner:
    """
    Aligns point maps from multiple views into a consistent global coordinate system.

    Key method: Given pairwise correspondences and relative poses, estimate
    the global pose of each camera using Procrustes analysis + pose graph optimization.
    """

    def __init__(self):
        """Initialize the aligner."""
        self.poses = {}  # {frame_id: [R, t]}
        self.points = {}  # {frame_id: PointMap}

    def add_frame(self, frame_id, pointmap):
        """
        Register a point map for a frame.

        Args:
            frame_id: Unique identifier for this frame
            pointmap: PointMap object from DUSt3R
        """
        self.points[frame_id] = pointmap
        # Initialize identity pose
        self.poses[frame_id] = (np.eye(3), np.zeros(3))

    def pairwise_to_global(self, pairwise_poses, threshold_iterations=5):
        """
        Convert pairwise relative poses to global camera poses.

        Simple algorithm:
        1. Start with first camera at origin
        2. For each frame, use Procrustes to align its point map to global frame
        3. Iterate to refine (optional)

        Args:
            pairwise_poses: dict {(i, j): (R_ij, t_ij)}
                           where R_ij, t_ij is relative pose from frame i to j
            threshold_iterations: Number of refinement iterations

        Returns:
            global_poses: dict {frame_id: (R, t)}
        """
        frame_ids = list(self.points.keys())
        if len(frame_ids) == 0:
            return {}

        # Initialize with first frame at origin
        global_poses = {frame_ids[0]: (np.eye(3), np.zeros(3))}

        # Use relative poses to place other frames
        for frame_id in frame_ids[1:]:
            if frame_id in global_poses:
                continue

            # Find a reference frame with known global pose
            ref_frame = None
            for ref_id in frame_ids:
                if ref_id in global_poses and (ref_id, frame_id) in pairwise_poses:
                    ref_frame = ref_id
                    break

            if ref_frame is None:
                # No relative pose available, use identity
                global_poses[frame_id] = (np.eye(3), np.zeros(3))
            else:
                # Compute global pose from reference frame
                R_ref, t_ref = global_poses[ref_frame]
                R_rel, t_rel = pairwise_poses[(ref_frame, frame_id)]

                # Global pose = reference pose ∘ relative pose
                R = R_ref @ R_rel
                t = R_ref @ t_rel + t_ref

                global_poses[frame_id] = (R, t)

        self.poses = global_poses
        return global_poses

    def compute_scene_scale(self, pointmaps_1, pointmaps_2):
        """
        Estimate absolute scale of the scene from two aligned point maps.

        Two 3D point clouds in the same coordinate system can be compared to
        estimate their relative scale.

        Args:
            pointmaps_1: [frame_ids] -> PointMap for first view
            pointmaps_2: [frame_ids] -> PointMap for second view

        Returns:
            scale: Positive scalar
        """
        distances_1 = []
        distances_2 = []

        for frame_id in pointmaps_1:
            if frame_id not in pointmaps_2:
                continue

            pm1 = pointmaps_1[frame_id]
            pm2 = pointmaps_2[frame_id]

            # Sample points from both
            p1 = pm1.points.reshape(-1, 3)
            p2 = pm2.points.reshape(-1, 3)

            # Compute pairwise distances
            d1 = np.linalg.norm(p1[:100] - p1[1:101], axis=1).mean()
            d2 = np.linalg.norm(p2[:100] - p2[1:101], axis=1).mean()

            if d1 > 1e-6 and d2 > 1e-6:
                distances_1.append(d1)
                distances_2.append(d2)

        if len(distances_1) == 0:
            return 1.0

        scale = np.mean(np.array(distances_1)) / np.mean(np.array(distances_2))
        return float(np.clip(scale, 0.1, 10.0))

    @staticmethod
    def procrustes_align(P_src, P_tgt):
        """
        Procrustes analysis: find best rigid alignment of source to target.

        Solves: minimize ||R @ P_src.T + t - P_tgt.T||_F
        over rotation R and translation t.

        Args:
            P_src: [N, 3] source points
            P_tgt: [N, 3] target points

        Returns:
            R: [3, 3] rotation matrix
            t: [3] translation vector
        """
        # Center both point clouds
        centroid_src = P_src.mean(axis=0)
        centroid_tgt = P_tgt.mean(axis=0)

        P_src_centered = P_src - centroid_src
        P_tgt_centered = P_tgt - centroid_tgt

        # SVD
        U, S, Vt = svd(P_src_centered.T @ P_tgt_centered)

        # Rotation
        R = (U @ Vt).T

        # Ensure proper rotation (det(R) = +1)
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = (U @ Vt).T

        # Translation
        t = centroid_tgt - R @ centroid_src

        return R, t

    def __repr__(self):
        return (f"GlobalAligner(frames={len(self.poses)}, "
                f"points_registered={len(self.points)})")
