# Phase 2: 3DGS + SLAM Development Specification

> Integrating 3D Gaussian Splatting with Simultaneous Localization and Mapping

---

## Overview

Phase 2 builds on Phase 1's 3DGS foundation to explore the integration of Gaussian Splatting with SLAM systems. This enables real-time, online 3D reconstruction from sequential camera input.

### Key Papers

1. **SplaTAM** (CVPR 2024): "SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM"
   - Paper: https://arxiv.org/abs/2312.02126
   - Code: https://github.com/spla-tam/SplaTAM

2. **MonoGS** (CVPR 2024): "Gaussian Splatting SLAM"
   - Paper: https://arxiv.org/abs/2312.06741
   - Code: https://github.com/muskie82/MonoGS

3. **GS-SLAM**: "GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting"
   - Paper: https://arxiv.org/abs/2311.11700

---

## Learning Objectives

By completing Phase 2, learners will:

1. Understand SLAM fundamentals and terminology
2. Learn how 3DGS integrates with SLAM pipelines
3. Implement tracking and mapping with Gaussians
4. Understand keyframe management strategies
5. Run SplaTAM and MonoGS on custom datasets

---

## Notebook Structure

### Notebook 00: Phase 2 Overview
- Introduction to 3DGS + SLAM paradigm
- Why combine 3DGS with SLAM?
- Comparison with traditional SLAM (ORB-SLAM, LSD-SLAM)
- Comparison with NeRF-based SLAM (iMAP, NICE-SLAM)
- Key challenges and solutions

### Notebook 01: SLAM Basics Review
- Visual Odometry (VO) fundamentals
- Feature-based vs Direct methods
- Bundle Adjustment basics
- Loop closure concepts
- Covisibility graphs
- Keyframe selection strategies

### Notebook 02: SplaTAM Architecture
- System overview
  ```
  RGB-D Input → Tracking → Mapping → Gaussian Map
                   ↑           ↓
                   ←←←←←←←←←←←←
  ```
- Tracking module: Gaussian-based camera pose estimation
- Mapping module: Gaussian optimization
- Silhouette-guided densification
- Implementation details

### Notebook 03: Gaussian Map Initialization
- From first frame initialization
- Depth-guided Gaussian placement
- Initial scale and opacity estimation
- Color initialization from RGB
- Handling RGB-D vs monocular input

### Notebook 04: Camera Tracking with Gaussians
- Rendering-based pose estimation
- Photometric loss for tracking
- Gradient-based pose optimization
- Handling tracking failures
- Comparison with feature-based tracking

### Notebook 05: Online Gaussian Optimization
- Incremental training strategies
- Window-based optimization
- Memory management for growing maps
- Balancing quality vs speed
- Progressive SH degree

### Notebook 06: Keyframe Management
- Keyframe selection criteria
  - Sufficient camera motion
  - Covisibility overlap
  - Tracking quality
- Keyframe graph structure
- Local vs global optimization
- Keyframe culling

### Notebook 07: SplaTAM Code Walkthrough
- Repository structure
- Key modules analysis
  - `slam.py`: Main SLAM loop
  - `camera.py`: Camera tracking
  - `gaussian.py`: Gaussian management
  - `render.py`: Differentiable rendering
- Configuration files
- Running on Replica dataset

### Notebook 08: MonoGS Comparison
- Monocular vs RGB-D approaches
- Scale ambiguity handling
- Depth estimation integration
- Architecture differences from SplaTAM
- Performance comparison

### Notebook 09: Custom Dataset SLAM
- Recording your own RGB-D sequences
- Using RealSense/Kinect cameras
- Dataset format requirements
- Running SplaTAM on custom data
- Troubleshooting common issues

---

## Source Code Modules

### src/slam/

```
src/slam/
├── __init__.py
├── keyframe.py        # Keyframe selection and management
├── tracking.py        # Camera pose estimation
├── mapping.py         # Gaussian map optimization
├── loop_closure.py    # Loop detection and closure
└── utils.py           # SLAM utilities
```

### Key Classes

#### KeyframeManager
```python
class KeyframeManager:
    """Manages keyframe selection and storage."""

    def __init__(self, config: KeyframeConfig):
        self.keyframes: List[Keyframe] = []
        self.covisibility_graph: Dict[int, Set[int]] = {}

    def should_add_keyframe(
        self,
        current_pose: torch.Tensor,
        current_image: torch.Tensor,
    ) -> bool:
        """Decide if current frame should be a keyframe."""
        pass

    def add_keyframe(
        self,
        frame_id: int,
        pose: torch.Tensor,
        image: torch.Tensor,
        depth: Optional[torch.Tensor] = None,
    ) -> Keyframe:
        """Add new keyframe to the map."""
        pass

    def get_covisible_keyframes(
        self,
        keyframe_id: int,
        min_overlap: float = 0.3,
    ) -> List[int]:
        """Get keyframes with sufficient overlap."""
        pass
```

#### GaussianTracker
```python
class GaussianTracker:
    """Camera tracking using Gaussian map."""

    def __init__(self, gaussians: GaussianModel, config: TrackingConfig):
        self.gaussians = gaussians
        self.config = config

    def track(
        self,
        image: torch.Tensor,
        depth: Optional[torch.Tensor],
        initial_pose: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        Estimate camera pose from image.

        Returns:
            Optimized pose and tracking loss
        """
        pass

    def render_from_pose(
        self,
        pose: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Render Gaussians from given pose."""
        pass
```

#### GaussianMapper
```python
class GaussianMapper:
    """Gaussian map optimization."""

    def __init__(self, config: MappingConfig):
        self.gaussians = GaussianModel(sh_degree=config.sh_degree)
        self.optimizer = None

    def add_gaussians_from_frame(
        self,
        image: torch.Tensor,
        depth: torch.Tensor,
        pose: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> int:
        """Add new Gaussians from a frame."""
        pass

    def optimize(
        self,
        keyframes: List[Keyframe],
        n_iterations: int = 100,
    ) -> float:
        """Optimize Gaussians using keyframes."""
        pass
```

---

## Dataset Requirements

### Supported Formats

1. **Replica Dataset** (recommended for testing)
   - Synthetic indoor scenes
   - Perfect depth and poses
   - Download: https://github.com/facebookresearch/Replica-Dataset

2. **TUM RGB-D Dataset**
   - Real RGB-D sequences
   - Ground truth poses from motion capture
   - Download: https://vision.in.tum.de/data/datasets/rgbd-dataset

3. **ScanNet**
   - Indoor scene reconstructions
   - Dense RGB-D data
   - Download: http://www.scan-net.org/

4. **Custom RGB-D**
   ```
   custom_dataset/
   ├── rgb/
   │   ├── frame_00000.png
   │   ├── frame_00001.png
   │   └── ...
   ├── depth/
   │   ├── frame_00000.png
   │   └── ...
   └── associations.txt  # Timestamps and file mappings
   ```

---

## Configuration

### Default Configuration (configs/slam_default.yaml)

```yaml
# Tracking
tracking:
  iterations: 40
  learning_rate: 0.01
  use_depth: true
  depth_weight: 0.5

# Mapping
mapping:
  iterations: 60
  learning_rate_position: 0.001
  learning_rate_features: 0.01
  densify_interval: 20

# Keyframe
keyframe:
  min_translation: 0.05  # meters
  min_rotation: 5.0      # degrees
  min_overlap: 0.7

# Gaussian
gaussian:
  sh_degree: 0  # Start with degree 0 for speed
  initial_opacity: 0.5
  initial_scale: 0.01

# System
system:
  device: "cuda"
  image_height: 480
  image_width: 640
```

---

## Timeline

### Week 1: Foundation
- Notebook 00: Overview
- Notebook 01: SLAM Basics
- Set up SplaTAM environment

### Week 2: Core Concepts
- Notebook 02: SplaTAM Architecture
- Notebook 03: Map Initialization
- Notebook 04: Camera Tracking

### Week 3: Advanced Topics
- Notebook 05: Online Optimization
- Notebook 06: Keyframe Management
- Implement src/slam module

### Week 4: Integration
- Notebook 07: SplaTAM Walkthrough
- Notebook 08: MonoGS Comparison
- Notebook 09: Custom Dataset

---

## Dependencies

```bash
# Core
torch>=2.0.0
numpy>=1.24.0
opencv-python>=4.8.0

# 3DGS
diff-gaussian-rasterization  # From official 3DGS
simple-knn

# SLAM specific
open3d>=0.17.0           # Point cloud processing
lietorch>=0.1.0          # Lie group operations (optional)
pytorch3d>=0.7.0         # 3D operations (optional)

# Visualization
plotly>=5.15.0
rerun-sdk>=0.8.0         # For 3D visualization
```

---

## Success Criteria

By the end of Phase 2:

1. **Understanding**: Learners can explain how 3DGS integrates with SLAM
2. **Implementation**: src/slam module is functional for basic tracking/mapping
3. **Practical**: Can run SplaTAM/MonoGS on custom RGB-D data
4. **Comparison**: Understand trade-offs between different approaches

---

## References

1. SplaTAM: https://spla-tam.github.io/
2. MonoGS: https://rmurai.co.uk/projects/GaussianSplattingSLAM/
3. 3DGS: https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/
4. TUM RGB-D Benchmark: https://vision.in.tum.de/data/datasets/rgbd-dataset/tools
5. Replica Dataset: https://github.com/facebookresearch/Replica-Dataset
