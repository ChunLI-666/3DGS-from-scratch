# Phase 4: Feed-forward Gaussian Splatting Development Specification

> **Document Version**: v1.0
> **Created**: 2026-02-12
> **Target Audience**: Learners with Phase 1 (3DGS) foundation; SLAM/MVS background helpful
> **Reference Outline**: `tutorials/Phase4_FeedForward_Gaussian.md`

---

## 1. Project Goal

### 1.1 Overview

Phase 4 introduces the **feed-forward paradigm** for 3D Gaussian Splatting. Unlike the per-scene optimization approach in Phase 1, feed-forward methods use pretrained neural networks to predict Gaussian parameters directly from input images in a single forward pass, enabling real-time novel view synthesis without per-scene training.

### 1.2 Key Papers

1. **MVSplat** (ECCV 2024): "MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images"
   - Paper: https://arxiv.org/abs/2403.14627
   - Code: https://github.com/donydchen/mvsplat
   - Key innovation: Cost Volume + Gaussian prediction

2. **pixelSplat** (CVPR 2024): "pixelSplat: 3D Gaussian Splats from Image Pairs"
   - Paper: https://arxiv.org/abs/2312.12337
   - Code: https://github.com/dcharatan/pixelsplat
   - Key innovation: Implicit geometry learning without Cost Volume

3. **DepthSplat** (2025): Depth-prior enhanced feed-forward Gaussian
   - Paper: https://arxiv.org/abs/2412.18010
   - Key innovation: Monocular depth prior integration

### 1.3 Learning Objectives

By completing Phase 4, learners will:

1. Understand the paradigm shift from optimization-based to feed-forward 3DGS
2. Master the Cost Volume / Plane Sweeping mechanism from MVS
3. Understand pixel-aligned Gaussian representation
4. Compare MVSplat vs pixelSplat architectures and trade-offs
5. Implement simplified feed-forward Gaussian prediction modules
6. Run MVSplat/pixelSplat inference on standard benchmarks
7. Understand 2025 advances (DepthSplat and beyond)

### 1.4 Design Principles

1. **Paradigm Contrast**: Every concept is compared with Phase 1's optimization approach
2. **Progressive Complexity**: From fundamentals to full architectures
3. **Runnable Code**: All modules work with PyTorch, no heavy dependencies for core concepts
4. **Visual-First**: Architecture diagrams, data flow, and interactive visualizations
5. **SLAM Connection**: Relate Cost Volume to MVS in SLAM, pixel-aligned to surfel mapping

---

## 2. Notebook Structure

### Overview Table

| Notebook | Topic | Duration | Difficulty |
|----------|-------|----------|------------|
| 00 | Phase 4 Overview & Paradigm Shift | 45min | ★★☆☆☆ |
| 01 | Cost Volume & Plane Sweeping | 75min | ★★★☆☆ |
| 02 | Pixel-aligned Gaussian Representation | 60min | ★★★☆☆ |
| 03 | MVSplat Architecture Deep Dive | 90min | ★★★★☆ |
| 04 | pixelSplat & Implicit Geometry | 75min | ★★★★☆ |
| 05 | Feed-forward Training & Loss Design | 75min | ★★★★☆ |
| 06 | MVSplat Code Walkthrough | 90min | ★★★★★ |
| 07 | Inference & Evaluation | 60min | ★★★☆☆ |
| 08 | DepthSplat & 2025 Advances | 60min | ★★★★☆ |
| 09 | MVSplat vs pixelSplat Comparison | 75min | ★★★★☆ |

---

### Notebook 00: Phase 4 Overview & Paradigm Shift

**Goal**: Establish the motivation and big picture for feed-forward 3DGS

**Content**:
1. **Recap of Phase 1 optimization approach**
   - Per-scene optimization: SfM → init → optimize → render
   - Limitations: slow, no generalization, needs COLMAP

2. **The feed-forward paradigm**
   - Pretrain once, infer everywhere
   - From minutes to seconds
   - No camera poses needed (potentially)

3. **Paradigm comparison table**
   ```
   ┌────────────────┬──────────────────┬──────────────────┐
   │     Aspect     │   Optimization   │   Feed-forward   │
   ├────────────────┼──────────────────┼──────────────────┤
   │ Training       │ Per-scene        │ Pre-trained      │
   │ Inference      │ N/A (train=infer)│ Single pass      │
   │ Generalization │ None             │ Cross-scene      │
   │ Camera poses   │ Required (SfM)   │ Optional         │
   │ Time           │ 30min+           │ Seconds          │
   │ Quality        │ High             │ Medium-High      │
   └────────────────┴──────────────────┴──────────────────┘
   ```

4. **Method landscape**
   - With Cost Volume: MVSplat
   - Without Cost Volume: pixelSplat
   - With depth prior: DepthSplat
   - Multi-view foundation: VGGT (Phase 5)

5. **Connection to SLAM**
   - Online 3D reconstruction → instant 3D
   - MVS in SLAM vs feed-forward reconstruction

**Visualizations**:
- Pipeline comparison diagram (optimization vs feed-forward)
- Method timeline / taxonomy chart
- Quality vs Speed trade-off scatter plot

---

### Notebook 01: Cost Volume & Plane Sweeping

**Goal**: Deeply understand the geometric foundation of MVSplat

**Content**:
1. **Multi-View Stereo (MVS) review**
   - Stereo matching basics
   - Epipolar geometry recap
   - Disparity and depth relationship

2. **Plane Sweeping algorithm**
   - Depth hypotheses sampling (uniform, log-uniform)
   - Homography warping at each depth plane
   - Feature matching cost computation

3. **Cost Volume construction**
   - Cost Volume tensor: C ∈ R^(H × W × D × C)
   - Variance-based cost (feature difference)
   - Multi-view aggregation

4. **From Cost Volume to depth**
   - Soft argmin for depth regression
   - Probability volume interpretation
   - Depth uncertainty estimation

5. **Connection to SLAM MVS**
   - PatchMatch in SLAM dense mapping
   - Keyframe-based MVS
   - Real-time Cost Volume in DeepMVS

**Code Implementation**:
```python
class PlaneSweepCostVolume:
    """Educational implementation of plane sweep cost volume."""

    def __init__(self, num_depths=64, min_depth=0.5, max_depth=100.0):
        self.depth_planes = torch.linspace(...)

    def build(self, feat_ref, feat_src, K_ref, K_src, T_src_ref):
        """Build cost volume by warping source features."""
        pass

    def depth_regression(self, cost_volume):
        """Soft argmin depth regression."""
        pass
```

**Visualizations**:
- Plane sweeping animation (depth planes + warping)
- Cost Volume slice visualization
- Depth probability distribution per pixel
- Interactive depth plane slider

---

### Notebook 02: Pixel-aligned Gaussian Representation

**Goal**: Understand how feed-forward methods represent Gaussians differently

**Content**:
1. **Original 3DGS Gaussian representation (recap)**
   - Random initialization from SfM point cloud
   - Free-form positions, scales, rotations
   - Adaptive densification

2. **Pixel-aligned Gaussians**
   - One Gaussian per pixel (structured layout)
   - Center = back-projected pixel at predicted depth
   - Parameters predicted by neural network
   - No densification needed

3. **Back-projection from depth**
   - μ = K^{-1} @ [u, v, 1]^T × depth
   - From image plane to 3D world

4. **Gaussian parameter prediction**
   - Depth → 3D position
   - Network → covariance (scale + rotation)
   - Network → opacity
   - Image color → color (or predicted SH)

5. **Comparison with surfel mapping (SLAM)**
   - Surfel = oriented disk ≈ flat Gaussian
   - Pixel-aligned surfel in ElasticFusion
   - Depth-based placement similarity

**Code Implementation**:
```python
class PixelAlignedGaussians:
    """Pixel-aligned Gaussian representation."""

    def from_depth_and_features(self, depth, features, K, pose):
        """Create Gaussians from predicted depth and features."""
        pass

    def to_gaussian_model(self):
        """Convert to standard GaussianModel for rendering."""
        pass
```

**Visualizations**:
- Side-by-side: random init vs pixel-aligned Gaussians
- Back-projection diagram
- Gaussian density map comparison
- 3D visualization of pixel-aligned point cloud

---

### Notebook 03: MVSplat Architecture Deep Dive

**Goal**: Full understanding of MVSplat's architecture

**Content**:
1. **Feature extraction (U-Net encoder)**
   - Multi-scale feature pyramid
   - Shared weights across views

2. **Cross-view interaction**
   - Cost Volume as cross-view feature
   - vs Transformer cross-attention (pixelSplat)

3. **Cost Volume processing (3D U-Net)**
   - Spatial + depth aggregation
   - Output: processed geometric features

4. **Gaussian prediction heads**
   - Depth head (from cost volume)
   - Covariance head (scale + rotation)
   - Opacity head
   - Color handling (direct or SH)

5. **Differentiable rendering for training**
   - Using gsplat / diff-gaussian-rasterization
   - Photometric loss on novel views

6. **Complete forward pass walkthrough**
   - Step-by-step data flow
   - Tensor shapes at each stage

**Code Implementation**:
```python
class SimplifiedMVSplat(nn.Module):
    """Simplified MVSplat for educational purposes."""

    def __init__(self, ...):
        self.feature_encoder = UNetEncoder(...)
        self.cost_volume_builder = CostVolumeBuilder(...)
        self.cost_volume_processor = CostVolumeProcessor(...)
        self.gaussian_heads = GaussianPredictionHeads(...)

    def forward(self, images, cameras):
        """Full MVSplat forward pass."""
        pass
```

**Visualizations**:
- Complete architecture diagram with tensor shapes
- Feature map visualization at each stage
- Cost Volume intermediate states
- Predicted Gaussian parameters visualization

---

### Notebook 04: pixelSplat & Implicit Geometry

**Goal**: Understand pixelSplat and compare with MVSplat

**Content**:
1. **pixelSplat motivation**
   - Learning geometry without explicit Cost Volume
   - End-to-end feature learning

2. **Architecture overview**
   - CNN backbone (EfficientNet / ResNet)
   - Cross-view attention mechanism
   - Direct Gaussian parameter regression

3. **Epipolar attention**
   - Attention along epipolar lines
   - Efficient cross-view reasoning
   - vs full Cost Volume

4. **Implicit depth learning**
   - Network learns depth without explicit supervision
   - Photometric supervision only
   - Challenges: ambiguity in textureless regions

5. **Detailed comparison with MVSplat**
   - Explicit vs implicit geometry
   - Speed vs accuracy trade-offs
   - Generalization behavior
   - Failure modes

**Code Implementation**:
```python
class SimplifiedPixelSplat(nn.Module):
    """Simplified pixelSplat for educational purposes."""

    def __init__(self, ...):
        self.backbone = ResNetBackbone(...)
        self.cross_attention = EpipolarAttention(...)
        self.gaussian_decoder = GaussianDecoder(...)

    def forward(self, images, cameras):
        """pixelSplat forward pass without explicit cost volume."""
        pass
```

**Visualizations**:
- Architecture comparison diagram (MVSplat vs pixelSplat)
- Attention map visualization
- Depth prediction comparison
- Rendering quality comparison

---

### Notebook 05: Feed-forward Training & Loss Design

**Goal**: Understand training strategies for feed-forward 3DGS

**Content**:
1. **Training data pipeline**
   - Dataset formats (RE10K, ACID, DL3DV)
   - View sampling strategies
   - Data augmentation

2. **Loss functions**
   - Photometric loss: L1 + SSIM
   - Depth supervision (optional)
   - Perceptual loss (LPIPS)

3. **Training strategies**
   - Multi-scale training
   - Curriculum learning (easy → hard views)
   - Handling variable number of input views

4. **Evaluation metrics**
   - PSNR, SSIM, LPIPS
   - Depth accuracy (if GT available)
   - Inference speed (FPS)

5. **Supervised vs self-supervised**
   - With depth GT (ScanNet, Hypersim)
   - Without depth GT (RE10K, video-based)

**Code Implementation**:
```python
class FeedForwardTrainer:
    """Training loop for feed-forward Gaussian methods."""

    def train_step(self, batch):
        """Single training step with loss computation."""
        pass

    def evaluate(self, test_loader):
        """Evaluation with PSNR/SSIM/LPIPS."""
        pass
```

---

### Notebook 06: MVSplat Code Walkthrough

**Goal**: Navigate and understand the official MVSplat codebase

**Content**:
1. **Repository structure**
   - configs/, src/, scripts/
   - Key files and their roles

2. **Model definition walkthrough**
   - Encoder architecture
   - Cost Volume implementation details
   - Gaussian prediction heads

3. **Data loading pipeline**
   - RE10K / ACID dataset loaders
   - View pair sampling
   - Camera parameter handling

4. **Training script analysis**
   - PyTorch Lightning integration
   - Hyperparameters and schedules
   - Logging and checkpointing

5. **Inference script analysis**
   - Loading pretrained models
   - Running on custom image pairs
   - Output format and visualization

---

### Notebook 07: Inference & Evaluation

**Goal**: Hands-on running and evaluating feed-forward methods

**Content**:
1. **Environment setup for MVSplat/pixelSplat**
   - Dependencies installation
   - Pretrained model download

2. **Running inference**
   - On RE10K test set
   - On custom image pairs
   - On video sequences

3. **Quantitative evaluation**
   - Computing PSNR/SSIM/LPIPS
   - Benchmark comparisons
   - Speed benchmarking

4. **Qualitative analysis**
   - Rendering quality inspection
   - Failure case analysis
   - Depth visualization

---

### Notebook 08: DepthSplat & 2025 Advances

**Goal**: Survey recent advances beyond MVSplat/pixelSplat

**Content**:
1. **DepthSplat**
   - Monocular depth prior (DPT, Depth Anything)
   - Integration with Cost Volume
   - Performance improvements

2. **Splatt3R / Flash3D**
   - Single-image 3D reconstruction
   - Foundation model integration

3. **Multi-view scaling**
   - From 2 views to N views
   - Efficiency challenges
   - Connection to VGGT (Phase 5)

4. **Trends and future directions**
   - Video-based feed-forward 3DGS
   - Dynamic scene handling
   - Integration with language models

---

### Notebook 09: MVSplat vs pixelSplat Comprehensive Comparison

**Goal**: Systematic experimental comparison

**Content**:
1. **Side-by-side architecture comparison**
2. **Quantitative benchmark**
   - RE10K, ACID, DTU
   - Speed comparison
3. **Qualitative comparison**
   - Indoor vs outdoor scenes
   - Sparse vs dense input views
   - Textureless regions
4. **When to use which method**
   - Decision tree / guidelines
5. **Summary and conclusions**

---

## 3. Source Code Modules

### src/feedforward/

```
src/feedforward/
├── __init__.py
├── cost_volume.py        # Cost Volume and Plane Sweeping
├── pixel_aligned.py      # Pixel-aligned Gaussian utilities
├── feature_encoder.py    # Feature extraction backbone
├── gaussian_predictor.py # Gaussian parameter prediction heads
├── depth_utils.py        # Depth processing utilities
└── cross_view.py         # Cross-view attention / interaction
```

### Key Classes

#### CostVolumeBuilder
```python
class CostVolumeBuilder(nn.Module):
    """Builds cost volume via plane sweeping."""

    def __init__(self, num_depths: int = 64,
                 min_depth: float = 0.5,
                 max_depth: float = 100.0,
                 sampling: str = 'log_uniform'):
        pass

    def build(self, feat_ref: torch.Tensor,
              feat_src: torch.Tensor,
              K_ref: torch.Tensor,
              K_src: torch.Tensor,
              T_src_ref: torch.Tensor) -> torch.Tensor:
        """Build cost volume. Returns [B, C, D, H, W]."""
        pass

    def warp_features(self, feat_src: torch.Tensor,
                      depth: float,
                      K_ref: torch.Tensor,
                      K_src: torch.Tensor,
                      T_src_ref: torch.Tensor) -> torch.Tensor:
        """Warp source features to reference at given depth."""
        pass
```

#### PixelAlignedGaussianPredictor
```python
class PixelAlignedGaussianPredictor(nn.Module):
    """Predict Gaussian parameters from features and depth."""

    def __init__(self, feature_dim: int = 64):
        pass

    def forward(self, features: torch.Tensor,
                depth: torch.Tensor,
                K: torch.Tensor) -> dict:
        """
        Returns dict with:
            - positions: [B, N, 3]
            - covariances: [B, N, 3, 3]
            - opacities: [B, N, 1]
            - colors: [B, N, 3]
        """
        pass

    def unproject_depth(self, depth: torch.Tensor,
                        K: torch.Tensor) -> torch.Tensor:
        """Unproject depth map to 3D points."""
        pass
```

#### CrossViewInteraction
```python
class CrossViewInteraction(nn.Module):
    """Cross-view feature interaction (attention or cost volume)."""

    def __init__(self, mode: str = 'cost_volume',
                 feature_dim: int = 64):
        pass

    def forward(self, feat1: torch.Tensor,
                feat2: torch.Tensor,
                cameras: dict) -> torch.Tensor:
        """Cross-view feature aggregation."""
        pass
```

---

## 4. Dataset Requirements

### Supported Datasets

1. **RealEstate10K (RE10K)** - Primary benchmark
   - Indoor/outdoor scenes from YouTube videos
   - Camera poses from COLMAP
   - Download: https://google.github.io/realestate10k/

2. **ACID** - Outdoor aerial scenes
   - Aerial photographs
   - Structure-from-Motion poses
   - Download: https://infinite-nature.github.io/

3. **DTU** - Object-centric scenes
   - Controlled multi-view captures
   - Ground truth depth
   - Download: https://roboimagedata.compute.dtu.dk/

4. **Custom Pairs**
   ```
   custom_data/
   ├── images/
   │   ├── view_00.jpg
   │   ├── view_01.jpg
   │   └── ...
   └── cameras.json  # Optional camera parameters
   ```

---

## 5. Configuration

### Default Configuration (configs/feedforward_default.yaml)

```yaml
# Model
model:
  type: "mvsplat"  # or "pixelsplat"
  feature_dim: 64
  num_depth_planes: 64
  min_depth: 0.5
  max_depth: 100.0
  depth_sampling: "log_uniform"

# Gaussian prediction
gaussian:
  predict_sh: false  # Use image color directly
  covariance_type: "2d"  # "2d" or "3d"
  opacity_activation: "sigmoid"

# Training
training:
  learning_rate: 1.0e-4
  batch_size: 4
  num_input_views: 2
  loss:
    rgb_weight: 1.0
    ssim_weight: 0.2
    depth_weight: 0.0  # Set >0 if depth GT available

# Evaluation
evaluation:
  metrics: ["psnr", "ssim", "lpips"]
  num_novel_views: 5

# System
system:
  device: "cuda"
  image_height: 256
  image_width: 256
```

---

## 6. Dependencies

```bash
# Core (same as Phase 1)
torch>=2.0.0
numpy>=1.24.0
opencv-python>=4.8.0

# Feed-forward specific
timm>=0.9.0              # Vision backbone models
einops>=0.7.0             # Tensor operations
kornia>=0.7.0             # Differentiable geometry

# Rendering
gsplat>=0.1.0             # Efficient Gaussian rendering
# OR diff-gaussian-rasterization from Phase 1

# Evaluation
lpips>=0.1.4              # Perceptual metric
scikit-image>=0.21.0      # SSIM computation

# Visualization
plotly>=5.15.0
matplotlib>=3.7.0
```

---

## 7. Development Plan

### Stage 1: Foundation (Notebooks 00-02)
- [ ] Create notebooks/phase4/ directory
- [ ] Notebook 00: Phase 4 Overview & Paradigm Shift
- [ ] Notebook 01: Cost Volume & Plane Sweeping
- [ ] Notebook 02: Pixel-aligned Gaussian Representation
- [ ] Implement src/feedforward/cost_volume.py
- [ ] Implement src/feedforward/pixel_aligned.py

### Stage 2: Core Architectures (Notebooks 03-05)
- [ ] Notebook 03: MVSplat Architecture Deep Dive
- [ ] Notebook 04: pixelSplat & Implicit Geometry
- [ ] Notebook 05: Feed-forward Training & Loss Design
- [ ] Implement src/feedforward/feature_encoder.py
- [ ] Implement src/feedforward/gaussian_predictor.py
- [ ] Implement src/feedforward/cross_view.py

### Stage 3: Practical (Notebooks 06-07)
- [ ] Notebook 06: MVSplat Code Walkthrough
- [ ] Notebook 07: Inference & Evaluation
- [ ] Implement src/feedforward/depth_utils.py

### Stage 4: Advanced (Notebooks 08-09)
- [ ] Notebook 08: DepthSplat & 2025 Advances
- [ ] Notebook 09: MVSplat vs pixelSplat Comparison
- [ ] Full testing and integration

---

## 8. Success Criteria

By the end of Phase 4:

1. **Conceptual**: Learners can explain the feed-forward 3DGS paradigm and its advantages
2. **Technical**: Understand Cost Volume, pixel-aligned Gaussians, and architecture details
3. **Practical**: Can run MVSplat/pixelSplat inference on custom images
4. **Comparative**: Can articulate when to use optimization vs feed-forward approaches
5. **Research**: Aware of 2025 trends (DepthSplat, scaling to many views)

---

## 9. References

1. MVSplat: https://donydchen.github.io/mvsplat/
2. pixelSplat: https://davidcharatan.com/pixelsplat/
3. DepthSplat: https://arxiv.org/abs/2412.18010
4. gsplat: https://docs.gsplat.studio/
5. RealEstate10K: https://google.github.io/realestate10k/
6. Phase 1 Development Spec: `docs/Phase1_Development_Spec.md`
7. Phase 4 Tutorial Outline: `tutorials/Phase4_FeedForward_Gaussian.md`
