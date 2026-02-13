# Changelog

All notable changes to this project will be documented in this file.

## [0.8.0] - 2026-02-13

### Added

#### Notebooks (Phase 4 Complete - Notebooks 06-09)
- `notebooks/phase4/06_mvsplat_code_walkthrough.ipynb`: MVSplat Code Walkthrough
  - Official MVSplat repository structure analysis
  - Model definition walkthrough (encoder_mvsplat.py, depth_predictor_multiview.py)
  - Gaussian activation functions (sigmoid, exp, normalize) with code
  - Data loading pipeline (RE10K format, camera parameter parsing)
  - View sampling strategy (50-90% overlap) with visualization
  - Training script analysis (PyTorch Lightning + Hydra config)
  - Official training hyperparameters (AdamW, lr=1.5e-4, 300K steps)
  - Inference pipeline timing breakdown (~25ms, 40 FPS)
  - Simplified vs official implementation comparison table
- `notebooks/phase4/07_inference_evaluation.ipynb`: Inference & Evaluation
  - Complete inference pipeline (preprocess → forward → Gaussians → render)
  - Image quality metrics implementation (PSNR, SSIM, approximate LPIPS)
  - Published benchmark results comparison (pixelSplat, MVSplat, DepthSplat)
  - Speed benchmarking at multiple resolutions (32x32 to 256x256)
  - Failure case analysis (textureless, repetitive, reflections, wide baseline)
  - Depth visualization and Gaussian quality assessment
- `notebooks/phase4/08_depthsplat_2025_advances.ipynb`: DepthSplat & 2025 Advances
  - Monocular depth prior revolution (MiDaS → DPT → Depth Anything V2)
  - DepthSplat architecture: depth-guided adaptive plane sampling
  - Depth prior encoder and feature fusion implementation
  - Scale-shift alignment for monocular depth integration
  - Single-image 3D methods survey (Splatt3R, Flash3D, LGM, GRM)
  - Multi-view scaling analysis (O(N) to O(N²) complexity)
  - Connection to VGGT (Phase 5 preview)
  - 2025 trends: video-based, dynamic scenes, language-guided 3DGS
  - Comprehensive 2024-2025 methods benchmark table
- `notebooks/phase4/09_mvsplat_vs_pixelsplat_comparison.ipynb`: Comprehensive Comparison
  - Side-by-side architecture diagrams (Cost Volume vs Epipolar Attention)
  - Detailed component comparison table (9 dimensions)
  - Computational complexity analysis (FLOPs, memory, speed scaling)
  - Quantitative benchmarks across RE10K, ACID, DTU datasets
  - Performance gap analysis with statistical breakdown
  - Qualitative comparison by scene type (6 categories with scoring)
  - Failure mode comparison (method-specific vs shared)
  - Decision tree for method selection
  - Practical selection checklist
  - Experimental side-by-side with simplified models
  - Phase 4 complete knowledge map and summary

### Changed
- Updated `DEVELOPMENT_ROADMAP.md`: Phase 4 marked as complete (all 10 notebooks done)
- Updated `CHANGELOG.md` with Phase 4 completion

---

## [0.7.0] - 2026-02-13

### Added

#### Notebooks (Phase 4 Core - Notebooks 01-05)
- `notebooks/phase4/01_cost_volume_plane_sweeping.ipynb`: Cost Volume & Plane Sweeping
  - Multi-View Stereo (MVS) fundamentals and stereo matching
  - Plane Sweeping algorithm with depth hypothesis sampling
  - Homography warping step-by-step implementation
  - Cost Volume construction and visualization
  - Soft argmin depth regression with temperature analysis
  - Integration with `src/feedforward.CostVolumeBuilder`
- `notebooks/phase4/02_pixel_aligned_gaussians.ipynb`: Pixel-aligned Gaussian Representation
  - Free-form vs pixel-aligned Gaussian comparison
  - Depth-to-3D back-projection implementation
  - Creating pixel-aligned Gaussians from predicted depth and features
  - Multi-view Gaussian merging with `PixelAlignedGaussians`
  - Connection to surfel mapping in SLAM
- `notebooks/phase4/03_mvsplat_architecture.ipynb`: MVSplat Architecture Deep Dive
  - Complete architecture diagram and data flow
  - Simplified U-Net feature encoder (shared Siamese weights)
  - Cost Volume processing with 3D CNN
  - Gaussian prediction heads (depth, scale, rotation, opacity)
  - Complete forward pass with tensor shape tracing
  - Multi-view Gaussian merging demonstration
  - Comparison with official MVSplat implementation
- `notebooks/phase4/04_pixelsplat_implicit_geometry.ipynb`: pixelSplat & Implicit Geometry
  - Explicit vs implicit geometry learning comparison
  - Epipolar geometry review and visualization
  - Epipolar cross-attention mechanism implementation
  - Attention map visualization and interpretation
  - Complete SimplifiedPixelSplat architecture
  - Computational cost comparison (MVSplat vs pixelSplat)
  - Strengths/weaknesses analysis and when-to-use guide
- `notebooks/phase4/05_training_loss_design.ipynb`: Feed-forward Training & Loss Design
  - Training paradigm comparison (per-scene vs feed-forward)
  - Synthetic dataset implementation (RE10K-style)
  - L1, SSIM, LPIPS loss functions with educational implementations
  - Combined loss with configurable weights
  - Complete training loop with toy model
  - Evaluation metrics (PSNR, SSIM, LPIPS) with benchmarks
  - Learning rate schedules (warmup + cosine decay)
  - Supervised vs self-supervised training discussion

### Changed
- Updated `DEVELOPMENT_ROADMAP.md` with notebooks 01-05 completion
- Updated `CHANGELOG.md` with Phase 4 progress

---

## [0.6.0] - 2026-02-12

### Added

#### Phase 4: Feed-forward Gaussian Splatting
- `docs/Phase4_Development_Spec.md`: Complete development specification
  - 10 notebooks planned (00-09)
  - Learning objectives: MVSplat, pixelSplat, DepthSplat
  - Source module specification (src/feedforward)
  - Dataset and configuration specifications

#### Notebooks (Phase 4 Start)
- `notebooks/phase4/00_phase4_overview.ipynb`: Phase 4 Overview & Paradigm Shift
  - Optimization vs feed-forward paradigm comparison
  - Pixel-aligned Gaussian concept introduction
  - Method taxonomy (MVSplat, pixelSplat, DepthSplat)
  - Quality vs speed trade-off visualization
  - Toy Gaussian predictor demo (PyTorch)
  - Back-projection from depth demonstration
  - Training loop explanation
  - Connection to Phase 1-3 concepts

#### Source Code Modules
- `src/feedforward/`: Feed-forward Gaussian Splatting module
  - `cost_volume.py`: Cost Volume and Plane Sweeping implementation
    - `PlaneSweeper`: Plane sweeping with homography warping
    - `CostVolumeBuilder`: Multi-view cost volume aggregation
    - `depth_regression_softargmin()`: Differentiable depth regression
    - `homography_warp()`: Feature warping via homography
    - `create_depth_planes()`: Depth hypothesis sampling
  - `pixel_aligned.py`: Pixel-aligned Gaussian representation
    - `PixelAlignedGaussians`: Structured Gaussian container
    - `unproject_depth_to_3d()`: Depth map back-projection
    - `create_pixel_grid()`: Pixel coordinate grid utility
  - `gaussian_predictor.py`: Neural prediction heads
    - `GaussianPredictionHeads`: Combined prediction module
    - `DepthHead`: Per-pixel depth prediction
    - `CovarianceHead`: Scale and rotation prediction (2D/3D modes)
    - `OpacityHead`: Opacity prediction with sigmoid activation

### Changed
- Updated `DEVELOPMENT_ROADMAP.md` with Phase 4 progress
- Updated `CHANGELOG.md` with Phase 4 additions

---

## [0.5.0] - 2026-02-12

### Added

#### Notebooks (Phase 2 Progress)
- `notebooks/phase2/01_slam_basics.ipynb`: SLAM Fundamentals Review
  - Visual Odometry (VO) concepts and implementation
  - Feature-based vs Direct methods comparison
  - Camera pose representation (SE(3), quaternions, 6D rotation)
  - Bundle Adjustment demonstration with PyTorch
  - Loop closure detection and correction
  - Keyframe selection strategies
  - Covisibility graph construction
- `notebooks/phase2/02_splatam_architecture.ipynb`: SplaTAM System Architecture
  - Complete system architecture overview
  - Configuration classes (TrackingConfig, MappingConfig, KeyframeConfig)
  - GaussianMap class for SLAM
  - CameraTracker with 6D+3D pose parameterization
  - GaussianMapper with depth unprojection
  - Silhouette-guided densification concept
  - Complete SplaTAMSystem integration
- `notebooks/phase2/03_map_initialization.ipynb`: Gaussian Map Initialization
  - Camera model review (projection/unprojection)
  - Depth-guided Gaussian placement
  - Scale estimation (basic and adaptive)
  - RGB to SH color conversion
  - Opacity initialization (logit parameterization)
  - Complete GaussianInitializer class
  - RGB-D vs monocular initialization comparison

### Changed
- Updated `DEVELOPMENT_ROADMAP.md` with Phase 2 progress

---

## [0.4.0] - 2026-02-12

### Added

#### Notebooks (Phase 1 Complete)
- `09_official_code_walkthrough.ipynb`: Official 3DGS code analysis
  - Repository structure overview
  - GaussianModel class analysis
  - CUDA rasterizer architecture (tile-based rendering)
  - Forward and backward pass explanation
  - Training loop walkthrough (train.py)
  - COLMAP integration details
  - Optimizer configuration
  - Densification implementation
  - Rendering pipeline
  - Key files summary
- `10_custom_data_training.ipynb`: Training on custom data
  - Image capture guidelines and best practices
  - Dataset directory structure
  - COLMAP processing pipeline
  - Using official convert.py
  - Training commands and parameters
  - Rendering trained models
  - Evaluation metrics (PSNR, SSIM, LPIPS)
  - Troubleshooting guide
  - Complete workflow example

### Changed
- Updated `DEVELOPMENT_ROADMAP.md` marking Phase 1 as complete
- Phase 2 development now prioritized

---

## [0.3.0] - 2026-02-12

### Added

#### Notebooks (Phase 1 Complete)
- `06_spherical_harmonics.ipynb`: Spherical Harmonics for view-dependent color
  - SH basis function visualization (degrees 0-3)
  - SH evaluation implementation in PyTorch
  - RGB to SH and SH to RGB conversion
  - View direction computation
  - SH coefficient optimization demo
  - Degree trade-offs analysis
- `07_adaptive_density_control.ipynb`: Adaptive Gaussian densification
  - Gradient-based densification criterion
  - Gaussian splitting for large Gaussians
  - Gaussian cloning for small Gaussians
  - Pruning based on opacity and scale
  - Complete DensificationController class
  - Official 3DGS schedule explanation
- `08_training_pipeline.ipynb`: Complete training pipeline
  - GaussianModel class with all parameters
  - Simple2DRenderer for demonstration
  - Loss functions (L1, SSIM, D-SSIM)
  - Learning rate scheduling
  - Complete training loop implementation
  - Training on synthetic target images

#### Source Code Modules
- `src/spherical_harmonics/sh_utils.py`
  - SH constants (degrees 0-3)
  - `eval_sh()`: Evaluate SH at view directions
  - `rgb_to_sh()` / `sh_to_rgb()`: Conversion utilities
  - `initialize_sh_from_rgb()`: Initialize SH from colors
  - `get_direction_from_camera()`: Compute view directions
  - `visualize_sh_basis()`: Generate visualization data
  - `SphericalHarmonicsEncoder`: nn.Module wrapper

### Changed
- Updated `DEVELOPMENT_ROADMAP.md` with completed Phase 1 core notebooks

---

## [0.2.0] - 2026-02-12

### Added

#### Notebooks (Phase 1 Core)
- `02_3d_gaussian_math.ipynb`: 3D Gaussian ellipsoid mathematics
  - Covariance matrix decomposition: Σ = RSS^TR^T
  - Quaternion to rotation matrix conversion
  - Complete Gaussian3D class with gradient support
  - Interactive 3D visualization
- `03_projection_splatting.ipynb`: Projection and splatting
  - Pinhole camera model and intrinsic/extrinsic parameters
  - 3D to 2D point projection
  - Projection Jacobian derivation and implementation
  - Covariance projection for Gaussian splatting
  - Complete splatting pipeline
- `04_differentiable_rendering.ipynb`: Differentiable rendering
  - Why differentiable rendering is essential for 3DGS
  - Gradient flow through rendering pipeline
  - Single and multiple Gaussian optimization demos
  - Loss functions (MSE, SSIM, D-SSIM)
- `05_alpha_blending.ipynb`: Alpha blending and volume rendering
  - Volume rendering equation
  - Transmittance computation
  - Front-to-back alpha blending
  - Early termination for efficiency
  - Gradient computation through blending

#### Source Code Modules
- `src/rendering/`
  - `rasterizer.py`: SimpleRasterizer and TileBasedRasterizer classes
  - `alpha_blending.py`: Front-to-back alpha compositing
  - `differentiable_render.py`: DifferentiableRenderer and GaussianRenderer classes

### Changed
- Updated `DEVELOPMENT_ROADMAP.md` with completed tasks

---

## [0.1.0] - 2026-02-12

### Added

#### Repository Structure
- Created complete directory structure for 3DGS tutorials
- Added `notebooks/phase1/` for Phase 1 interactive tutorials
- Added `src/` for educational source code modules
- Added `scripts/` for setup and deployment scripts
- Added `configs/` for configuration files
- Added `docs/` for documentation

#### Source Code Modules
- `src/gaussian/`
  - `gaussian_model.py`: Gaussian3D and GaussianModel classes
  - `covariance.py`: Covariance matrix utilities (quaternion, rotation, decomposition)
  - `projection.py`: 3D to 2D projection functions
- `src/visualization/`
  - `gaussian_viz.py`: 2D/3D Gaussian visualization tools
  - `training_viz.py`: Training progress visualization
  - `interactive_plots.py`: Interactive Jupyter widgets
- `src/rendering/`: Placeholder for rendering modules
- `src/spherical_harmonics/`: Placeholder for SH utilities
- `src/training/`: Placeholder for training utilities

#### Notebooks
- `00_environment_setup.ipynb`: Environment detection and setup verification
- `01_gaussian_basics.ipynb`: Introduction to 1D, 2D, and 3D Gaussian distributions

#### Scripts
- `setup_env.sh`: One-click environment setup
- `download_sample_data.sh`: Sample data download
- `install_official_3dgs.sh`: Official 3DGS installation

#### Documentation
- `README.md`: Project overview and quick start guide
- `docs/Phase1_Development_Spec.md`: Detailed development specification

### Infrastructure
- Added `requirements.txt` for Python dependencies
- Added `setup.py` for package installation
- Updated `.gitignore` for Python/Jupyter/3DGS files
- Added `configs/default.yaml` for default training configuration
