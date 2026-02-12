# Changelog

All notable changes to this project will be documented in this file.

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
