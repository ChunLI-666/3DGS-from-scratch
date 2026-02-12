# Changelog

All notable changes to this project will be documented in this file.

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
