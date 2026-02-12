# 3DGS from Scratch

> A comprehensive, hands-on tutorial series for learning 3D Gaussian Splatting from the ground up.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ChunLI-666/3DGS-from-scratch/blob/develop/notebooks/phase1/00_environment_setup.ipynb)

## Overview

This repository provides interactive Jupyter notebooks that guide you through understanding and implementing 3D Gaussian Splatting (3DGS). Designed for learners with a SLAM background, it bridges traditional geometric methods with modern neural rendering techniques.

## Features

- **Interactive Notebooks**: Step-by-step tutorials with runnable code
- **Rich Visualizations**: 3D interactive plots to understand abstract concepts
- **Modular Design**: Clean, educational code implementations
- **Official Code Integration**: One-click deployment of the original 3DGS implementation
- **Docker Support**: Containerized environment for easy setup

## Quick Start

### Option 1: Google Colab (Recommended for beginners)

Click the "Open in Colab" badge above to start immediately.

### Option 2: Local Setup

```bash
# Clone the repository
git clone https://github.com/ChunLI-666/3DGS-from-scratch.git
cd 3DGS-from-scratch

# Create conda environment
conda create -n 3dgs-tutorial python=3.10 -y
conda activate 3dgs-tutorial

# Install dependencies
pip install -r requirements.txt

# Start Jupyter Lab
jupyter lab
```

### Option 3: Docker

```bash
docker-compose up -d
# Open http://localhost:8888 in your browser
```

## Tutorial Structure

### Phase 1: 3DGS Foundation

| Notebook | Topic | Duration |
|----------|-------|----------|
| [00_environment_setup](notebooks/phase1/00_environment_setup.ipynb) | Environment Configuration | 30 min |
| [01_gaussian_basics](notebooks/phase1/01_gaussian_basics.ipynb) | Gaussian Distribution Basics | 45 min |
| [02_3d_gaussian_math](notebooks/phase1/02_3d_gaussian_math.ipynb) | 3D Gaussian Ellipsoid Math | 60 min |
| [03_projection_splatting](notebooks/phase1/03_projection_splatting.ipynb) | Projection & Splatting | 60 min |
| [04_differentiable_rendering](notebooks/phase1/04_differentiable_rendering.ipynb) | Differentiable Rendering | 90 min |
| [05_alpha_blending](notebooks/phase1/05_alpha_blending.ipynb) | Alpha Blending & Depth Sorting | 60 min |
| [06_spherical_harmonics](notebooks/phase1/06_spherical_harmonics.ipynb) | Spherical Harmonics | 75 min |
| [07_adaptive_density](notebooks/phase1/07_adaptive_density.ipynb) | Adaptive Density Control | 60 min |
| [08_training_pipeline](notebooks/phase1/08_training_pipeline.ipynb) | Complete Training Pipeline | 90 min |
| [09_official_code_walkthrough](notebooks/phase1/09_official_code_walkthrough.ipynb) | Official Code Walkthrough | 120 min |
| [10_custom_data_training](notebooks/phase1/10_custom_data_training.ipynb) | Training on Custom Data | 90 min |

### Phase 2-6: Coming Soon

- Phase 2: 3DGS + SLAM Integration
- Phase 3: DUSt3R
- Phase 4: Feed-forward Gaussian Methods
- Phase 5: VGGT
- Phase 6: Frontier Research

## Repository Structure

```
3DGS-from-scratch/
├── notebooks/          # Interactive Jupyter tutorials
│   └── phase1/         # Phase 1 notebooks
├── src/                # Educational source code
│   ├── gaussian/       # Gaussian model implementations
│   ├── rendering/      # Rendering modules
│   ├── spherical_harmonics/
│   ├── training/       # Training utilities
│   └── visualization/  # Visualization tools
├── repos/              # Official paper implementations (submodules)
├── scripts/            # Setup and utility scripts
├── data/               # Sample data
├── configs/            # Configuration files
├── docker/             # Docker configuration
└── docs/               # Documentation
```

## Requirements

- Python 3.9+
- PyTorch 2.0+
- CUDA 11.7+ (for GPU acceleration)
- 8GB+ GPU memory (12GB+ recommended for training)

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](docs/CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) - Original implementation
- [INRIA GraphDeco](https://team.inria.fr/graphdeco/) - Research team behind 3DGS

## Citation

If you find this tutorial helpful, please star this repository and cite the original 3DGS paper:

```bibtex
@article{kerbl3Dgaussians,
  author    = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
  title     = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
  journal   = {ACM Transactions on Graphics},
  number    = {4},
  volume    = {42},
  month     = {July},
  year      = {2023},
  url       = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}
```
