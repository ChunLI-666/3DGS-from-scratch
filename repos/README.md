# Repos Directory

This directory is for storing cloned paper implementations as git submodules.

## Included Repositories

### Phase 1: 3DGS Foundation
- `gaussian-splatting/` - Official 3D Gaussian Splatting implementation
  - Source: https://github.com/graphdeco-inria/gaussian-splatting

### Phase 2: 3DGS + SLAM (Future)
- `SplaTAM/` - SplaTAM implementation
- `GS-SLAM/` - GS-SLAM implementation

## Installation

To set up the official 3DGS repository:

```bash
cd scripts
bash install_official_3dgs.sh
```

Or manually:

```bash
git submodule add https://github.com/graphdeco-inria/gaussian-splatting.git repos/gaussian-splatting
git submodule update --init --recursive
```
