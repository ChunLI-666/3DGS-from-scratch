# Development Roadmap

> This document tracks the development progress and plans for all phases.

---

## Phase 1: 3DGS Foundation - Development Status

### Completed (Week 1-2)

| Task | Status | Notes |
|------|--------|-------|
| Repository structure | Done | Created complete directory layout |
| src/gaussian module | Done | gaussian_model.py, covariance.py, projection.py |
| src/visualization module | Done | gaussian_viz.py, training_viz.py, interactive_plots.py |
| src/rendering module | Done | rasterizer.py, alpha_blending.py, differentiable_render.py |
| Environment scripts | Done | setup_env.sh, install_official_3dgs.sh |
| Notebook 00: Environment Setup | Done | Full environment check and setup |
| Notebook 01: Gaussian Basics | Done | 1D/2D/3D Gaussians with visualization |
| Notebook 02: 3D Gaussian Math | Done | Covariance decomposition, quaternions, Gaussian3D class |
| Notebook 03: Projection & Splatting | Done | Camera models, projection Jacobian, splatting pipeline |
| Notebook 04: Differentiable Rendering | Done | Gradient flow, optimization demos, loss functions |
| Notebook 05: Alpha Blending | Done | Volume rendering, transmittance, front-to-back blending |

### In Progress (Week 3)

| Task | Status | Priority |
|------|--------|----------|
| Notebook 06: Spherical Harmonics | Pending | High |
| Notebook 07: Adaptive Density Control | Pending | High |
| Notebook 08: Training Pipeline | Pending | High |

### Planned (Week 4-6)

| Task | Status | Priority |
|------|--------|----------|
| Notebook 09: Official Code Walkthrough | Pending | Medium |
| Notebook 10: Custom Data Training | Pending | Medium |
| src/spherical_harmonics module | Pending | Medium |
| src/training module | Pending | High |
| Docker configuration | Pending | Low |
| CI/CD for notebook testing | Pending | Low |

---

## Phase 2: 3DGS + SLAM - Task Breakdown

### Notebooks to Create

| Notebook | Topic | Key Concepts |
|----------|-------|--------------|
| 00 | Overview | SLAM + 3DGS integration paradigm |
| 01 | SLAM Basics Review | VO, BA, loop closure |
| 02 | SplaTAM Architecture | Tracking + Mapping with Gaussians |
| 03 | Gaussian Map Initialization | From SLAM keyframes |
| 04 | Online Optimization | Incremental training |
| 05 | Keyframe Management | Selection and pruning |
| 06 | SplaTAM Code Walkthrough | Official implementation |
| 07 | GS-SLAM Comparison | Alternative approach |
| 08 | Custom Dataset SLAM | Running on your own data |

### Repositories to Integrate

- [ ] SplaTAM: https://github.com/spla-tam/SplaTAM
- [ ] GS-SLAM: (when available)
- [ ] MonoGS: https://github.com/muskie82/MonoGS

### Key Source Modules

- `src/slam/`: SLAM-specific utilities
  - `keyframe.py`: Keyframe selection
  - `tracking.py`: Camera tracking
  - `mapping.py`: Gaussian map management

---

## Phase 3: DUSt3R - Task Breakdown

### Notebooks to Create

| Notebook | Topic | Key Concepts |
|----------|-------|--------------|
| 00 | Overview | Pose-free reconstruction paradigm |
| 01 | ViT Foundations | Vision Transformer basics |
| 02 | Pointmap Representation | DUSt3R's output format |
| 03 | Global Alignment | Multi-view consistency |
| 04 | DUSt3R Architecture | Encoder-decoder structure |
| 05 | MASt3R Extension | Matching capabilities |
| 06 | Code Walkthrough | Official implementation |
| 07 | Custom Data | Running on your own images |

### Repositories to Integrate

- [ ] DUSt3R: https://github.com/naver/dust3r
- [ ] MASt3R: https://github.com/naver/mast3r

---

## Phase 4: Feed-forward Gaussian - Task Breakdown

### Notebooks to Create

| Notebook | Topic | Key Concepts |
|----------|-------|--------------|
| 00 | Overview | From optimization to prediction |
| 01 | Cost Volume Basics | MVS foundations |
| 02 | Epipolar Geometry | Geometric constraints |
| 03 | pixelSplat Architecture | Epipolar-based prediction |
| 04 | MVSplat Architecture | Cost volume approach |
| 05 | Comparison Study | pixelSplat vs MVSplat |
| 06 | Code Walkthrough | Both implementations |
| 07 | DepthSplat | Latest advances |

### Repositories to Integrate

- [ ] pixelSplat: https://github.com/dcharatan/pixelsplat
- [ ] MVSplat: https://github.com/donydchen/mvsplat
- [ ] DepthSplat: (when available)

---

## Phase 5: VGGT - Task Breakdown

### Notebooks to Create

| Notebook | Topic | Key Concepts |
|----------|-------|--------------|
| 00 | Overview | Foundation model for 3D vision |
| 01 | Architecture | Multi-head prediction |
| 02 | Training Strategy | Large-scale learning |
| 03 | Inference | Using VGGT for various tasks |
| 04 | Fast3R Comparison | Alternative approach |
| 05 | Integration with Gaussians | Combining with 3DGS |

### Repositories to Integrate

- [ ] VGGT: (official when available)
- [ ] Fast3R: https://github.com/facebookresearch/fast3r

---

## Phase 6: Frontier Research - Task Breakdown

### Topics to Cover

| Topic | Content |
|-------|---------|
| Dynamic Scenes | 4D Gaussian Splatting |
| Large Scale | City-scale reconstruction |
| Generative | Gaussian generation models |
| Compression | Efficient storage and streaming |
| Applications | Robotics, AR/VR, autonomous driving |

---

## Development Priorities

### Immediate Next Steps (This Week)

1. **Notebook 06: Spherical Harmonics**
   - SH basis functions
   - View-dependent color encoding
   - Implementation in PyTorch

2. **Notebook 07: Adaptive Density Control**
   - Gaussian splitting and cloning
   - Pruning low-opacity Gaussians
   - Densification strategies

3. **Notebook 08: Training Pipeline**
   - Complete training loop
   - Loss computation
   - Optimization scheduling

### Medium-term Goals (Next 2 Weeks)

1. Complete Phase 1 notebooks (06-10)
2. Implement spherical_harmonics module
3. Implement training module
4. Add Docker configuration
5. Set up CI/CD for testing

### Long-term Goals (Next Month)

1. Begin Phase 2 development
2. Integrate official 3DGS repository
3. Create video tutorials
4. Build community contribution guidelines

---

## Notes

- Development follows the spec in `docs/Phase1_Development_Spec.md`
- Each notebook should be testable in Google Colab
- Code should be educational first, optimized second
- Visualizations are critical for understanding
