# Development Roadmap

> This document tracks the development progress and plans for all phases.

---

## Phase 1: 3DGS Foundation - Development Status

### Completed (Phase 1 Complete)

| Task | Status | Notes |
|------|--------|-------|
| Repository structure | Done | Created complete directory layout |
| src/gaussian module | Done | gaussian_model.py, covariance.py, projection.py |
| src/visualization module | Done | gaussian_viz.py, training_viz.py, interactive_plots.py |
| src/rendering module | Done | rasterizer.py, alpha_blending.py, differentiable_render.py |
| src/spherical_harmonics module | Done | sh_utils.py with SH evaluation and utilities |
| Environment scripts | Done | setup_env.sh, install_official_3dgs.sh |
| Notebook 00: Environment Setup | Done | Full environment check and setup |
| Notebook 01: Gaussian Basics | Done | 1D/2D/3D Gaussians with visualization |
| Notebook 02: 3D Gaussian Math | Done | Covariance decomposition, quaternions, Gaussian3D class |
| Notebook 03: Projection & Splatting | Done | Camera models, projection Jacobian, splatting pipeline |
| Notebook 04: Differentiable Rendering | Done | Gradient flow, optimization demos, loss functions |
| Notebook 05: Alpha Blending | Done | Volume rendering, transmittance, front-to-back blending |
| Notebook 06: Spherical Harmonics | Done | SH basis functions, view-dependent color, optimization |
| Notebook 07: Adaptive Density Control | Done | Splitting, cloning, pruning, densification controller |
| Notebook 08: Training Pipeline | Done | Complete training loop, loss functions, LR scheduling |
| Notebook 09: Official Code Walkthrough | Done | Repository structure, CUDA rasterizer, COLMAP integration |
| Notebook 10: Custom Data Training | Done | Image capture, COLMAP, training, evaluation |

### Future Improvements

| Task | Status | Priority |
|------|--------|----------|
| src/training module | Pending | Medium |
| Docker configuration | Pending | Low |
| CI/CD for notebook testing | Pending | Low |
| Video tutorials | Pending | Low |

---

## Phase 2: 3DGS + SLAM - Task Breakdown

### Completed

| Task | Status | Notes |
|------|--------|-------|
| Notebook 00: Overview | Done | 3DGS + SLAM paradigm introduction |
| Notebook 01: SLAM Basics | Done | VO, BA, loop closure, keyframe selection |
| Notebook 02: SplaTAM Architecture | Done | Tracking + mapping system design |
| Notebook 03: Map Initialization | Done | Depth-guided Gaussian placement |
| src/slam module | Done | keyframe.py, tracking.py, mapping.py |

### In Progress

| Notebook | Topic | Key Concepts |
|----------|-------|--------------|
| 04 | Camera Tracking | Render-and-compare pose optimization |
| 05 | Online Optimization | Incremental Gaussian training |
| 06 | Keyframe Management | Selection and pruning strategies |
| 07 | SplaTAM Code Walkthrough | Official implementation analysis |
| 08 | MonoGS Comparison | Monocular approach |
| 09 | Custom Dataset SLAM | Running on your own data |

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

### Completed

| Task | Status | Notes |
|------|--------|-------|
| Development Spec | Done | docs/Phase4_Development_Spec.md |
| Notebook 00: Overview | Done | Paradigm shift, method landscape, toy demo |
| src/feedforward module | Done | cost_volume.py, pixel_aligned.py, gaussian_predictor.py |
| Notebook 01: Cost Volume | Done | MVS, plane sweeping, homography warp, soft argmin |
| Notebook 02: Pixel-aligned Gaussians | Done | Back-projection, structured layout, multi-view merge |
| Notebook 03: MVSplat Architecture | Done | U-Net encoder, 3D CNN, prediction heads, forward pass |
| Notebook 04: pixelSplat | Done | Epipolar cross-attention, implicit geometry, comparison |
| Notebook 05: Training & Loss | Done | L1+SSIM+LPIPS, training loop, evaluation metrics |
| Notebook 06: MVSplat Code Walkthrough | Done | Official repo structure, model definition, data pipeline, training analysis |
| Notebook 07: Inference & Evaluation | Done | Inference pipeline, PSNR/SSIM/LPIPS metrics, speed benchmarking, failure cases |
| Notebook 08: DepthSplat & 2025 Advances | Done | Depth priors (Depth Anything V2), adaptive planes, single-image 3D, multi-view scaling |
| Notebook 09: MVSplat vs pixelSplat | Done | Architecture comparison, benchmarks, qualitative analysis, decision framework |

### Phase 4 Complete ✓

### Key Source Modules

- `src/feedforward/`: Feed-forward 3DGS utilities
  - `cost_volume.py`: Plane sweeping, cost volume, soft argmin
  - `pixel_aligned.py`: Pixel-aligned Gaussian representation
  - `gaussian_predictor.py`: Neural prediction heads

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

### Immediate Next Steps

1. **Phase 2: 3DGS + SLAM** (In Progress)
   - Complete remaining notebooks (04-09)
   - Integrate SplaTAM and MonoGS examples
   - Add real dataset processing

2. **Phase 4: Feed-forward Gaussian** (In Progress)
   - Complete remaining notebooks (01-09)
   - Integrate MVSplat and pixelSplat examples
   - Add RE10K / ACID dataset processing

### Medium-term Goals

1. Complete Phase 2 development (notebooks 04-09)
2. Complete Phase 4 development (notebooks 01-09)
3. Start Phase 3 (DUSt3R) development
4. Integrate external repositories
5. Add Docker configuration
6. Set up CI/CD for testing

### Long-term Goals

1. Complete Phases 2-6
2. Create video tutorials
3. Build community contribution guidelines

---

## Notes

- Development follows the specs in `docs/Phase1_Development_Spec.md`, `docs/Phase2_Development_Spec.md`, `docs/Phase4_Development_Spec.md`
- Each notebook should be testable in Google Colab
- Code should be educational first, optimized second
- Visualizations are critical for understanding
