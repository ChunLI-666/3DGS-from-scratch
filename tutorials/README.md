# 3D Gaussian Splatting 学习路线图 - 详细学习计划

> 专为有SLAM背景的研究者设计的系统性学习指南
> 
> 版本: 1.0 | 更新日期: 2026-02-10

---

## 📋 目录

1. [Phase 1: 3DGS基础深化](./Phase1_3DGS_Foundation.md)
2. [Phase 2: 3DGS+SLAM](./Phase2_3DGS_SLAM.md)
3. [Phase 3: DUSt3R](./Phase3_DUSt3R.md)
4. [Phase 4: Feed-forward Gaussian](./Phase4_FeedForward_Gaussian.md)
5. [Phase 5: VGGT](./Phase5_VGGT.md)
6. [Phase 6: 前沿融合](./Phase6_Frontier.md)

---

## 🎯 路线图评估总览

### 整体合理性评估: ⭐⭐⭐⭐⭐ (5/5)

这份路线图设计得非常合理，具有以下优点：

| 评估维度 | 评分 | 说明 |
|---------|------|------|
| **逻辑递进性** | ⭐⭐⭐⭐⭐ | 从基础到应用，从优化到前馈，层次清晰 |
| **模块化设计** | ⭐⭐⭐⭐⭐ | 各Phase可独立学习，依赖关系明确 |
| **SLAM视角** | ⭐⭐⭐⭐⭐ | 充分考虑SLAM背景学习者的知识迁移 |
| **前沿覆盖** | ⭐⭐⭐⭐⭐ | 涵盖2023-2025年最新进展 |
| **实践导向** | ⭐⭐⭐⭐ | 包含代码实践，但可增加更多调试指南 |

### 依赖关系验证

```
Phase 1: 3DGS基础 ✓
    │
    ├──► Phase 2: 3DGS+SLAM ✓ (依赖Phase 1)
    │
    ├──► Phase 4: Feed-forward Gaussian ✓ (依赖Phase 1概念)
    │
Phase 3: DUSt3R ✓ (可并行)
    │
    ├──► Phase 5: VGGT ✓ (依赖Phase 3)
    │
    └──► Phase 4: 部分方法 (可选依赖)
```

### 关键改进建议

1. **新增内容建议**:
   - 在Phase 1中增加CUDA光栅化原理深入讲解
   - 在Phase 4中增加DepthSplat等2025最新方法
   - 在Phase 5中增加Fast3R作为VGGT的对比学习

2. **时间估算调整**:
   - Phase 1: 2-3周 → 建议3-4周（CUDA部分需要更多时间）
   - Phase 4: 3周 → 建议4周（两种路径都需要充分理解）

3. **新增实验建议**:
   - 增加跨Phase的联合实验（如VGGT+MVSplat）

---

## 📚 文档使用指南

每个Phase的详细文档包含以下结构：

1. **学习目标与核心问题** - 明确本阶段要解决的问题
2. **理论基础深度讲解** - 从SLAM视角对比理解
3. **算法详解与图解** - 可视化关键概念
4. **代码实践指南** - 关键代码片段与注释
5. **实验与检查点** - 可验证的学习成果
6. **常见问题与解答** - 调试技巧与陷阱规避
7. **延伸阅读** - 相关论文与资源

---

## 🛠️ 环境准备速查

```bash
# 建议的目录结构
~/3dgs_study/
├── phase1_gaussian_splatting/    # 原版3DGS
├── phase2_splatam/               # SplaTAM
├── phase3_dust3r/                # DUSt3R
├── phase4_mvsplat/               # MVSplat
├── phase4_pixelsplat/            # pixelSplat
├── phase5_vggt/                  # VGGT
└── datasets/                     # 共享数据集

# 显存需求
- Phase 1-2: 12GB+ (训练需要)
- Phase 3: 16GB+ (ViT-L模型)
- Phase 4: 12GB+ (推理) / 24GB+ (训练)
- Phase 5: 24GB+ (VGGT-1B模型)
```

---

## 📖 推荐学习顺序

### 保守路线 (6个月)
```
Month 1: Phase 1 → 理解3DGS核心概念
Month 2: Phase 2 → 理解SLAM集成
Month 3: Phase 3 → 理解几何学习范式
Month 4: Phase 4 → 掌握前馈方法
Month 5: Phase 4 → 对比学习pixelSplat
Month 6: Phase 5 → 掌握基础模型
```

### 快速路线 (3个月)
```
Week 1-2:  Phase 1 (快速过)
Week 3-6:  Phase 4 (重点) + 回溯Phase 1/2
Week 7-8:  Phase 3 (并行)
Week 9-10: Phase 5
Week 11-12: 综合实验
```

### SLAMer专项 (2个月)
```
Week 1:   Phase 1 快速回顾
Week 2-4: Phase 4 重点学习 (Cost Volume与MVS关联)
Week 5-6: Phase 3 理解Pose-free重建
Week 7-8: Phase 5 理解前端替代方案
```

---

## 🔗 关键论文索引

| Phase | 核心论文 | 年份 | 会议 |
|-------|---------|------|------|
| 1 | 3D Gaussian Splatting for Real-Time Radiance Field Rendering | 2023 | SIGGRAPH |
| 2 | SplaTAM: Splat, Track & Map 3D Gaussians | 2024 | CVPR |
| 2 | GS-SLAM: Dense Visual SLAM with 3D Gaussians | 2024 | - |
| 3 | DUSt3R: Geometric 3D Vision Made Easy | 2024 | CVPR |
| 4 | MVSplat: Efficient 3D Gaussian Splatting | 2024 | ECCV |
| 4 | pixelSplat: 3D Gaussian Splats from Image Pairs | 2024 | CVPR |
| 5 | VGGT: Visual Geometry Grounded Transformer | 2025 | CVPR (Best Paper) |
| 5 | Fast3R: 3D Reconstruction of 1000+ Images | 2025 | CVPR |

---

开始你的学习之旅吧！建议从 [Phase 1](./Phase1_3DGS_Foundation.md) 开始，或根据你的背景选择合适的切入点。
