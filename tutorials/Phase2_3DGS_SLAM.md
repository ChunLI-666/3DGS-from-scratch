# Phase 2: 3DGS+SLAM（关键过渡）

> **目标**: 理解3DGS如何嵌入SLAM框架，建立新旧范式桥梁  
> **时间**: 2-3周  
> **依赖**: Phase 1（理解3DGS优化过程）

---

## 📋 学习路径概览

```
Week 1: 理论基础
├── Day 1-2: 3DGS+SLAM系统架构对比
├── Day 3-4: SplaTAM Tracking机制
└── Day 5-7: SplaTAM Mapping与Gaussian Update

Week 2: 代码实践
├── Day 8-10: SplaTAM代码走读
├── Day 11-12: 在TUM/Replica数据集上运行
└── Day 13-14: 与ORB-SLAM3对比实验

Week 3: 进阶与扩展
├── Day 15-17: GS-SLAM与MonoGS对比
└── Day 18-21: 探索改进方向
```

---

## 1. 理论基础深度讲解

### 1.1 核心问题：为什么3DGS+SLAM？

#### 传统SLAM的局限

```
传统稠密SLAM (如ElasticFusion, NICE-SLAM):
┌─────────────────────────────────────────────────────────────┐
│  场景表示: 体素网格 / 隐式MLP / 点云                          │
│  渲染方式: 直接投影 / 体渲染 (慢)                              │
│  地图质量: 几何一致但外观模糊                                  │
│  实时性:   体素方法实时，隐式方法慢                           │
└─────────────────────────────────────────────────────────────┘
```

#### 3DGS+SLAM的优势

```
3DGS-based SLAM (SplaTAM, GS-SLAM):
┌─────────────────────────────────────────────────────────────┐
│  场景表示: 显式高斯椭球 (可微分)                              │
│  渲染方式: 光栅化 (实时，100+ FPS)                            │
│  地图质量: 几何+外观同时优化，照片级真实感                      │
│  实时性:   跟踪10+ FPS，建图实时                               │
│  附加优势: 可编辑、可组合、支持透明物体                        │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 系统架构对比

#### 传统RGB-D SLAM (以ORB-SLAM2为例)

```
┌─────────────────────────────────────────────────────────────────┐
│                    ORB-SLAM2 Architecture                        │
└─────────────────────────────────────────────────────────────────┘

RGB-D Input
    │
    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Tracking   │────►│  Local BA    │────►│ Loop Closing │
│  (ORB特征)    │     │ (局部优化)    │     │ (回环检测)    │
└──────────────┘     └──────────────┘     └──────────────┘
    │                       │                    │
    ▼                       ▼                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  相机位姿     │     │  局部地图点   │     │  全局一致性   │
│  (SE3)       │     │  (3D点云)    │     │  (Pose Graph)│
└──────────────┘     └──────────────┘     └──────────────┘

地图表示: 稀疏/稠密点云 (ORB-SLAM2仅稀疏)
渲染: 直接投影 (无真实感)
```

#### SplaTAM架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    SplaTAM Architecture                          │
└─────────────────────────────────────────────────────────────────┘

RGB-D Input
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Tracking                                 │
│  ┌──────────────┐    ┌──────────────────────────────────────┐  │
│  │  渲染当前帧   │───►│  与输入RGB-D比较，优化相机位姿        │  │
│  │  (Splatting) │    │  (可微分渲染梯度)                     │  │
│  └──────────────┘    └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼ (关键帧)
┌─────────────────────────────────────────────────────────────────┐
│                         Mapping                                  │
│  ┌──────────────┐    ┌──────────────────────────────────────┐  │
│  │  添加新高斯   │    │  联合优化高斯参数 + 相机位姿          │  │
│  │  (初始化)    │    │  (RGB-D渲染损失)                      │  │
│  └──────────────┘    └──────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Gaussian Update                               │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  几何验证    │───►│  更新现有    │───►│  删除不稳定  │      │
│  │  (深度一致)  │    │  高斯参数    │    │  高斯        │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
└─────────────────────────────────────────────────────────────────┘

关键差异:
1. 跟踪使用可微分渲染 (而非特征匹配或直接法)
2. 地图是高斯场景 (而非点云或体素)
3. 渲染是光栅化 (而非体渲染)
```

### 1.3 SplaTAM Tracking机制详解

#### 1.3.1 核心思想

```
传统跟踪: 最小化几何误差 (重投影误差 / 光度误差)
SplaTAM跟踪: 最小化渲染误差 (可微分渲染)

┌──────────────────────────────────────────────────────────────┐
│  SplaTAM Tracking Loss                                       │
│                                                              │
│  L_tracking = λ_rgb * |I - Î|_1 + λ_depth * |D - D̂|_1       │
│                                                              │
│  其中:                                                        │
│    I, D = 输入RGB-D图像                                       │
│    Î, D̂ = 从高斯地图渲染的图像                                 │
│    渲染过程对相机位姿可微分                                    │
└──────────────────────────────────────────────────────────────┘
```

#### 1.3.2 位姿优化过程

```python
# 伪代码：SplaTAM Tracking
def track_camera_pose(gaussian_map, current_rgbd, initial_pose):
    """
    通过可微分渲染优化相机位姿
    """
    pose = initial_pose.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([pose], lr=0.01)
    
    for iter in range(num_iterations):
        # 1. 使用当前位姿渲染
        rendered_rgb, rendered_depth = render(gaussian_map, pose, K)
        
        # 2. 计算渲染损失
        loss_rgb = F.l1_loss(rendered_rgb, current_rgbd.rgb)
        loss_depth = F.l1_loss(rendered_depth, current_rgbd.depth)
        loss = loss_rgb + loss_depth
        
        # 3. 反向传播到位姿
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return pose
```

**SLAM对比**:
- ORB-SLAM: 特征匹配 + PnP，离散优化
- DSO: 直接法，光度误差，但地图是点云
- SplaTAM: 直接法 + 可微分渲染，地图是高斯

### 1.4 Mapping与Gaussian Update

#### 1.4.1 新观测融合

```
当相机移动到新区域时:

1. 深度图引导的高斯初始化:
   ┌─────────────────────────────────────────────────────────┐
   │  对于深度图中的每个像素 (u,v):                           │
   │    - 反投影到3D: P = K^(-1) * [u, v, 1]^T * D(u,v)      │
   │    - 初始化新高斯中心: μ = P                            │
   │    - 初始化缩放: s = [k*D(u,v), k*D(u,v), ε]           │
   │    - 初始化颜色: c = I(u,v)                             │
   └─────────────────────────────────────────────────────────┘

2. 与现有高斯融合:
   - 如果新高斯位置接近现有高斯，更新现有高斯
   - 否则，添加新高斯
```

#### 1.4.2 联合优化

```
关键帧的RGB-D与地图联合优化:

L_mapping = Σ_keyframes [ |I_k - Î_k|_1 + |D_k - D̂_k|_1 ]

优化变量:
  - 高斯参数: {μ, S, R, σ, SH}
  - 关键帧位姿: {T_k}

注意: 与原始3DGS不同，位姿也是优化变量！
```

### 1.5 不同3DGS+SLAM方法对比

| 方法 | 输入 | 跟踪方式 | 建图方式 | 特点 |
|------|------|---------|---------|------|
| **SplaTAM** | RGB-D | 可微分渲染 | 在线高斯更新 | 最经典，易理解 |
| **GS-SLAM** | RGB-D | 可微分渲染 + 粗到细 | 自适应扩展策略 | 更快，更鲁棒 |
| **MonoGS** | RGB单目 | 可微分渲染 | 单目深度估计 | 无需深度传感器 |
| **Photo-SLAM** | RGB | 特征跟踪 + 神经渲染 | 关键帧高斯 | 混合方法 |

---

## 2. 算法详解

### 2.1 SplaTAM完整流程

```
┌─────────────────────────────────────────────────────────────────┐
│                    SplaTAM Pipeline                              │
└─────────────────────────────────────────────────────────────────┘

初始化:
  ├── 第一帧RGB-D初始化高斯地图
  │   └── 每个深度像素 → 一个高斯
  └── 设置第一帧为关键帧

对于每一帧 t:
  │
  ├── 1. Tracking (跟踪)
  │   ├── 使用上一帧位姿作为初始估计
  │   ├── 迭代优化位姿 (渲染损失)
  │   └── 输出: 当前位姿 T_t
  │
  ├── 2. Keyframe Selection (关键帧选择)
  │   ├── 计算与最近关键帧的:
  │   │   - 位姿距离 (平移 + 旋转)
  │   │   - 共视高斯比例
  │   └── 如果超过阈值，设为关键帧
  │
  └── 3. Mapping (仅关键帧)
      ├── 3.1 添加新高斯
      │   ├── 深度图反投影
      │   └── 与现有高斯融合
      │
      ├── 3.2 联合优化
      │   ├── 优化: 最近N个关键帧的位姿
      │   ├── 优化: 可见高斯参数
      │   └── 损失: RGB-D渲染损失
      │
      └── 3.3 Gaussian Update
          ├── 几何验证 (深度一致性)
          ├── 更新高斯参数
          └── 删除不稳定高斯
```

### 2.2 关键创新点

#### 2.2.1 为什么SplaTAM不需要Loop Closing？

```
传统SLAM需要回环检测的原因:
1. 位姿漂移累积 → 全局不一致
2. 地图点重复 → 冗余和不一致

SplaTAM的解决方案:
1. 稠密RGB-D跟踪 → 漂移较小
2. 高斯地图是"软"的 → 新观测自然融合
3. 联合优化 → 隐式回环

但: 大规模场景仍然需要显式回环！
```

#### 2.2.2 与ORB-SLAM3的精度对比

```
TUM RGB-D fr1/desk 序列:

方法          ATE RMSE (cm)
────────────────────────────
ORB-SLAM2     1.60
ORB-SLAM3     1.40
SplaTAM       3.35
GS-SLAM       2.80

分析:
- SplaTAM精度略低于ORB-SLAM系列
- 原因: 稠密跟踪比稀疏特征跟踪噪声更大
- 优势: 稠密地图 + 真实感渲染
```

---

## 3. 代码实践指南

### 3.1 SplaTAM环境搭建

```bash
# 克隆代码
git clone https://github.com/spla-tam/SplaTAM.git
cd SplaTAM

# 创建环境
conda create -n splatam python=3.9
conda activate splatam

# 安装依赖
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118
pip install -r requirements.txt

# 安装3DGS子模块
cd submodules/diff-gaussian-rasterization
pip install -e .
cd ../simple-knn
pip install -e .
cd ../..
```

### 3.2 关键代码走读

#### 3.2.1 跟踪模块 (`splatam.py`)

```python
def tracking_step(self, current_frame):
    """
    单步跟踪
    """
    # 初始化位姿 (使用上一帧或运动模型)
    if self.prev_pose is not None:
        initial_pose = self.prev_pose @ self.motion_model
    else:
        initial_pose = torch.eye(4, device='cuda')
    
    # 可优化位姿
    cam_rot = SO3.from_matrix(initial_pose[:3, :3]).log()
    cam_trans = initial_pose[:3, 3]
    
    cam_rot.requires_grad = True
    cam_trans.requires_grad = True
    
    optimizer = torch.optim.Adam([cam_rot, cam_trans], lr=0.01)
    
    # 迭代优化
    for iter in range(self.tracking_iters):
        # 构建位姿矩阵
        pose = torch.eye(4, device='cuda')
        pose[:3, :3] = SO3.exp(cam_rot).matrix()
        pose[:3, 3] = cam_trans
        
        # 渲染
        rendered_rgb, rendered_depth = self.render_from_pose(pose)
        
        # 计算损失
        loss_rgb = F.l1_loss(rendered_rgb, current_frame.rgb)
        loss_depth = F.l1_loss(rendered_depth, current_frame.depth)
        loss = loss_rgb + loss_depth
        
        # 优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # 保存结果
    self.prev_pose = pose.detach()
    return pose.detach()
```

#### 3.2.2 建图模块

```python
def mapping_step(self, keyframe):
    """
    关键帧建图
    """
    # 1. 添加新高斯
    new_gaussians = self.initialize_gaussians_from_depth(keyframe)
    self.gaussian_map.add_gaussians(new_gaussians)
    
    # 2. 联合优化
    # 收集最近的关键帧
    local_keyframes = self.get_local_keyframes(keyframe, k=5)
    
    # 优化变量
    params = []
    for gf in self.gaussian_map.gaussians:
        params.extend([gf.xyz, gf.scaling, gf.rotation, gf.opacity, gf.sh])
    for kf in local_keyframes:
        params.extend([kf.pose])
    
    optimizer = torch.optim.Adam(params, lr=0.001)
    
    for iter in range(self.mapping_iters):
        total_loss = 0
        
        # 对每个局部关键帧计算损失
        for kf in local_keyframes:
            rendered_rgb, rendered_depth = self.render_from_pose(kf.pose)
            loss = F.l1_loss(rendered_rgb, kf.rgb) + F.l1_loss(rendered_depth, kf.depth)
            total_loss += loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
    
    # 3. Gaussian Update
    self.update_gaussians()
```

### 3.3 运行实验

```bash
# TUM RGB-D数据集
python splatam.py --config configs/tum/rgbd_dataset_freiburg1_desk.yaml

# Replica数据集
python splatam.py --config configs/replica/room0.yaml

# 评估轨迹
python evaluation/evaluate_ate.py --groundtruth groundtruth.txt --estimated estimated.txt
```

---

## 4. 实验与检查点

### 4.1 实验1: 跟踪机制对比

**目标**: 对比SplaTAM跟踪与ORB-SLAM2跟踪

```python
def compare_tracking():
    """
    在相同序列上运行两种方法，对比:
    1. 跟踪精度 (ATE)
    2. 运行速度 (FPS)
    3. 鲁棒性 (丢失次数)
    """
    # 运行ORB-SLAM2
    orb_slam_trajectory = run_orb_slam2(sequence)
    
    # 运行SplaTAM
    splatam_trajectory = run_splatam(sequence)
    
    # 计算ATE
    orb_ate = compute_ate(orb_slam_trajectory, ground_truth)
    splatam_ate = compute_ate(splatam_trajectory, ground_truth)
    
    print(f"ORB-SLAM2 ATE: {orb_ate:.3f}m")
    print(f"SplaTAM ATE: {splatam_ate:.3f}m")
```

### 4.2 实验2: 地图质量评估

**目标**: 评估高斯地图的渲染质量

```python
def evaluate_map_quality(gaussian_map, test_views):
    """
    评估地图质量
    """
    psnr_list = []
    ssim_list = []
    
    for view in test_views:
        # 渲染
        rendered = render(gaussian_map, view.pose, view.K)
        
        # 计算指标
        psnr = compute_psnr(rendered, view.rgb)
        ssim = compute_ssim(rendered, view.rgb)
        
        psnr_list.append(psnr)
        ssim_list.append(ssim)
    
    print(f"PSNR: {np.mean(psnr_list):.2f} dB")
    print(f"SSIM: {np.mean(ssim_list):.4f}")
```

### 4.3 ✅ 完成检查点

- [ ] **检查点1**: 能画出SplaTAM系统框图
  ```
  Input: RGB-D Stream
    │
    ├──► Tracking ──► Camera Pose
    │      (渲染损失优化)
    │
    └──► Mapping (Keyframe)
           ├──► Gaussian Initialization
           ├──► Joint Optimization
           └──► Gaussian Update
  ```

- [ ] **检查点2**: 理解为什么SplaTAM不需要显式Loop Closing
  ```
  原因:
  1. 稠密RGB-D跟踪漂移较小
  2. 高斯地图支持"软"融合
  3. 联合优化隐式修正漂移
  
  局限: 大规模场景仍需显式回环
  ```

- [ ] **检查点3**: 对比SplaTAM与ORB-SLAM2精度
  ```bash
  # 在TUM fr1/desk上运行
  # SplaTAM ATE: ~3.5cm
  # ORB-SLAM2 ATE: ~1.6cm
  
  # 结论: SplaTAM精度略低，但提供稠密地图
  ```

---

## 5. 常见问题与解答

### Q1: SplaTAM在纯RGB上能工作吗？

**A**: 原版SplaTAM需要RGB-D。单目版本需要深度估计（见MonoGS）。

### Q2: 如何处理动态物体？

**A**: 当前3DGS+SLAM方法大多假设静态场景。动态物体处理仍是开放问题。

### Q3: 与NeRF-SLAM相比如何？

**A**: 
- 速度: 3DGS >> NeRF (光栅化 vs 体渲染)
- 质量: 相当，3DGS略好
- 内存: 3DGS更高 (显式表示)

---

## 6. 延伸阅读

### 必读论文

1. **SplaTAM**: "Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM" CVPR 2024
   - [arXiv:2312.02126](https://arxiv.org/abs/2312.02126)

2. **GS-SLAM**: "GS-SLAM: Dense Visual SLAM with 3D Gaussian Splatting"
   - [arXiv:2311.11700](https://arxiv.org/abs/2311.11700)

3. **MonoGS**: "Gaussian Splatting SLAM"
   - [arXiv:2312.06704](https://arxiv.org/abs/2312.06704)

### 下一步

- [Phase 3: DUSt3R](./Phase3_DUSt3R.md) - 学习几何学习范式
- [Phase 4: Feed-forward Gaussian](./Phase4_FeedForward_Gaussian.md) - 学习前馈方法
