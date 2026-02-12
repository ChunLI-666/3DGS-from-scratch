# Phase 4: Feed-forward Gaussian（核心目标）

> **目标**: 掌握前馈式3DGS，实现零样本新视角合成  
> **时间**: 4周（建议先MVSplat后pixelSplat，对比学习）  
> **依赖**: Phase 1（3DGS概念）；建议有SLAM中的MVS/深度估计基础

---

## 📋 学习路径概览

```
Week 1: MVSplat理论基础
├── Day 1-2: 前馈vs优化的范式对比
├── Day 3-4: Cost Volume与Plane Sweeping
└── Day 5-7: MVSplat网络架构

Week 2: MVSplat代码实践
├── Day 8-10: 环境搭建与代码走读
├── Day 11-12: 在RE10K/ACID上运行
└── Day 13-14: 自定义数据测试

Week 3: pixelSplat对比学习
├── Day 15-17: pixelSplat架构（无Cost Volume）
├── Day 18-19: 代码实践
└── Day 20-21: MVSplat vs pixelSplat对比

Week 4: 进阶与2025新方法
├── Day 22-24: DepthSplat等2025方法
└── Day 25-28: 综合实验与总结
```

---

## 1. 理论基础深度讲解

### 1.1 核心问题：为什么需要Feed-forward Gaussian？

#### 原始3DGS的局限

```
原始3DGS (优化范式):
┌─────────────────────────────────────────────────────────────────┐
│  每场景优化: 30分钟 - 数小时                                       │
│  需要: 多视角图像 + 精确相机位姿 (COLMAP)                          │
│  优化: 高斯参数 (μ, S, R, σ, SH)                                  │
│                                                                 │
│  问题:                                                           │
│  1. 每场景重新训练，无法泛化                                       │
│  2. 依赖COLMAP，失败场景无法使用                                   │
│  3. 训练时间长，无法实时应用                                       │
└─────────────────────────────────────────────────────────────────┘
```

#### Feed-forward范式的优势

```
Feed-forward 3DGS:
┌─────────────────────────────────────────────────────────────────┐
│  单次前馈: 秒级                                                   │
│  需要: 2-5张图像 (无需位姿)                                        │
│  输出: 直接是优化好的高斯场景                                       │
│                                                                 │
│  优势:                                                           │
│  1. 跨场景泛化 (训练一次，多处使用)                                 │
│  2. 无需COLMAP，端到端                                             │
│  3. 实时推理，适合AR/VR                                            │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 范式对比：优化 vs 前馈

```
┌─────────────────────────────────────────────────────────────────┐
│              优化范式 vs 前馈范式                                 │
└─────────────────────────────────────────────────────────────────┘

优化范式 (原始3DGS):
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  输入图像     │─────►│  每场景优化   │─────►│  高斯场景    │
│  + COLMAP位姿 │      │  (30分钟+)   │      │  (该场景专用)│
└──────────────┘      └──────────────┘      └──────────────┘
                              │
                              ▼ (梯度下降)
                        ┌──────────────┐
                        │  优化高斯参数 │
                        │  μ, S, R, σ  │
                        └──────────────┘

前馈范式 (MVSplat/pixelSplat):
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│  输入图像     │─────►│  网络前馈    │─────►│  高斯场景    │
│  (无需位姿)   │      │  (秒级)      │      │  (直接可用)  │
└──────────────┘      └──────────────┘      └──────────────┘
                              │
                              ▼ (CNN/Transformer)
                        ┌──────────────┐
                        │  预测高斯参数 │
                        │  μ, S, R, σ  │
                        └──────────────┘

关键区别:
┌────────────────┬──────────────────┬──────────────────┐
│     维度       │   优化范式        │    前馈范式       │
├────────────────┼──────────────────┼──────────────────┤
│ 训练阶段       │ 每场景重新优化    │ 预训练网络权重    │
│ 推理阶段       │ 无 (优化即推理)   │ 单次前馈         │
│ 泛化能力       │ 无               │ 跨场景泛化        │
│ 相机位姿       │ 必需 (COLMAP)    │ 不需要           │
│ 时间成本       │ 30分钟+          │ 秒级             │
│ 质量           │ 高 (充分优化)     │ 中等 (快速)       │
└────────────────┴──────────────────┴──────────────────┘
```

### 1.3 MVSplat架构详解

#### 1.3.1 核心思想：Cost Volume + 3DGS

```
MVSplat核心创新:
将MVS (Multi-View Stereo) 的Cost Volume与3DGS结合

┌─────────────────────────────────────────────────────────────────┐
│  MVS传统方法 (如PatchMatch, MVSNet):                            │
│  - Cost Volume存储多视图匹配代价                                  │
│  - 从Cost Volume估计深度图                                        │
│  - 深度图融合为点云                                               │
│                                                                 │
│  MVSplat改进:                                                    │
│  - Cost Volume直接预测高斯参数                                    │
│  - 跳过显式深度估计                                               │
│  - 端到端可微分训练                                               │
└─────────────────────────────────────────────────────────────────┘
```

#### 1.3.2 Plane Sweeping构建Cost Volume

```
Plane Sweeping原理:

对于参考图像上的每个像素 (u,v)，在多个深度平面 d ∈ [d_min, d_max] 上采样:

┌─────────────────────────────────────────────────────────────────┐
│  深度平面 d1    深度平面 d2    深度平面 d3                        │
│       │              │              │                           │
│       ▼              ▼              ▼                           │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐                      │
│  │ ═══════ │    │  ═════  │    │   ══    │  ← 深度假设平面       │
│  └─────────┘    └─────────┘    └─────────┘                      │
│       │              │              │                           │
│       └──────────────┼──────────────┘                           │
│                      ▼                                          │
│              投影到源图像                                        │
│                      │                                          │
│              计算特征匹配代价                                     │
│                      │                                          │
│                      ▼                                          │
│              Cost Volume: C(u,v,d)                               │
│              - 高代价: 不匹配                                    │
│              - 低代价: 匹配 (正确深度)                            │
└─────────────────────────────────────────────────────────────────┘

可视化:

参考图像              源图像1              源图像2
┌─────┐              ┌─────┐              ┌─────┐
│  ●  │ ───────────► │  ●  │ ───────────► │  ●  │
│     │   投影到     │     │   投影到     │     │
└─────┘   深度d     └─────┘   深度d     └─────┘
   │                                        │
   └────────────────────────────────────────┘
                    │
                    ▼
              特征相似度
                    │
                    ▼
              Cost(u,v,d) = ||F_ref(u,v) - F_src(u',v')||
```

#### 1.3.3 完整网络架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    MVSplat Architecture                          │
└─────────────────────────────────────────────────────────────────┘

输入: N张图像 {I_1, I_2, ..., I_N} (通常N=2或5)

Step 1: 特征提取
┌─────────────────────────────────────────────────────────────────┐
│  每张图像 I_i ──► U-Net Encoder ──► Feature Map F_i             │
│                                                                 │
│  F_i ∈ R^(H/4 × W/4 × C)  (下采样4倍)                            │
└─────────────────────────────────────────────────────────────────┘

Step 2: Cost Volume构建
┌─────────────────────────────────────────────────────────────────┐
│  对于参考图像 I_ref 和每张源图像 I_src:                          │
│                                                                 │
│  1. Plane Sweeping:                                              │
│     - 在 [d_min, d_max] 范围内采样 D 个深度平面                   │
│     - 每个像素 (u,v) 投影到深度 d_k 得到 (u',v')                 │
│                                                                 │
│  2. 特征Warping:                                                 │
│     - 从 F_src 中采样特征: F_src(u', v')                         │
│     - 与 F_ref(u,v) 计算方差 (variance)                          │
│                                                                 │
│  3. Cost Volume:                                                 │
│     Cost ∈ R^(H/4 × W/4 × D × C)                                 │
│     - 方差越低，说明深度假设越正确                                │
└─────────────────────────────────────────────────────────────────┘

Step 3: Cost Volume处理
┌─────────────────────────────────────────────────────────────────┐
│  Cost Volume ──► 3D U-Net ──► Processed Cost                    │
│                                                                 │
│  3D U-Net作用:                                                   │
│  - 聚合空间信息 (2D)                                              │
│  - 聚合深度信息 (1D)                                              │
│  - 输出: 每个像素的深度分布 + 几何特征                            │
└─────────────────────────────────────────────────────────────────┘

Step 4: 高斯参数预测
┌─────────────────────────────────────────────────────────────────┐
│  对于每个像素 (u,v):                                              │
│                                                                 │
│  输入: Processed Cost(u,v) + 图像特征                            │
│  输出: 一个3D高斯                                                 │
│                                                                 │
│  预测参数:                                                        │
│  - 深度 d (从深度分布采样)                                        │
│  - 3D位置: μ = unproject(u, v, d, K)                             │
│  - 协方差: Σ = f(Cost, 特征)  (学习得到)                         │
│  - 不透明度: σ = g(Cost, 特征)                                   │
│  - 颜色: c = 图像颜色 或 预测SH                                  │
└─────────────────────────────────────────────────────────────────┘

Step 5: 可微分渲染
┌─────────────────────────────────────────────────────────────────┐
│  预测的高斯场景 ──► 3DGS Rasterizer ──► 渲染图像 Î               │
│                                                                 │
│  损失: L = |I_gt - Î|_1  (与真实图像比较)                        │
└─────────────────────────────────────────────────────────────────┘
```

### 1.4 pixelSplat架构对比

#### 1.4.1 核心区别：无显式Cost Volume

```
┌─────────────────────────────────────────────────────────────────┐
│              MVSplat vs pixelSplat                               │
└─────────────────────────────────────────────────────────────────┘

MVSplat (使用Cost Volume):
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  图像输入 │──►│  Cost    │──►│  3D U-Net │──►│ 高斯预测 │
│          │   │  Volume  │   │          │   │          │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
                    │
                    ▼
              显式几何推理
              (Plane Sweeping)

pixelSplat (无Cost Volume):
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  图像输入 │──►│ Encoder  │──►│ Decoder  │──►│ 高斯预测 │
│          │   │ (CNN)    │   │ (CNN)    │   │          │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
                                    │
                                    ▼
                              隐式几何学习
                              (网络自己学习深度)

对比:
┌────────────────┬──────────────────┬──────────────────┐
│     维度       │     MVSplat      │    pixelSplat    │
├────────────────┼──────────────────┼──────────────────┤
│ 显式几何       │ 有 (Cost Volume)  │ 无               │
│ 深度估计       │ 显式             │ 隐式             │
│ 参数数量       │ 较少             │ 较多             │
│ 泛化能力       │ 较强             │ 较弱             │
│ 速度           │ 快 (22 FPS)      │ 较慢 (10 FPS)    │
│ 无纹理区域     │ 较鲁棒           │ 容易失败         │
└────────────────┴──────────────────┴──────────────────┘
```

### 1.5 与SLAM中MVS的对比

```
┌─────────────────────────────────────────────────────────────────┐
│         SLAM中的MVS vs Feed-forward Gaussian                     │
└─────────────────────────────────────────────────────────────────┘

SLAM中的稠密重建 (如REMMAP, MVDepth):
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  关键帧      │──►│ 局部MVS      │──►│ 深度图融合   │
│  + 位姿      │   │ (PatchMatch) │   │ 到地图       │
└──────────────┘   └──────────────┘   └──────────────┘
   在线运行           实时性要求高         增量式更新

Feed-forward Gaussian:
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  任意图像    │──►│ 网络前馈     │──►│ 完整高斯场景 │
│  (无需位姿)  │   │ (秒级)       │   │ (一次性)     │
└──────────────┘   └──────────────┘   └──────────────┘
   离线/在线         预训练模型          可直接渲染

应用场景对比:
- SLAM MVS: 在线增量重建，需要实时性
- Feed-forward: 快速新视角合成，可接受秒级延迟
```

---

## 2. 算法详解

### 2.1 MVSplat训练策略

```
训练数据:
- RealEstate10K: 室内场景视频
- ACID: 室外场景视频
- 从视频中采样2-5帧作为输入

损失函数:
L = L_rgb + λ_depth * L_depth

其中:
- L_rgb: 渲染图像与GT的L1损失
- L_depth: 预测深度与GT深度的L1损失 (如果有)

数据增强:
- 随机裁剪
- 颜色抖动
- 随机选择参考帧
```

### 2.2 Pixel-aligned Gaussian（核心概念）

```
Pixel-aligned Gaussian:
每个像素对应一个高斯中心

┌─────────────────────────────────────────────────────────────────┐
│  图像 (H×W)                                                     │
│  ┌─────────────────────┐                                        │
│  │ ● ● ● ● ● ● ● ● ● ● │  每个像素 ● 对应一个高斯               │
│  │ ● ● ● ● ● ● ● ● ● ● │                                        │
│  │ ● ● ● ● ● ● ● ● ● ● │  高斯中心: 从像素反投影到3D            │
│  │ ● ● ● ● ● ● ● ● ● ● │                                        │
│  └─────────────────────┘  高斯参数: 网络预测                      │
│                                                                 │
│  与原始3DGS的区别:                                               │
│  - 原始3DGS: 高斯随机初始化，自适应分布                           │
│  - Feed-forward: 高斯与像素对齐，结构化分布                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. 代码实践指南

### 3.1 MVSplat环境搭建

```bash
# 克隆代码
git clone https://github.com/donydchen/mvsplat.git
cd mvsplat

# 创建环境
conda create -n mvsplat python=3.10
conda activate mvsplat

# 安装PyTorch
pip install torch==2.1.0 torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装PyTorch Lightning
pip install pytorch-lightning==2.2.5

# 安装其他依赖
pip install -r requirements.txt

# 安装gsplat (用于渲染)
pip install gsplat

# 下载预训练模型
mkdir -p checkpoints
# 从项目页面下载 mvsplat_re10k.ckpt
```

### 3.2 MVSplat关键代码

#### 3.2.1 Cost Volume构建

```python
class CostVolume(nn.Module):
    """
    MVSplat Cost Volume构建
    """
    def __init__(self, num_depths=64, min_depth=1.0, max_depth=100.0):
        super().__init__()
        self.num_depths = num_depths
        self.min_depth = min_depth
        self.max_depth = max_depth
        
        # 对数采样深度平面
        self.depth_planes = torch.exp(torch.linspace(
            math.log(min_depth), math.log(max_depth), num_depths
        ))
    
    def forward(self, feat_ref, feat_src, K_ref, K_src, T_src_ref):
        """
        Args:
            feat_ref: [B, C, H, W] 参考图像特征
            feat_src: [B, C, H, W] 源图像特征
            K_ref, K_src: 相机内参
            T_src_ref: 从参考到源的位姿变换
        Returns:
            cost_volume: [B, C, D, H, W]
        """
        B, C, H, W = feat_ref.shape
        D = self.num_depths
        
        # 构建Cost Volume
        cost_volume = torch.zeros(B, C, D, H, W, device=feat_ref.device)
        
        for i, depth in enumerate(self.depth_planes):
            # 1. 将参考图像像素投影到3D
            # pixel_coords: [B, 3, H, W]
            pixel_coords = create_meshgrid(H, W, device=feat_ref.device)
            pixel_coords = pixel_coords.unsqueeze(0).repeat(B, 1, 1, 1)
            
            # 2. 反投影到3D
            # X = Z * K^(-1) * [u, v, 1]
            points_3d = depth * torch.matmul(
                torch.inverse(K_ref), 
                pixel_coords.reshape(B, 3, -1)
            ).reshape(B, 3, H, W)
            
            # 3. 投影到源图像
            points_3d_homo = torch.cat([
                points_3d, 
                torch.ones(B, 1, H, W, device=points_3d.device)
            ], dim=1).reshape(B, 4, -1)
            
            # 应用位姿变换
            points_3d_src = torch.matmul(T_src_ref, points_3d_homo)
            points_3d_src = points_3d_src[:, :3] / (points_3d_src[:, 3:4] + 1e-8)
            
            # 投影到图像平面
            pixel_coords_src = torch.matmul(K_src, points_3d_src)
            pixel_coords_src = pixel_coords_src[:, :2] / (pixel_coords_src[:, 2:3] + 1e-8)
            pixel_coords_src = pixel_coords_src.reshape(B, 2, H, W)
            
            # 4. 从源特征图采样
            # grid_sample: 双线性插值
            feat_src_warped = F.grid_sample(
                feat_src, 
                pixel_coords_src.permute(0, 2, 3, 1),
                align_corners=False
            )
            
            # 5. 计算方差 (cost)
            # 方差越低，匹配越好
            cost = (feat_ref - feat_src_warped).pow(2)
            cost_volume[:, :, i] = cost
        
        return cost_volume
```

#### 3.2.2 高斯预测

```python
class GaussianPredictor(nn.Module):
    """
    从Cost Volume预测高斯参数
    """
    def __init__(self, channels=64):
        super().__init__()
        
        # 处理Cost Volume的3D U-Net
        self.cost_volume_encoder = CostVolumeEncoder(channels)
        
        # 预测头
        self.depth_head = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Softplus()  # 深度为正
        )
        
        self.covariance_head = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 4, 1)  # 2x2协方差矩阵 (对称)
        )
        
        self.opacity_head = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, cost_volume, feat_image, K):
        """
        Args:
            cost_volume: [B, C, D, H, W]
            feat_image: [B, C, H, W]
            K: 相机内参
        Returns:
            gaussians: 高斯参数字典
        """
        B, C, D, H, W = cost_volume.shape
        
        # 处理Cost Volume
        cost_feat = self.cost_volume_encoder(cost_volume)  # [B, C, H, W]
        
        # 融合图像特征
        fused_feat = cost_feat + feat_image
        
        # 预测深度
        depth = self.depth_head(fused_feat)  # [B, 1, H, W]
        
        # 预测协方差 (在图像平面)
        cov_2d = self.covariance_head(fused_feat)  # [B, 4, H, W]
        # 构建2x2协方差矩阵
        cov_matrix = torch.zeros(B, 2, 2, H, W, device=cov_2d.device)
        cov_matrix[:, 0, 0] = cov_2d[:, 0]  # sigma_xx
        cov_matrix[:, 0, 1] = cov_2d[:, 1]  # sigma_xy
        cov_matrix[:, 1, 0] = cov_2d[:, 1]  # sigma_yx
        cov_matrix[:, 1, 1] = cov_2d[:, 2]  # sigma_yy
        
        # 预测不透明度
        opacity = self.opacity_head(fused_feat)  # [B, 1, H, W]
        
        # 构建3D高斯
        # 从深度和像素坐标反投影3D位置
        pixel_coords = create_meshgrid(H, W, device=depth.device)
        points_3d = depth * torch.matmul(
            torch.inverse(K), 
            pixel_coords.reshape(B, 3, -1)
        ).reshape(B, 3, H, W)
        
        # 将2D协方差提升到3D (简化假设)
        # 实际实现更复杂，考虑深度不确定性
        
        gaussians = {
            'positions': points_3d.permute(0, 2, 3, 1).reshape(B, -1, 3),  # [B, N, 3]
            'covariances': cov_matrix.permute(0, 3, 4, 1, 2).reshape(B, -1, 2, 2),
            'opacities': opacity.reshape(B, -1, 1),
            'colors': feat_image.permute(0, 2, 3, 1).reshape(B, -1, C)
        }
        
        return gaussians
```

### 3.3 运行实验

```bash
# 在RE10K测试集上评估
python test.py --config configs/re10k.yaml --checkpoint checkpoints/mvsplat_re10k.ckpt

# 在自定义数据上运行
python inference.py \
    --image1 path/to/image1.jpg \
    --image2 path/to/image2.jpg \
    --checkpoint checkpoints/mvsplat_re10k.ckpt \
    --output output/
```

---

## 4. 实验与检查点

### 4.1 实验1: 稀疏视角重建

**目标**: 测试不同输入视角数量的影响

```python
def test_sparse_view_reconstruction(model, scene_path, view_counts=[2, 3, 5, 10]):
    """
    测试不同视角数量的重建效果
    """
    results = {}
    
    for n_views in view_counts:
        # 采样n_views张图像
        images = load_scene_images(scene_path, n_views)
        
        # 重建
        gaussians = model.reconstruct(images)
        
        # 渲染测试视角
        test_views = load_test_views(scene_path)
        psnr_list = []
        
        for view in test_views:
            rendered = render(gaussians, view.pose, view.K)
            psnr = compute_psnr(rendered, view.image)
            psnr_list.append(psnr)
        
        results[n_views] = {
            'psnr': np.mean(psnr_list),
            'num_gaussians': gaussians['positions'].shape[0]
        }
    
    return results

# 预期结果:
# n_views=2:  PSNR ~20dB (基础重建)
# n_views=5:  PSNR ~25dB (较好)
# n_views=10: PSNR ~28dB (接近优化方法)
```

### 4.2 实验2: MVSplat vs pixelSplat对比

```python
def compare_methods(scene_path):
    """
    对比MVSplat和pixelSplat
    """
    images = load_scene_images(scene_path, n_views=2)
    
    # MVSplat
    start = time.time()
    gaussians_mvsplat = mvsplat.reconstruct(images)
    time_mvsplat = time.time() - start
    
    # pixelSplat
    start = time.time()
    gaussians_pixelsplat = pixelsplat.reconstruct(images)
    time_pixelsplat = time.time() - start
    
    # 评估
    test_view = load_test_view(scene_path)
    
    rendered_mvsplat = render(gaussians_mvsplat, test_view.pose, test_view.K)
    rendered_pixelsplat = render(gaussians_pixelsplat, test_view.pose, test_view.K)
    
    psnr_mvsplat = compute_psnr(rendered_mvsplat, test_view.image)
    psnr_pixelsplat = compute_psnr(rendered_pixelsplat, test_view.image)
    
    print(f"MVSplat:   PSNR={psnr_mvsplat:.2f}dB, Time={time_mvsplat:.2f}s")
    print(f"pixelSplat: PSNR={psnr_pixelsplat:.2f}dB, Time={time_pixelsplat:.2f}s")
```

### 4.3 ✅ 完成检查点

- [ ] **检查点1**: 能画出MVSplat网络结构图
  ```
  Input Images (2-5 views)
    │
    ├──► U-Net Encoder ──► Feature Maps
    │
    ├──► Cost Volume (Plane Sweeping)
    │
    ├──► 3D U-Net Processing
    │
    ├──► Gaussian Prediction (per pixel)
    │    - Depth
    │    - Covariance
    │    - Opacity
    │    - Color
    │
    └──► 3DGS Rendering
  ```

- [ ] **检查点2**: 理解Pixel-aligned Gaussian
  ```
  Pixel-aligned Gaussian:
  - 每个像素对应一个高斯
  - 高斯中心 = 像素反投影到3D
  - 高斯参数 = 网络预测
  
  与原始3DGS对比:
  - 原始: 随机初始化，自适应分布
  - Feed-forward: 结构化，与像素对齐
  ```

- [ ] **检查点3**: 成功运行MVSplat并测试稀疏视角
  ```bash
  python inference.py --image1 img1.jpg --image2 img2.jpg
  
  # 预期输出:
  # - 重建的高斯场景
  # - 新视角渲染结果
  # - 推理时间 < 5s
  ```

---

## 5. 常见问题与解答

### Q1: 为什么MVSplat比pixelSplat快？

**A**: 
1. MVSplat参数更少（Cost Volume结构更高效）
2. MVSplat有显式几何先验，学习更容易
3. pixelSplat需要更多参数隐式学习几何

### Q2: Feed-forward方法在无纹理区域表现如何？

**A**: 
- 不如优化方法（原始3DGS）
- MVSplat比pixelSplat更鲁棒（Cost Volume提供先验）
- 这是当前研究的热点问题

### Q3: 如何扩展到更多视角（>5张）？

**A**: 
- MVSplat原生支持任意数量
- 但计算量随视角线性增长
- 见VGGT/Fast3R的改进方案

---

## 6. 延伸阅读

### 必读论文

1. **MVSplat**: "MVSplat: Efficient 3D Gaussian Splatting from Sparse Multi-View Images" ECCV 2024
   - [arXiv:2403.14627](https://arxiv.org/abs/2403.14627)

2. **pixelSplat**: "pixelSplat: 3D Gaussian Splats from Image Pairs" CVPR 2024
   - [arXiv:2312.12337](https://arxiv.org/abs/2312.12337)

3. **DepthSplat** (2025): 结合单目深度先验
   - [arXiv:2412.18010](https://arxiv.org/abs/2412.18010)

### 下一步

- [Phase 5: VGGT](./Phase5_VGGT.md) - 学习多视图基础模型
