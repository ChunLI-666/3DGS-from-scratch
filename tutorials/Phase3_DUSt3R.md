# Phase 3: DUSt3R（几何学习范式）

> **目标**: 理解如何用神经网络替代几何算法（特征匹配+三角化）  
> **时间**: 2-3周  
> **依赖**: 无（可与Phase 4并行）  
> **与SLAM联系**: 对比传统前端（特征提取+匹配+RANSAC）

---

## 📋 学习路径概览

```
Week 1: 理论基础
├── Day 1-2: DUSt3R核心思想与SLAM对比
├── Day 3-4: Pointmap表示与网络架构
└── Day 5-7: 损失函数与训练策略

Week 2: 代码实践
├── Day 8-10: 官方代码走读
├── Day 11-12: 运行推理与可视化
└── Day 13-14: 与COLMAP对比实验

Week 3: 进阶与扩展
├── Day 15-17: 理解下游任务（位姿估计、深度）
└── Day 18-21: 探索MASt3R等扩展
```

---

## 1. 理论基础深度讲解

### 1.1 核心问题：为什么需要DUSt3R？

#### 传统SfM/SLAM的局限

```
传统SfM Pipeline (COLMAP):
┌─────────────────────────────────────────────────────────────────┐
│  1. 特征提取 (SIFT/SuperPoint)                                   │
│  2. 特征匹配 + RANSAC几何验证                                    │
│  3. 增量式重建 (三角化 + BA)                                     │
│  4. 稠密重建 (PatchMatch MVS)                                    │
└─────────────────────────────────────────────────────────────────┘

问题:
- 多阶段pipeline，误差累积
- 需要相机内参
- 弱纹理/重复纹理区域失败
- 计算复杂度高
```

#### DUSt3R的解决方案

```
DUSt3R Pipeline:
┌─────────────────────────────────────────────────────────────────┐
│  输入: 两张图像 I1, I2 (无需相机参数)                             │
│                    ↓                                             │
│  网络: ViT Encoder + Cross-Attention Decoder                      │
│                    ↓                                             │
│  输出: Pointmap P1, P2 (在统一坐标系下的3D点云)                    │
│                    ↓                                             │
│  后处理: 全局对齐 + 位姿估计                                       │
└─────────────────────────────────────────────────────────────────┘

优势:
- 端到端训练，统一优化
- 无需相机内参
- 对弱纹理更鲁棒
- 单次前馈，速度快
```

### 1.2 Pointmap表示（核心创新）

#### 1.2.1 什么是Pointmap？

```
传统深度图: 每个像素存储深度值 Z(u,v)
Pointmap: 每个像素存储3D坐标 X(u,v) = [X, Y, Z]

对比:
┌──────────────────────────────────────────────────────────────┐
│  深度图 (Depth Map)            Pointmap                       │
│                                                              │
│  [z11 z12 z13 ...]            [[x11,y11,z11] [x12,y12,z12]...]│
│  [z21 z22 z23 ...]            [[x21,y21,z21] [x22,y22,z22]...]│
│  [...          ]              [...                          ]│
│                                                              │
│  - 需要相机内参反投影           - 直接是3D坐标                 │
│  - 相机坐标系                   - 统一坐标系                   │
│  - 尺度不确定                   - 绝对尺度 (训练数据决定)        │
└──────────────────────────────────────────────────────────────┘
```

#### 1.2.2 DUSt3R的双Pointmap输出

```
对于图像对 (I1, I2)，DUSt3R输出:

┌─────────────────────────────────────────────────────────────────┐
│  Pointmap1: P1 ∈ R^(H×W×3)                                      │
│  - 图像I1中每个像素的3D坐标                                       │
│  - 坐标系: 以I1的相机中心为原点                                    │
│                                                                 │
│  Pointmap2: P2 ∈ R^(H×W×3)                                      │
│  - 图像I2中每个像素的3D坐标                                       │
│  - 坐标系: 同样以I1的相机中心为原点！！！                          │
│                                                                 │
│  关键: 两个Pointmap在同一坐标系下，直接可配准！                      │
└─────────────────────────────────────────────────────────────────┘

可视化:

图像1          图像2           统一3D坐标系
┌─────┐       ┌─────┐         ╱╲
│ I1  │       │ I2  │        ╱  ╲
└─────┘       └─────┘       ╱ ●  ╲
    ↓             ↓        ╱  │   ╲
   P1             P2      ╱   │    ╲
    │             │      ●────┼─────●
    └──────┬──────┘     ╱     │      ╲
           ▼           P1     │       P2
      统一坐标系             (同一原点)
```

### 1.3 网络架构详解

#### 1.3.1 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    DUSt3R Architecture                           │
└─────────────────────────────────────────────────────────────────┘

输入图像 I1 (H×W×3)          输入图像 I2 (H×W×3)
      │                            │
      ▼                            ▼
┌──────────────┐            ┌──────────────┐
│  ViT Encoder │            │  ViT Encoder │
│  (共享权重)   │            │  (共享权重)   │
└──────────────┘            └──────────────┘
      │                            │
      ▼                            ▼
   F1 ∈ R^(N×D)                 F2 ∈ R^(N×D)
      │                            │
      └──────────┬─────────────────┘
                 ▼
      ┌──────────────────────┐
      │  Cross-Attention     │
      │  Decoder (多层)       │
      │  - 自注意力 (intra)   │
      │  - 交叉注意力 (inter) │
      └──────────────────────┘
                 │
      ┌──────────┴──────────┐
      ▼                     ▼
  Head1                   Head2
      │                     │
      ▼                     ▼
  P1 (H×W×3)            P2 (H×W×3)
  Confidence1           Confidence2
```

#### 1.3.2 CroCo预训练基础

```
DUSt3R基于CroCo (Cross-View Completion)预训练:

CroCo任务:
┌─────────────────────────────────────────────────────────────────┐
│  输入: 图像对 (I1, I2) 从不同视角拍摄同一场景                     │
│  任务: 用I1的信息重建I2的3D几何                                   │
│  本质: 学习跨视角的几何对应关系                                   │
└─────────────────────────────────────────────────────────────────┘

关键设计:
- 非对称解码器: 一个解码器关注I1，另一个关注I2
- 交叉注意力: 实现跨视图信息交换
- 自监督: 无需3D标注，从图像对学习
```

### 1.4 损失函数与训练

#### 1.4.1 回归损失

```
L_regression = Σ |P_pred - P_gt|_1 * Confidence

其中Confidence是网络预测的置信度，用于加权可靠像素
```

#### 1.4.2 置信度损失

```
网络同时预测置信度图 C ∈ [0,1]^(H×W)

L_confidence: 鼓励网络对不确定区域预测低置信度

总损失: L = L_regression + λ * L_confidence
```

### 1.5 与SLAM前端的对比

```
┌─────────────────────────────────────────────────────────────────┐
│              传统SLAM前端 vs DUSt3R                              │
└─────────────────────────────────────────────────────────────────┘

传统SLAM前端 (ORB-SLAM):
┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────┐
│ 特征提取    │──►│ 特征匹配    │──►│ RANSAC验证  │──►│ 位姿估计 │
│ (SIFT/ORB)  │   │ (最近邻)    │   │ (几何约束)  │   │ (EPnP)   │
└─────────────┘   └─────────────┘   └─────────────┘   └──────────┘
   离散              可能错误            鲁棒但慢           需要内参

DUSt3R:
┌─────────────────────────────────────────────────────────────────┐
│                    端到端网络 (单次前馈)                          │
│  输入: 图像对 ──► ViT Encoder ──► Cross-Attention ──► Pointmap  │
│  输出: 直接是3D点云，无需相机内参                                 │
│  置信度: 自动识别可靠/不可靠区域                                  │
└─────────────────────────────────────────────────────────────────┘

对比总结:
┌────────────────┬──────────────────┬──────────────────┐
│     维度       │   传统前端        │     DUSt3R       │
├────────────────┼──────────────────┼──────────────────┤
│ 相机内参       │ 必需             │ 不需要           │
│ 弱纹理区域     │ 容易失败         │ 较鲁棒           │
│ 速度           │ 快 (ms级)        │ 较快 (0.5s)      │
│ 精度           │ 高精度 (像素级)   │ 中等精度         │
│ 可微分性       │ 否 (RANSAC离散)   │ 是               │
│ 端到端训练     │ 否               │ 是               │
└────────────────┴──────────────────┴──────────────────┘
```

---

## 2. 算法详解

### 2.1 下游任务：从Pointmap到位姿

#### 2.1.1 相对位姿估计

```
已知: P1, P2 (在同一坐标系下)
求: T_12 (从相机1到相机2的变换)

方法: Procrustes分析 (SVD)

1. 计算质心:
   c1 = mean(P1)
   c2 = mean(P2)

2. 去质心:
   P1' = P1 - c1
   P2' = P2 - c2

3. SVD:
   U, S, Vt = svd(P2'^T @ P1')

4. 计算旋转和平移:
   R = U @ Vt
   t = c2 - R @ c1

5. 构建变换矩阵:
   T_12 = [R  t]
          [0  1]
```

#### 2.1.2 全局对齐（多视图）

```
对于N张图像，两两运行DUSt3R得到相对位姿

问题: 相对位姿有噪声，需要全局一致

解决方案: 全局位姿图优化 (Pose Graph Optimization)

目标: 最小化所有相对位姿约束的误差

L = Σ_{i,j} ||T_ij^-1 @ T_i^-1 @ T_j - I||^2

其中T_i是第i帧的全局位姿
```

### 2.2 与COLMAP的对比

```
┌─────────────────────────────────────────────────────────────────┐
│                    DUSt3R vs COLMAP                              │
└─────────────────────────────────────────────────────────────────┘

场景: DTU数据集 (扫描物体)

方法          推理时间    精度 (Chamfer Distance)
────────────────────────────────────────────────
COLMAP        25s         0.89mm
DUSt3R        0.8s        0.92mm

结论:
- DUSt3R快30倍
- 精度相当
- 无需相机内参

失败案例分析:
1. 无纹理区域: DUSt3R优于COLMAP
2. 重复纹理: 两者都可能失败
3. 大基线: DUSt3R更鲁棒
4. 尺度模糊: DUSt3R依赖训练数据尺度
```

---

## 3. 代码实践指南

### 3.1 环境搭建

```bash
# 克隆代码
git clone https://github.com/naver/dust3r.git
cd dust3r

# 创建环境
conda create -n dust3r python=3.11
conda activate dust3r

# 安装依赖
pip install torch==2.2.0 torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 下载预训练模型
mkdir -p checkpoints
# 从Hugging Face下载 DUSt3R_ViTLarge_BaseDec512Dpt.pth
```

### 3.2 关键代码走读

#### 3.2.1 模型定义

```python
class AsymmetricCroCo3DStereo(nn.Module):
    """
    DUSt3R核心模型
    """
    def __init__(self, img_size=512, patch_size=16, embed_dim=1024,
                 enc_depth=24, dec_depth=12, num_heads=16):
        super().__init__()
        
        # 图像分块嵌入
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim)
        
        # 编码器 (ViT)
        self.enc_blocks = nn.ModuleList([
            Block(embed_dim, num_heads) for _ in range(enc_depth)
        ])
        
        # 解码器 (Cross-Attention)
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads) for _ in range(dec_depth)
        ])
        self.dec_blocks2 = nn.ModuleList([
            DecoderBlock(embed_dim, num_heads) for _ in range(dec_depth)
        ])
        
        # 输出头
        self.head = DPTHead(embed_dim)  # 预测Pointmap
        self.head2 = DPTHead(embed_dim)
        
    def forward(self, img1, img2):
        # 编码
        feat1 = self.encode_image(img1)  # [B, N, D]
        feat2 = self.encode_image(img2)
        
        # 解码 (非对称)
        feat1_dec = self.decode(feat1, feat2, self.dec_blocks)
        feat2_dec = self.decode(feat2, feat1, self.dec_blocks2)
        
        # 预测Pointmap
        pointmap1 = self.head(feat1_dec)  # [B, H, W, 3]
        pointmap2 = self.head2(feat2_dec)
        
        # 预测置信度
        conf1 = self.predict_confidence(feat1_dec)
        conf2 = self.predict_confidence(feat2_dec)
        
        return pointmap1, pointmap2, conf1, conf2
```

#### 3.2.2 推理代码

```python
def inference(model, img1_path, img2_path, device='cuda'):
    """
    运行DUSt3R推理
    """
    # 加载图像
    img1 = load_image(img1_path).to(device)
    img2 = load_image(img2_path).to(device)
    
    # 前向传播
    with torch.no_grad():
        pred1, pred2, conf1, conf2 = model(img1, img2)
    
    # 提取Pointmap
    pointmap1 = pred1['pts3d']  # [H, W, 3]
    pointmap2 = pred2['pts3d_in_other_view']  # 在img1坐标系下
    
    return pointmap1, pointmap2, conf1, conf2
```

### 3.3 运行实验

```python
# 示例：两视图重建
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo

# 加载模型
model = AsymmetricCroCo3DStereo.from_pretrained(
    'checkpoints/DUSt3R_ViTLarge_BaseDec512Dpt.pth'
).to('cuda')

# 推理
pointmap1, pointmap2, conf1, conf2 = inference(
    model, 'image1.jpg', 'image2.jpg'
)

# 可视化
import open3d as o3d

# 转换为点云
pcd1 = o3d.geometry.PointCloud()
pcd1.points = o3d.utility.Vector3dVector(pointmap1.reshape(-1, 3).cpu().numpy())
pcd1.colors = o3d.utility.Vector3dVector(load_image('image1.jpg').reshape(-1, 3))

pcd2 = o3d.geometry.PointCloud()
pcd2.points = o3d.utility.Vector3dVector(pointmap2.reshape(-1, 3).cpu().numpy())

# 显示
o3d.visualization.draw_geometries([pcd1, pcd2])
```

---

## 4. 实验与检查点

### 4.1 实验1: 与COLMAP对比

```python
def compare_with_colmap(img1_path, img2_path):
    """
    对比DUSt3R与COLMAP
    """
    # DUSt3R
    start = time.time()
    pcd_dust3r = run_dust3r(img1_path, img2_path)
    time_dust3r = time.time() - start
    
    # COLMAP
    start = time.time()
    pcd_colmap = run_colmap(img1_path, img2_path)
    time_colmap = time.time() - start
    
    print(f"DUSt3R: {time_dust3r:.2f}s")
    print(f"COLMAP: {time_colmap:.2f}s")
    print(f"Speedup: {time_colmap/time_dust3r:.1f}x")
```

### 4.2 实验2: 置信度分析

```python
def analyze_confidence(pointmap, confidence, threshold=0.5):
    """
    分析置信度图
    """
    # 高置信度点
    high_conf_mask = confidence > threshold
    high_conf_points = pointmap[high_conf_mask]
    
    # 低置信度点 (通常是无纹理区域或遮挡)
    low_conf_mask = confidence <= threshold
    low_conf_points = pointmap[low_conf_mask]
    
    print(f"High confidence: {high_conf_mask.sum()} pixels")
    print(f"Low confidence: {low_conf_mask.sum()} pixels")
    
    # 可视化
    visualize_pointcloud(high_conf_points, color='green')
    visualize_pointcloud(low_conf_points, color='red')
```

### 4.3 ✅ 完成检查点

- [ ] **检查点1**: 能解释DUSt3R与传统SfM的区别
  ```
  核心区别:
  1. 端到端学习 vs 多阶段pipeline
  2. 无需相机内参
  3. Pointmap表示 vs 深度图+位姿
  4. 置信度预测 vs RANSAC内点率
  ```

- [ ] **检查点2**: 理解Confidence Map的作用
  ```
  Confidence Map:
  - 网络预测的每个像素的可靠性
  - 用于加权损失函数
  - 后处理中过滤低置信度点
  - 类似于SLAM中的"匹配内点率"
  ```

- [ ] **检查点3**: 成功运行DUSt3R并可视化结果
  ```python
  # 运行代码
  python demo.py --image1 img1.jpg --image2 img2.jpg
  
  # 预期输出:
  # - Pointmap可视化
  # - 置信度热力图
  # - 相对位姿估计
  ```

---

## 5. 常见问题与解答

### Q1: DUSt3R需要相机内参吗？

**A**: 不需要！这是其核心优势之一。网络直接从图像学习几何。

### Q2: DUSt3R的尺度如何确定？

**A**: 尺度来自训练数据（ScanNet等）。绝对尺度不确定，需要后续对齐。

### Q3: 如何处理多视图（>2张图像）？

**A**: 两两运行DUSt3R，然后进行全局对齐。见Fast3R/VGGT的改进。

---

## 6. 延伸阅读

### 必读论文

1. **DUSt3R**: "DUSt3R: Geometric 3D Vision Made Easy" CVPR 2024
   - [arXiv:2312.14132](https://arxiv.org/abs/2312.14132)

2. **CroCo**: "CroCo: Self-Supervised Pretraining for 3D Vision Tasks"
   - [arXiv:2210.02567](https://arxiv.org/abs/2210.02567)

3. **MASt3R**: "Grounding Image Matching in 3D with MASt3R" ECCV 2024
   - 改进的匹配方法

### 下一步

- [Phase 4: Feed-forward Gaussian](./Phase4_FeedForward_Gaussian.md) - 学习前馈3DGS
- [Phase 5: VGGT](./Phase5_VGGT.md) - 学习多视图基础模型
