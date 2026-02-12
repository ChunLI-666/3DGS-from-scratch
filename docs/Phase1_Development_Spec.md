# Phase 1: 3DGS基础教学材料 - 需求规格文档

> **文档版本**: v1.0
> **创建日期**: 2026-02-12
> **目标读者**: 3DGS初学者（假设有SLAM基础）

---

## 1. 项目目标

### 1.1 总体目标
为3DGS初学者提供一套**可运行、可视化、开箱即用**的交互式教学材料，让学习者能够：
- 从零开始理解3DGS的核心原理
- 通过可视化手段直观理解抽象概念
- 动手运行和修改代码，加深理解
- 快速部署经典论文的开源实现

### 1.2 设计原则
1. **开箱即用**: 环境配置一键完成，减少配置门槛
2. **循序渐进**: 从简单概念到复杂实现，层层递进
3. **可视化优先**: 每个概念都配有交互式可视化
4. **代码可运行**: 所有代码片段都经过测试，可直接执行
5. **与论文代码集成**: 支持一键部署官方实现

---

## 2. 仓库结构设计

```
3DGS-from-scratch/
├── README.md                           # 项目总览和快速开始
├── requirements.txt                    # 基础Python依赖
├── setup.py                            # 包安装配置
├── .gitignore
├── .github/
│   └── workflows/
│       └── test_notebooks.yml          # CI测试notebook
│
├── docs/                               # 文档目录
│   ├── Phase1_Development_Spec.md      # 本需求文档
│   ├── CONTRIBUTING.md                 # 贡献指南
│   └── CHANGELOG.md                    # 更新日志
│
├── tutorials/                          # 原有的学习路线图 (Markdown)
│   ├── README.md
│   ├── Phase1_3DGS_Foundation.md
│   ├── Phase2_3DGS_SLAM.md
│   └── ...
│
├── notebooks/                          # 交互式教程 (Jupyter Notebooks)
│   ├── phase1/                         # Phase 1 教程
│   │   ├── 00_environment_setup.ipynb  # 环境配置
│   │   ├── 01_gaussian_basics.ipynb    # 高斯基础
│   │   ├── 02_3d_gaussian_math.ipynb   # 3D高斯数学
│   │   ├── 03_projection_splatting.ipynb # 投影与Splatting
│   │   ├── 04_differentiable_rendering.ipynb # 可微分渲染
│   │   ├── 05_alpha_blending.ipynb     # Alpha混合
│   │   ├── 06_spherical_harmonics.ipynb # 球谐函数
│   │   ├── 07_adaptive_density.ipynb   # 自适应密度控制
│   │   ├── 08_training_pipeline.ipynb  # 完整训练流程
│   │   ├── 09_official_code_walkthrough.ipynb # 官方代码走读
│   │   └── 10_custom_data_training.ipynb # 自定义数据训练
│   │
│   ├── phase2/                         # Phase 2 教程 (后续开发)
│   ├── phase3/                         # Phase 3 教程 (后续开发)
│   └── ...
│
├── src/                                # 教学用源代码
│   ├── __init__.py
│   ├── gaussian/                       # 高斯相关模块
│   │   ├── __init__.py
│   │   ├── gaussian_model.py           # 简化版高斯模型
│   │   ├── covariance.py               # 协方差计算
│   │   └── projection.py               # 投影函数
│   │
│   ├── rendering/                      # 渲染相关模块
│   │   ├── __init__.py
│   │   ├── rasterizer.py               # 纯Python光栅化器
│   │   ├── alpha_blending.py           # Alpha混合
│   │   └── differentiable_render.py    # 可微分渲染
│   │
│   ├── spherical_harmonics/            # 球谐函数
│   │   ├── __init__.py
│   │   └── sh_utils.py
│   │
│   ├── training/                       # 训练相关
│   │   ├── __init__.py
│   │   ├── loss.py                     # 损失函数
│   │   ├── optimizer.py                # 优化器配置
│   │   └── density_control.py          # 密度控制
│   │
│   └── visualization/                  # 可视化工具
│       ├── __init__.py
│       ├── gaussian_viz.py             # 高斯可视化
│       ├── training_viz.py             # 训练过程可视化
│       └── interactive_plots.py        # 交互式图表
│
├── repos/                              # 经典论文代码 (Git Submodules)
│   ├── README.md                       # 各仓库说明
│   ├── gaussian-splatting/             # 官方3DGS (submodule)
│   └── diff-gaussian-rasterization/    # CUDA光栅化器 (submodule)
│
├── scripts/                            # 部署和工具脚本
│   ├── setup_env.sh                    # 一键环境配置
│   ├── download_sample_data.sh         # 下载示例数据
│   ├── install_official_3dgs.sh        # 安装官方3DGS
│   └── run_training.sh                 # 运行训练示例
│
├── data/                               # 数据目录
│   ├── .gitkeep
│   ├── sample_images/                  # 示例图片
│   └── sample_scenes/                  # 示例场景
│
├── configs/                            # 配置文件
│   ├── default.yaml                    # 默认配置
│   └── training/
│       ├── quick_test.yaml             # 快速测试配置
│       └── full_training.yaml          # 完整训练配置
│
├── tests/                              # 测试代码
│   ├── test_gaussian.py
│   ├── test_rendering.py
│   └── test_notebooks.py
│
└── docker/                             # Docker环境
    ├── Dockerfile                      # 基础镜像
    ├── Dockerfile.cuda                 # CUDA版本
    └── docker-compose.yml              # 编排配置
```

---

## 3. Phase 1 Notebook 详细设计

### 3.1 总览

| Notebook | 主题 | 预计时长 | 难度 |
|----------|------|---------|------|
| 00 | 环境配置 | 30min | ★☆☆☆☆ |
| 01 | 高斯分布基础 | 45min | ★★☆☆☆ |
| 02 | 3D高斯椭球数学 | 60min | ★★★☆☆ |
| 03 | 投影与Splatting | 60min | ★★★☆☆ |
| 04 | 可微分渲染 | 90min | ★★★★☆ |
| 05 | Alpha混合与深度排序 | 60min | ★★★☆☆ |
| 06 | 球谐函数与外观建模 | 75min | ★★★★☆ |
| 07 | 自适应密度控制 | 60min | ★★★★☆ |
| 08 | 完整训练流程 | 90min | ★★★★★ |
| 09 | 官方代码走读 | 120min | ★★★★★ |
| 10 | 自定义数据训练 | 90min | ★★★☆☆ |

---

### 3.2 各Notebook详细内容

#### Notebook 00: 环境配置 (environment_setup.ipynb)

**目标**: 一键配置完整的开发环境

**内容**:
1. **环境检测**
   - 检查Python版本
   - 检查CUDA版本和GPU信息
   - 检查可用显存

2. **依赖安装**
   - 基础依赖 (numpy, torch, matplotlib)
   - 可视化依赖 (plotly, ipywidgets)
   - 3DGS特定依赖

3. **验证安装**
   - 导入测试
   - GPU加速测试
   - 渲染测试

**可视化**:
- GPU信息卡片
- 环境检查进度条
- 安装状态总览表

**代码片段**:
```python
# 环境检测示例
import torch
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU设备: {torch.cuda.get_device_name(0)}")
print(f"可用显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

#### Notebook 01: 高斯分布基础 (gaussian_basics.ipynb)

**目标**: 从1D到3D，建立高斯分布的直觉

**内容**:
1. **1D高斯分布**
   - 概率密度函数
   - 均值和方差的意义
   - 交互式滑块调节参数

2. **2D高斯分布**
   - 协方差矩阵的引入
   - 等高线可视化
   - 相关性与椭圆形状的关系

3. **3D高斯分布预览**
   - 从2D到3D的扩展
   - 3D可视化初探

**可视化**:
- 1D高斯曲线 (交互式，可调节μ和σ)
- 2D高斯热力图 + 等高线
- 2D高斯的3D表面图
- 协方差矩阵对形状的影响动画

**关键公式**:
$$G(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

$$G(\mathbf{x}) = \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\Sigma^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

---

#### Notebook 02: 3D高斯椭球数学 (3d_gaussian_math.ipynb)

**目标**: 深入理解3D高斯的数学表示

**内容**:
1. **高斯椭球的参数化**
   - 中心位置 μ ∈ ℝ³
   - 协方差矩阵 Σ ∈ ℝ³ˣ³
   - 为什么需要正定性

2. **协方差分解: Σ = RSS^TR^T**
   - 缩放矩阵 S = diag(s_x, s_y, s_z)
   - 旋转矩阵 R ∈ SO(3)
   - 四元数表示旋转

3. **参数可视化**
   - 3D交互式椭球可视化
   - 调节S观察形状变化
   - 调节R观察朝向变化

4. **与SLAM的对比**
   - 点云中的surfel
   - 协方差与不确定性椭球

**可视化**:
- 3D椭球渲染 (使用plotly/pyvista)
- 参数调节滑块 (s_x, s_y, s_z, qw, qx, qy, qz)
- 多个高斯同时显示
- 协方差矩阵热力图

**代码实现**:
```python
class Gaussian3D:
    def __init__(self, mean, scaling, rotation_quat):
        self.mean = mean  # [3]
        self.scaling = scaling  # [3]
        self.rotation = rotation_quat  # [4] (w, x, y, z)

    def get_covariance(self):
        """计算协方差矩阵 Σ = R @ S @ S.T @ R.T"""
        R = quaternion_to_rotation_matrix(self.rotation)
        S = torch.diag(self.scaling)
        return R @ S @ S.T @ R.T
```

---

#### Notebook 03: 投影与Splatting (projection_splatting.ipynb)

**目标**: 理解3D高斯如何投影到2D图像

**内容**:
1. **相机模型回顾**
   - 内参矩阵 K
   - 外参矩阵 [R|t]
   - 透视投影

2. **3D高斯投影**
   - 中心点投影: μ_2d = project(μ_3d)
   - 协方差投影: Σ' = J @ W @ Σ @ W^T @ J^T
   - 仿射近似与雅可比矩阵

3. **Splatting可视化**
   - 单个高斯的投影
   - 多个高斯的叠加效果
   - 不同视角下的变化

**可视化**:
- 3D场景 + 相机视锥
- 投影过程动画
- 2D图像平面上的高斯splat
- 交互式相机位置调节

**关键代码**:
```python
def project_gaussian(gaussian_3d, camera):
    """将3D高斯投影到2D"""
    # 1. 变换到相机坐标系
    mean_cam = camera.transform_point(gaussian_3d.mean)

    # 2. 计算投影雅可比
    J = compute_projection_jacobian(mean_cam, camera.K)

    # 3. 投影协方差
    cov_3d = gaussian_3d.get_covariance()
    R_cam = camera.R
    cov_cam = R_cam @ cov_3d @ R_cam.T
    cov_2d = J @ cov_cam @ J.T

    # 4. 投影中心
    mean_2d = camera.project(gaussian_3d.mean)

    return Gaussian2D(mean_2d, cov_2d)
```

---

#### Notebook 04: 可微分渲染 (differentiable_rendering.ipynb)

**目标**: 理解渲染过程的可微分性

**内容**:
1. **前向渲染 vs 逆向渲染**
   - 3DGS: 从3D到2D (前向)
   - NeRF: 从2D ray采样3D (逆向)

2. **光栅化流程**
   - 高斯排序 (按深度)
   - 像素覆盖判断
   - 颜色计算

3. **梯度传播**
   - 每个参数的梯度计算
   - 链式法则应用
   - PyTorch自动微分演示

4. **与传统方法对比**
   - 不可微分的地方 (SLAM特征匹配)
   - 为什么可微分很重要

**可视化**:
- 渲染流程图 (动画)
- 梯度流向可视化
- 参数更新对渲染的影响

**代码实现**:
```python
class DifferentiableRasterizer(torch.nn.Module):
    """纯Python实现的可微分光栅化器（教学用）"""

    def forward(self, gaussians, camera, image_size):
        """
        前向渲染
        """
        H, W = image_size
        image = torch.zeros(3, H, W)

        # 投影所有高斯
        gaussians_2d = [project_gaussian(g, camera) for g in gaussians]

        # 按深度排序
        sorted_indices = sort_by_depth(gaussians, camera)

        # 逐像素渲染
        for y in range(H):
            for x in range(W):
                image[:, y, x] = self.render_pixel(
                    x, y, gaussians_2d, sorted_indices
                )

        return image
```

---

#### Notebook 05: Alpha混合与深度排序 (alpha_blending.ipynb)

**目标**: 深入理解Alpha混合渲染

**内容**:
1. **Alpha混合原理**
   - 透明度与不透明度
   - 前向混合公式
   - 透射率累积

2. **公式推导**
   $$C = \sum_{i=1}^{N} c_i \cdot \alpha_i \cdot T_i$$
   $$T_i = \prod_{j=1}^{i-1}(1-\alpha_j)$$

3. **深度排序的重要性**
   - 排序 vs 不排序的区别
   - 排序算法选择

4. **实现与优化**
   - 逐像素实现
   - 批量实现
   - GPU优化思路

**可视化**:
- 透明球叠加效果
- 排序前后对比
- 透射率衰减曲线
- 交互式Alpha调节

---

#### Notebook 06: 球谐函数与外观建模 (spherical_harmonics.ipynb)

**目标**: 理解视角相关外观的建模方法

**内容**:
1. **为什么需要球谐函数**
   - 视角相关的颜色变化
   - 镜面反射、高光

2. **球谐函数基础**
   - 球面上的正交基
   - 不同阶数的意义
   - L=0,1,2,3的可视化

3. **3DGS中的SH应用**
   - 每个高斯的SH系数
   - 根据视角计算颜色
   - 训练过程中的渐进策略

4. **实现与存储**
   - SH系数格式
   - 颜色计算代码

**可视化**:
- 球谐基函数3D可视化
- 不同SH阶数的渲染效果对比
- 视角变化时的颜色变化动画

**代码实现**:
```python
def eval_sh(deg, sh_coeffs, view_dir):
    """
    计算给定视角方向的颜色

    Args:
        deg: SH阶数 (0-3)
        sh_coeffs: SH系数 [n_coeffs, 3]
        view_dir: 视角方向 [3]

    Returns:
        color: RGB颜色 [3]
    """
    result = SH_C0 * sh_coeffs[0]

    if deg > 0:
        x, y, z = view_dir
        result = result + SH_C1 * (-y * sh_coeffs[1] + z * sh_coeffs[2] - x * sh_coeffs[3])

    if deg > 1:
        # ... 更高阶项

    return result + 0.5  # 加上0.5偏移
```

---

#### Notebook 07: 自适应密度控制 (adaptive_density.ipynb)

**目标**: 理解3DGS的核心创新之一

**内容**:
1. **为什么需要密度控制**
   - 初始化不足的区域
   - 过度重建的区域

2. **三种操作**
   - 分裂 (Split): 大高斯变小
   - 克隆 (Clone): 增加覆盖
   - 剪枝 (Prune): 移除无用高斯

3. **触发条件**
   - view-space gradient阈值
   - 高斯大小判断
   - 不透明度阈值

4. **与SLAM的类比**
   - 关键帧选择
   - 地图点culling

**可视化**:
- 训练过程中高斯数量变化曲线
- Split/Clone/Prune过程动画
- 梯度分布热力图
- 高斯分布演变动画

---

#### Notebook 08: 完整训练流程 (training_pipeline.ipynb)

**目标**: 整合所有概念，实现完整训练

**内容**:
1. **数据准备**
   - 图像加载
   - 相机位姿
   - 初始点云

2. **模型初始化**
   - 从点云初始化高斯
   - 参数设置

3. **训练循环**
   - 前向渲染
   - 损失计算 (L1 + SSIM)
   - 反向传播
   - 密度控制

4. **监控与可视化**
   - 损失曲线
   - 渲染质量变化
   - 高斯数量变化

**代码实现**: 完整的训练脚本，约200行

---

#### Notebook 09: 官方代码走读 (official_code_walkthrough.ipynb)

**目标**: 深入理解官方实现

**内容**:
1. **代码结构概览**
   - 目录结构
   - 关键文件

2. **GaussianModel类详解**
   - 参数定义
   - 协方差计算
   - 优化器设置

3. **渲染器详解**
   - CUDA光栅化接口
   - 参数传递

4. **训练脚本详解**
   - train.py流程
   - 关键超参数

**代码片段**: 带详细注释的官方代码关键部分

---

#### Notebook 10: 自定义数据训练 (custom_data_training.ipynb)

**目标**: 在自己的数据上训练3DGS

**内容**:
1. **数据采集指南**
   - 拍摄技巧
   - 图片数量建议

2. **COLMAP位姿估计**
   - 安装COLMAP
   - 运行SfM
   - 结果解析

3. **数据转换**
   - 转换为3DGS格式
   - 验证数据正确性

4. **训练与评估**
   - 运行训练
   - 结果可视化
   - 常见问题排查

---

## 4. 经典论文代码集成

### 4.1 官方3DGS仓库

**仓库**: https://github.com/graphdeco-inria/gaussian-splatting

**集成方式**: Git Submodule

**一键部署脚本** (`scripts/install_official_3dgs.sh`):
```bash
#!/bin/bash
set -e

echo "=== 安装官方3DGS ==="

# 1. 克隆仓库
cd repos
git submodule update --init gaussian-splatting

# 2. 创建conda环境
conda create -n gaussian_splatting python=3.9 -y
conda activate gaussian_splatting

# 3. 安装PyTorch
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# 4. 安装依赖
cd gaussian-splatting
pip install -r requirements.txt

# 5. 编译CUDA模块
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

echo "=== 安装完成 ==="
```

### 4.2 示例数据下载

**脚本** (`scripts/download_sample_data.sh`):
```bash
#!/bin/bash
# 下载官方示例数据集

mkdir -p data/sample_scenes
cd data/sample_scenes

# 下载小型测试场景
wget https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip
unzip tandt_db.zip
```

---

## 5. 环境配置方案

### 5.1 推荐配置

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.9-3.11 | 推荐3.10 |
| PyTorch | 2.0+ | 需要CUDA支持 |
| CUDA | 11.7/11.8/12.1 | 根据显卡选择 |
| NumPy | 1.24+ | |
| Matplotlib | 3.7+ | |
| Plotly | 5.15+ | 交互式3D可视化 |
| ipywidgets | 8.0+ | Jupyter交互组件 |

### 5.2 Docker方案

**Dockerfile.cuda**:
```dockerfile
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# 基础依赖
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip git wget \
    && rm -rf /var/lib/apt/lists/*

# Python依赖
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

# Jupyter配置
RUN pip3 install jupyterlab
EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
```

### 5.3 Colab支持

每个Notebook顶部添加Colab兼容代码:
```python
# @title 环境配置 (Colab)
import os
if 'COLAB_GPU' in os.environ:
    !pip install -q plotly ipywidgets
    !git clone https://github.com/ChunLI-666/3DGS-from-scratch.git
    %cd 3DGS-from-scratch
```

---

## 6. 可视化工具设计

### 6.1 可视化库选择

| 场景 | 库 | 原因 |
|------|------|------|
| 2D图表 | matplotlib | 标准、兼容性好 |
| 交互式2D | plotly | 支持缩放、悬停 |
| 3D可视化 | plotly / pyvista | 交互式3D |
| 动画 | matplotlib.animation | 简单动画 |
| 交互组件 | ipywidgets | 滑块、按钮 |

### 6.2 可视化模块设计

```python
# src/visualization/gaussian_viz.py

class GaussianVisualizer:
    """高斯可视化工具"""

    def plot_2d_gaussian(self, mean, cov, ax=None):
        """绘制2D高斯等高线"""
        pass

    def plot_3d_ellipsoid(self, gaussian, fig=None):
        """绘制3D高斯椭球"""
        pass

    def plot_multiple_gaussians(self, gaussians, camera=None):
        """绘制多个高斯"""
        pass

    def animate_projection(self, gaussian, camera_trajectory):
        """动画展示投影过程"""
        pass


class TrainingVisualizer:
    """训练过程可视化"""

    def plot_loss_curve(self, losses):
        """绘制损失曲线"""
        pass

    def plot_gaussian_count(self, counts):
        """绘制高斯数量变化"""
        pass

    def render_comparison(self, gt, rendered):
        """GT与渲染对比"""
        pass
```

---

## 7. 开发计划

### 7.1 第一阶段: 基础框架 (Week 1)

- [ ] 创建仓库结构
- [ ] 编写环境配置脚本
- [ ] 创建基础src模块框架
- [ ] 编写Notebook 00 (环境配置)
- [ ] 编写Notebook 01 (高斯基础)

### 7.2 第二阶段: 核心概念 (Week 2-3)

- [ ] 编写Notebook 02 (3D高斯数学)
- [ ] 编写Notebook 03 (投影与Splatting)
- [ ] 编写Notebook 04 (可微分渲染)
- [ ] 编写Notebook 05 (Alpha混合)
- [ ] 实现src/gaussian模块
- [ ] 实现src/rendering模块

### 7.3 第三阶段: 进阶内容 (Week 4-5)

- [ ] 编写Notebook 06 (球谐函数)
- [ ] 编写Notebook 07 (自适应密度控制)
- [ ] 编写Notebook 08 (完整训练流程)
- [ ] 实现src/spherical_harmonics模块
- [ ] 实现src/training模块

### 7.4 第四阶段: 集成与测试 (Week 6)

- [ ] 编写Notebook 09 (官方代码走读)
- [ ] 编写Notebook 10 (自定义数据训练)
- [ ] 集成官方3DGS代码
- [ ] 编写Docker配置
- [ ] 全面测试与调试

---

## 8. 质量标准

### 8.1 代码标准
- 所有代码可直接运行
- 关键代码有详细注释
- 遵循PEP8规范
- 包含类型提示

### 8.2 文档标准
- 每个Notebook有明确的学习目标
- 概念解释清晰，配有图示
- 与SLAM的对比分析
- 常见问题解答

### 8.3 测试标准
- 在Colab测试通过
- 在本地环境测试通过
- Docker环境测试通过

---

## 9. 下一步行动

请确认以上需求规格，我将按以下顺序开发：

1. **首先**: 创建仓库结构
2. **然后**: 开发Notebook 00 (环境配置)
3. **接着**: 开发Notebook 01 (高斯基础)

是否需要调整任何部分？
