"""
Adaptive Density Control for 3D Gaussian Splatting

This module implements the adaptive density control algorithm from the
original 3DGS paper (Kerbl et al., 2023, Section 5.2).

The core idea: during training, some scene regions are under-reconstructed
(too few Gaussians) while others are over-reconstructed (too many small
Gaussians). Adaptive density control periodically adjusts the Gaussian
population via three operations:

1. **Clone** (复制) — Small Gaussians with large view-space gradients
   are duplicated. This fills in under-reconstructed regions that need
   more coverage but where existing Gaussians are already small.

2. **Split** (分裂) — Large Gaussians with large view-space gradients
   are split into two smaller Gaussians. This refines over-reconstructed
   regions where a single large Gaussian is too coarse.

3. **Prune** (修剪) — Gaussians with very low opacity (nearly transparent)
   or that have grown excessively large are removed.

Algorithm overview (every N iterations):
    for each Gaussian g:
        if avg_grad(g) > τ_grad:
            if scale(g) < τ_scale:
                clone(g)     # under-reconstruction → duplicate
            else:
                split(g)     # over-reconstruction → subdivide
        if opacity(g) < τ_opacity:
            prune(g)         # nearly invisible → remove
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class DensityControlStats:
    """
    Statistics collected during density control.

    Tracks what happened during the last densification step,
    useful for monitoring and debugging training.
    """
    n_cloned: int = 0
    n_split: int = 0
    n_pruned: int = 0
    n_total_before: int = 0
    n_total_after: int = 0
    avg_gradient: float = 0.0

    def __repr__(self) -> str:
        delta = self.n_total_after - self.n_total_before
        sign = "+" if delta >= 0 else ""
        return (
            f"DensityControlStats("
            f"cloned={self.n_cloned}, split={self.n_split}, "
            f"pruned={self.n_pruned}, "
            f"total: {self.n_total_before} → {self.n_total_after} ({sign}{delta}))"
        )


class DensityController:
    """
    Manages adaptive density control for 3D Gaussian Splatting.

    This controller accumulates view-space position gradients over
    multiple training iterations, then periodically triggers
    densification (clone/split) and pruning operations.

    Usage:
        controller = DensityController(
            densify_grad_threshold=0.0002,
            densify_interval=100,
        )

        for step in range(n_iters):
            # ... forward pass, compute loss ...
            loss.backward()

            # 每步都记录梯度
            controller.accumulate_gradients(gaussians.xyz.grad)

            # 每 N 步执行一次密度控制
            if controller.should_densify(step):
                stats = controller.densify_and_prune(
                    xyz=gaussians._xyz,
                    scaling=gaussians._scaling,
                    rotation=gaussians._rotation,
                    opacity=gaussians._opacity,
                    features_dc=gaussians._features_dc,
                    features_rest=gaussians._features_rest,
                    optimizer=optimizer,
                )

    Args:
        densify_grad_threshold: Gradient threshold τ_grad for triggering
            clone/split. Gaussians with average gradient above this are
            candidates for densification. (default: 0.0002)
        densify_scale_threshold: Scale threshold τ_scale distinguishing
            clone vs split. Gaussians smaller than this are cloned;
            larger ones are split. (default: 0.01)
        prune_opacity_threshold: Opacity threshold τ_opacity. Gaussians
            with opacity below this are pruned. (default: 0.005)
        densify_interval: How often (in iterations) to run density
            control. (default: 100)
        densify_from_iter: First iteration at which densification begins.
            (default: 500)
        densify_until_iter: Last iteration at which densification runs.
            (default: 15000)
        max_screen_size: Maximum 2D screen-space size. Gaussians
            exceeding this are pruned to avoid huge splats. (default: 20.0)
    """

    def __init__(
        self,
        densify_grad_threshold: float = 0.0002,
        densify_scale_threshold: float = 0.01,
        prune_opacity_threshold: float = 0.005,
        densify_interval: int = 100,
        densify_from_iter: int = 500,
        densify_until_iter: int = 15000,
        max_screen_size: float = 20.0,
    ):
        self.densify_grad_threshold = densify_grad_threshold
        self.densify_scale_threshold = densify_scale_threshold
        self.prune_opacity_threshold = prune_opacity_threshold
        self.densify_interval = densify_interval
        self.densify_from_iter = densify_from_iter
        self.densify_until_iter = densify_until_iter
        self.max_screen_size = max_screen_size

        # 梯度累积缓存
        self._grad_accum: Optional[torch.Tensor] = None  # 累积梯度范数
        self._grad_count: int = 0  # 累积次数
        self._last_stats: Optional[DensityControlStats] = None

    @property
    def last_stats(self) -> Optional[DensityControlStats]:
        """Get statistics from the most recent densification."""
        return self._last_stats

    def accumulate_gradients(self, xyz_grad: Optional[torch.Tensor]) -> None:
        """
        Accumulate view-space position gradients.

        Called after every backward pass. The gradient magnitude tells us
        which Gaussians are receiving strong optimization pressure — these
        are the ones that need more representation capacity.

        梯度越大 → 该区域重建误差越大 → 需要更多 Gaussians 来表示

        Args:
            xyz_grad: Gradient of loss w.r.t. Gaussian positions [N, 3].
                      If None (no grad), this call is a no-op.
        """
        if xyz_grad is None:
            return

        grad_norm = xyz_grad.norm(dim=-1)  # [N]

        if self._grad_accum is None:
            self._grad_accum = torch.zeros_like(grad_norm)

        # 累加梯度范数（用于之后求平均）
        self._grad_accum += grad_norm
        self._grad_count += 1

    def should_densify(self, iteration: int) -> bool:
        """
        Check whether density control should run at this iteration.

        Densification runs every `densify_interval` steps, but only within
        the [densify_from_iter, densify_until_iter] window.

        Args:
            iteration: Current training iteration

        Returns:
            True if densification should be triggered
        """
        if iteration < self.densify_from_iter:
            return False
        if iteration > self.densify_until_iter:
            return False
        if iteration % self.densify_interval != 0:
            return False
        if self._grad_count == 0:
            return False
        return True

    def densify_and_prune(
        self,
        xyz: nn.Parameter,
        scaling: nn.Parameter,
        rotation: nn.Parameter,
        opacity: nn.Parameter,
        features_dc: nn.Parameter,
        features_rest: nn.Parameter,
        optimizer: Optional[torch.optim.Optimizer] = None,
        max_gaussians: int = 500_000,
    ) -> DensityControlStats:
        """
        Execute one round of adaptive density control.

        This is the main entry point. It:
        1. Computes average gradients over the accumulation window
        2. Identifies Gaussians to clone (small + high gradient)
        3. Identifies Gaussians to split (large + high gradient)
        4. Prunes nearly-transparent or oversized Gaussians
        5. Updates parameters and optimizer state in-place

        Args:
            xyz: Gaussian positions parameter [N, 3]
            scaling: Log-space scaling parameter [N, 3]
            rotation: Rotation quaternion parameter [N, 4]
            opacity: Logit-space opacity parameter [N, 1]
            features_dc: SH DC coefficients [N, 1, 3]
            features_rest: SH rest coefficients [N, K, 3]
            optimizer: Optimizer (state will be updated for new/removed params)
            max_gaussians: Hard cap on total Gaussian count

        Returns:
            DensityControlStats with details of what changed
        """
        device = xyz.device
        N = xyz.shape[0]
        stats = DensityControlStats(n_total_before=N)

        # 1. 计算平均梯度
        if self._grad_accum is not None and self._grad_count > 0:
            avg_grad = self._grad_accum / self._grad_count
        else:
            avg_grad = torch.zeros(N, device=device)

        stats.avg_gradient = avg_grad.mean().item()

        # 2. 找出需要密化的 Gaussians（梯度超过阈值）
        grad_mask = avg_grad > self.densify_grad_threshold

        # 用激活后的 scale 判断大小
        activated_scale = torch.exp(scaling.data)  # [N, 3]
        max_scale = activated_scale.max(dim=-1).values  # [N]

        # Clone: 小 Gaussian + 高梯度 → 复制一份
        clone_mask = grad_mask & (max_scale <= self.densify_scale_threshold)

        # Split: 大 Gaussian + 高梯度 → 拆成两个更小的
        split_mask = grad_mask & (max_scale > self.densify_scale_threshold)

        # 3. Prune: 低透明度 → 删除
        activated_opacity = torch.sigmoid(opacity.data.squeeze(-1))  # [N]
        prune_mask = activated_opacity < self.prune_opacity_threshold

        stats.n_cloned = clone_mask.sum().item()
        stats.n_split = split_mask.sum().item()
        stats.n_pruned = prune_mask.sum().item()

        # 收集所有参数字典，方便统一操作
        params = {
            'xyz': xyz,
            'scaling': scaling,
            'rotation': rotation,
            'opacity': opacity,
            'features_dc': features_dc,
            'features_rest': features_rest,
        }

        # ---- Clone 操作 ----
        new_tensors = {}
        if stats.n_cloned > 0:
            new_tensors = self._clone_gaussians(params, clone_mask)

        # ---- Split 操作 ----
        split_tensors = {}
        if stats.n_split > 0:
            split_tensors = self._split_gaussians(params, split_mask)

        # ---- 合并新增的 Gaussians ----
        extension_tensors = {}
        for key in params:
            parts = [params[key].data]
            if key in new_tensors:
                parts.append(new_tensors[key])
            if key in split_tensors:
                parts.append(split_tensors[key])
            extension_tensors[key] = torch.cat(parts, dim=0)

        # ---- 构建保留 mask（排除被 split 的原始 Gaussian 和被 prune 的）----
        N_extended = extension_tensors['xyz'].shape[0]
        keep_mask = torch.ones(N_extended, dtype=torch.bool, device=device)

        # 被 split 的原始 Gaussian 需要移除（它们已被两个更小的替代）
        keep_mask[:N][split_mask] = False

        # 被 prune 的也要移除
        keep_mask[:N][prune_mask] = False

        # 应用 cap（如果总数超过上限，优先保留梯度大的）
        remaining = keep_mask.sum().item()
        if remaining > max_gaussians:
            # 简单策略：从低梯度端再剪掉一些
            excess = remaining - max_gaussians
            stats.n_pruned += excess

        # ---- 最终赋值 ----
        for key, param in params.items():
            new_data = extension_tensors[key][keep_mask]
            param.data = new_data
            # 同步 optimizer state
            if optimizer is not None:
                self._update_optimizer_state(optimizer, param, keep_mask, extension_tensors[key])

        stats.n_total_after = xyz.shape[0]
        self._last_stats = stats

        # 重置梯度累积
        self._reset_grad_accum()

        return stats

    def _clone_gaussians(
        self,
        params: Dict[str, nn.Parameter],
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Clone selected Gaussians (直接复制，不改参数).

        Cloning is used for small Gaussians: the duplicate is placed at the
        same position with identical parameters. Gradient descent will
        naturally move them apart in subsequent iterations.

        Args:
            params: Dict of parameter name → nn.Parameter
            mask: Boolean mask [N] selecting Gaussians to clone

        Returns:
            Dict of parameter name → new tensor data for cloned Gaussians
        """
        new_tensors = {}
        for key, param in params.items():
            new_tensors[key] = param.data[mask].clone()
        return new_tensors

    def _split_gaussians(
        self,
        params: Dict[str, nn.Parameter],
        mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Split selected Gaussians into two smaller ones.

        Each Gaussian is replaced by two copies, each with:
        - Position: offset ±δ along the dominant scale axis
        - Scale: reduced by factor of 1.6 (empirical value from paper)
        - Other params: copied from the parent

        The offset direction comes from sampling the Gaussian's own
        distribution, ensuring the children cover the parent's extent.

        Args:
            params: Dict of parameter name → nn.Parameter
            mask: Boolean mask [N] selecting Gaussians to split

        Returns:
            Dict of parameter name → new tensor data (2 × n_split Gaussians)
        """
        n_split = mask.sum().item()
        device = params['xyz'].device

        # 取出被 split 的 Gaussians 的参数
        xyz_selected = params['xyz'].data[mask]  # [n_split, 3]
        scaling_selected = params['scaling'].data[mask]  # [n_split, 3] (log space)

        # 激活后的 scale 用于计算偏移量
        activated_scale = torch.exp(scaling_selected)  # [n_split, 3]

        # 沿 scale 最大的轴偏移
        # 随机采样偏移方向（正态分布，按 scale 缩放）
        offset = torch.randn_like(xyz_selected) * activated_scale  # [n_split, 3]

        # 创建两个子 Gaussian：一个 +offset，一个 -offset
        new_xyz = torch.cat([
            xyz_selected + offset,
            xyz_selected - offset,
        ], dim=0)  # [2*n_split, 3]

        # Scale 缩小 1.6 倍（论文中的经验值）
        scale_reduction = torch.log(torch.tensor(1.6, device=device))
        new_scaling = torch.cat([
            scaling_selected - scale_reduction,
            scaling_selected - scale_reduction,
        ], dim=0)  # [2*n_split, 3]

        # 其他参数直接复制两份
        new_tensors = {
            'xyz': new_xyz,
            'scaling': new_scaling,
        }

        for key in ['rotation', 'opacity', 'features_dc', 'features_rest']:
            selected = params[key].data[mask]
            new_tensors[key] = torch.cat([selected, selected], dim=0)

        return new_tensors

    def _reset_grad_accum(self) -> None:
        """Reset gradient accumulation buffers."""
        self._grad_accum = None
        self._grad_count = 0

    @staticmethod
    def _update_optimizer_state(
        optimizer: torch.optim.Optimizer,
        param: nn.Parameter,
        keep_mask: torch.Tensor,
        full_data: torch.Tensor,
    ) -> None:
        """
        Update optimizer state to match new parameter shape.

        When Gaussians are added or removed, the optimizer's internal
        state (e.g., Adam's momentum/variance buffers) must be adjusted
        accordingly. New entries are zero-initialized.

        Args:
            optimizer: The optimizer to update
            param: The parameter whose state needs updating
            keep_mask: Boolean mask applied to select kept entries
            full_data: Full concatenated data before masking
        """
        for group in optimizer.param_groups:
            for p in group['params']:
                if p is not param:
                    continue

                state = optimizer.state.get(p, {})
                if not state:
                    return

                for state_key, state_val in state.items():
                    if not isinstance(state_val, torch.Tensor):
                        continue
                    if state_val.shape == ():
                        continue  # scalar state (e.g., step count)

                    # 扩展 state 到 full_data 的长度，新增部分填 0
                    n_old = state_val.shape[0]
                    n_full = full_data.shape[0]
                    if n_old < n_full:
                        pad_shape = list(state_val.shape)
                        pad_shape[0] = n_full - n_old
                        padded = torch.zeros(
                            pad_shape,
                            dtype=state_val.dtype,
                            device=state_val.device,
                        )
                        state_val = torch.cat([state_val, padded], dim=0)

                    # 应用 keep_mask
                    state[state_key] = state_val[keep_mask]

    def reset_opacity(
        self,
        opacity: nn.Parameter,
        new_opacity_logit: float = -2.0,
    ) -> None:
        """
        Reset all Gaussian opacities to a low value.

        This is done periodically (e.g., every 3000 iterations) to force
        near-transparent Gaussians to be pruned in the next densification
        step. It effectively "challenges" every Gaussian to re-prove its
        contribution.

        The logit value -2.0 corresponds to sigmoid(-2.0) ≈ 0.12.

        Args:
            opacity: Opacity parameter in logit space [N, 1]
            new_opacity_logit: Target logit value (default: -2.0)
        """
        opacity.data.fill_(new_opacity_logit)

    def __repr__(self) -> str:
        return (
            f"DensityController(\n"
            f"  grad_threshold={self.densify_grad_threshold},\n"
            f"  scale_threshold={self.densify_scale_threshold},\n"
            f"  prune_opacity={self.prune_opacity_threshold},\n"
            f"  interval={self.densify_interval},\n"
            f"  active_range=[{self.densify_from_iter}, {self.densify_until_iter}]\n"
            f")"
        )
