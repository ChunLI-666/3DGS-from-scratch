"""
Training Visualization Tools

This module provides visualization functions for monitoring
the 3DGS training process.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TrainingStats:
    """Container for training statistics."""
    iteration: int
    loss: float
    l1_loss: float
    ssim_loss: float
    num_gaussians: int
    mean_opacity: float
    mean_scaling: float
    psnr: Optional[float] = None


class TrainingVisualizer:
    """
    Visualizer for monitoring 3DGS training progress.
    """

    def __init__(self):
        self.stats_history: List[TrainingStats] = []
        self.rendered_images: List[Tuple[int, np.ndarray]] = []
        self.gt_image: Optional[np.ndarray] = None

    def log_stats(self, stats: TrainingStats):
        """Log training statistics."""
        self.stats_history.append(stats)

    def log_image(self, iteration: int, image: Union[np.ndarray, torch.Tensor]):
        """Log a rendered image."""
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        self.rendered_images.append((iteration, image))

    def set_gt_image(self, image: Union[np.ndarray, torch.Tensor]):
        """Set ground truth image for comparison."""
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        self.gt_image = image

    def plot_loss_curve(
        self,
        ax: Optional[plt.Axes] = None,
        show_components: bool = True,
    ) -> plt.Axes:
        """
        Plot the training loss curve.

        Args:
            ax: Matplotlib axes
            show_components: Show L1 and SSIM components separately

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        iterations = [s.iteration for s in self.stats_history]
        losses = [s.loss for s in self.stats_history]

        ax.plot(iterations, losses, 'b-', linewidth=2, label='Total Loss')

        if show_components:
            l1_losses = [s.l1_loss for s in self.stats_history]
            ssim_losses = [s.ssim_loss for s in self.stats_history]
            ax.plot(iterations, l1_losses, 'g--', alpha=0.7, label='L1 Loss')
            ax.plot(iterations, ssim_losses, 'r--', alpha=0.7, label='SSIM Loss')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        return ax

    def plot_gaussian_count(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """
        Plot the number of Gaussians over training.

        Args:
            ax: Matplotlib axes

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        iterations = [s.iteration for s in self.stats_history]
        counts = [s.num_gaussians for s in self.stats_history]

        ax.plot(iterations, counts, 'b-', linewidth=2)
        ax.fill_between(iterations, counts, alpha=0.3)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Number of Gaussians')
        ax.set_title('Gaussian Count During Training')
        ax.grid(True, alpha=0.3)

        return ax

    def plot_opacity_distribution(
        self,
        opacities: Union[np.ndarray, torch.Tensor],
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot the distribution of Gaussian opacities.

        Args:
            opacities: Opacity values [N]
            ax: Matplotlib axes

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        if isinstance(opacities, torch.Tensor):
            opacities = opacities.detach().cpu().numpy()

        ax.hist(opacities, bins=50, edgecolor='black', alpha=0.7)
        ax.axvline(opacities.mean(), color='red', linestyle='--', label=f'Mean: {opacities.mean():.3f}')

        ax.set_xlabel('Opacity')
        ax.set_ylabel('Count')
        ax.set_title('Opacity Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_training_dashboard(self, figsize: Tuple[int, int] = (16, 10)) -> plt.Figure:
        """
        Create a comprehensive training dashboard.

        Args:
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 3, figure=fig)

        # Loss curve
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_loss_curve(ax1)

        # Gaussian count
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_gaussian_count(ax2)

        # PSNR (if available)
        ax3 = fig.add_subplot(gs[0, 2])
        psnrs = [s.psnr for s in self.stats_history if s.psnr is not None]
        if psnrs:
            iterations = [s.iteration for s in self.stats_history if s.psnr is not None]
            ax3.plot(iterations, psnrs, 'g-', linewidth=2)
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('PSNR (dB)')
            ax3.set_title('PSNR Over Training')
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'PSNR not available', ha='center', va='center')
            ax3.set_title('PSNR')

        # Rendered images comparison
        if self.rendered_images and self.gt_image is not None:
            ax4 = fig.add_subplot(gs[1, 0])
            ax4.imshow(self.gt_image)
            ax4.set_title('Ground Truth')
            ax4.axis('off')

            ax5 = fig.add_subplot(gs[1, 1])
            _, latest_render = self.rendered_images[-1]
            ax5.imshow(np.clip(latest_render, 0, 1))
            ax5.set_title(f'Rendered (iter {self.rendered_images[-1][0]})')
            ax5.axis('off')

            ax6 = fig.add_subplot(gs[1, 2])
            diff = np.abs(self.gt_image - latest_render)
            ax6.imshow(diff)
            ax6.set_title('Absolute Difference')
            ax6.axis('off')

        plt.tight_layout()
        return fig

    def render_comparison(
        self,
        gt: Union[np.ndarray, torch.Tensor],
        rendered: Union[np.ndarray, torch.Tensor],
        figsize: Tuple[int, int] = (14, 5),
    ) -> plt.Figure:
        """
        Side-by-side comparison of ground truth and rendered image.

        Args:
            gt: Ground truth image
            rendered: Rendered image
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if isinstance(gt, torch.Tensor):
            gt = gt.detach().cpu().numpy()
        if isinstance(rendered, torch.Tensor):
            rendered = rendered.detach().cpu().numpy()

        # Handle channel-first format
        if gt.ndim == 3 and gt.shape[0] == 3:
            gt = gt.transpose(1, 2, 0)
        if rendered.ndim == 3 and rendered.shape[0] == 3:
            rendered = rendered.transpose(1, 2, 0)

        fig, axes = plt.subplots(1, 3, figsize=figsize)

        axes[0].imshow(np.clip(gt, 0, 1))
        axes[0].set_title('Ground Truth')
        axes[0].axis('off')

        axes[1].imshow(np.clip(rendered, 0, 1))
        axes[1].set_title('Rendered')
        axes[1].axis('off')

        diff = np.abs(gt - rendered)
        axes[2].imshow(diff)
        axes[2].set_title('Difference')
        axes[2].axis('off')

        plt.tight_layout()
        return fig


def plot_loss_curve(
    losses: List[float],
    iterations: Optional[List[int]] = None,
    ax: Optional[plt.Axes] = None,
    title: str = 'Training Loss',
    **kwargs
) -> plt.Axes:
    """
    Simple function to plot a loss curve.

    Args:
        losses: Loss values
        iterations: Iteration numbers (optional)
        ax: Matplotlib axes
        title: Plot title
        **kwargs: Additional plot arguments

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    if iterations is None:
        iterations = list(range(len(losses)))

    ax.plot(iterations, losses, **kwargs)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    return ax
