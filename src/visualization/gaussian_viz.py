"""
Visualization Tools for 3D Gaussian Splatting Tutorials

This module provides visualization functions for understanding
Gaussian distributions and the splatting process.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional, List, Tuple, Union
import warnings


def plot_2d_gaussian(
    mean: Union[np.ndarray, torch.Tensor],
    covariance: Union[np.ndarray, torch.Tensor],
    ax: Optional[plt.Axes] = None,
    n_std: float = 2.0,
    color: str = 'blue',
    alpha: float = 0.3,
    label: Optional[str] = None,
    show_center: bool = True,
    **kwargs
) -> plt.Axes:
    """
    Plot a 2D Gaussian distribution as an ellipse.

    Args:
        mean: Center of the Gaussian [2]
        covariance: Covariance matrix [2, 2]
        ax: Matplotlib axes (created if None)
        n_std: Number of standard deviations for ellipse size
        color: Color of the ellipse
        alpha: Transparency
        label: Label for legend
        show_center: Whether to show center point
        **kwargs: Additional arguments for Ellipse

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Convert to numpy
    if isinstance(mean, torch.Tensor):
        mean = mean.detach().cpu().numpy()
    if isinstance(covariance, torch.Tensor):
        covariance = covariance.detach().cpu().numpy()

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Compute angle of rotation
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))

    # Compute width and height (2 * n_std * sqrt(eigenvalue))
    width = 2 * n_std * np.sqrt(eigenvalues[1])
    height = 2 * n_std * np.sqrt(eigenvalues[0])

    # Create ellipse
    ellipse = Ellipse(
        xy=mean,
        width=width,
        height=height,
        angle=angle,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        label=label,
        **kwargs
    )
    ax.add_patch(ellipse)

    # Show center point
    if show_center:
        ax.plot(mean[0], mean[1], 'o', color=color, markersize=5)

    ax.set_aspect('equal')
    return ax


def plot_2d_gaussian_heatmap(
    mean: Union[np.ndarray, torch.Tensor],
    covariance: Union[np.ndarray, torch.Tensor],
    ax: Optional[plt.Axes] = None,
    extent: float = 3.0,
    resolution: int = 100,
    cmap: str = 'viridis',
    show_contours: bool = True,
    n_contours: int = 5,
) -> plt.Axes:
    """
    Plot a 2D Gaussian as a heatmap with optional contours.

    Args:
        mean: Center [2]
        covariance: Covariance matrix [2, 2]
        ax: Matplotlib axes
        extent: Plot extent in standard deviations
        resolution: Grid resolution
        cmap: Colormap
        show_contours: Whether to show contour lines
        n_contours: Number of contour levels

    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    # Convert to numpy
    if isinstance(mean, torch.Tensor):
        mean = mean.detach().cpu().numpy()
    if isinstance(covariance, torch.Tensor):
        covariance = covariance.detach().cpu().numpy()

    # Compute extent based on eigenvalues
    eigenvalues = np.linalg.eigvalsh(covariance)
    max_std = np.sqrt(max(eigenvalues))

    # Create grid
    x = np.linspace(mean[0] - extent * max_std, mean[0] + extent * max_std, resolution)
    y = np.linspace(mean[1] - extent * max_std, mean[1] + extent * max_std, resolution)
    X, Y = np.meshgrid(x, y)

    # Compute Gaussian values
    pos = np.dstack((X, Y))
    cov_inv = np.linalg.inv(covariance)
    diff = pos - mean
    mahal = np.einsum('...i,ij,...j->...', diff, cov_inv, diff)
    Z = np.exp(-0.5 * mahal)

    # Plot heatmap
    im = ax.imshow(
        Z, extent=[x[0], x[-1], y[0], y[-1]],
        origin='lower', cmap=cmap, aspect='equal'
    )

    # Add contours
    if show_contours:
        ax.contour(X, Y, Z, levels=n_contours, colors='white', alpha=0.5)

    ax.set_xlabel('x')
    ax.set_ylabel('y')

    return ax


def plot_3d_ellipsoid(
    mean: Union[np.ndarray, torch.Tensor],
    covariance: Union[np.ndarray, torch.Tensor],
    ax: Optional[Axes3D] = None,
    n_std: float = 2.0,
    color: str = 'blue',
    alpha: float = 0.3,
    resolution: int = 20,
    wireframe: bool = False,
) -> Axes3D:
    """
    Plot a 3D Gaussian as an ellipsoid.

    Args:
        mean: Center [3]
        covariance: Covariance matrix [3, 3]
        ax: Matplotlib 3D axes
        n_std: Number of standard deviations
        color: Color
        alpha: Transparency
        resolution: Surface resolution
        wireframe: Use wireframe instead of surface

    Returns:
        Matplotlib 3D axes
    """
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    # Convert to numpy
    if isinstance(mean, torch.Tensor):
        mean = mean.detach().cpu().numpy()
    if isinstance(covariance, torch.Tensor):
        covariance = covariance.detach().cpu().numpy()

    # Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Create unit sphere
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Scale by eigenvalues and rotate
    radii = n_std * np.sqrt(eigenvalues)
    sphere_points = np.stack([x, y, z], axis=-1)  # [res, res, 3]

    # Apply transformation
    ellipsoid_points = np.einsum(
        'ij,klj->kli',
        eigenvectors @ np.diag(radii),
        sphere_points
    ) + mean

    # Plot
    if wireframe:
        ax.plot_wireframe(
            ellipsoid_points[:, :, 0],
            ellipsoid_points[:, :, 1],
            ellipsoid_points[:, :, 2],
            color=color, alpha=alpha
        )
    else:
        ax.plot_surface(
            ellipsoid_points[:, :, 0],
            ellipsoid_points[:, :, 1],
            ellipsoid_points[:, :, 2],
            color=color, alpha=alpha, shade=True
        )

    # Plot center
    ax.scatter(*mean, color=color, s=50)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return ax


def plot_multiple_gaussians_3d(
    means: List[Union[np.ndarray, torch.Tensor]],
    covariances: List[Union[np.ndarray, torch.Tensor]],
    colors: Optional[List[str]] = None,
    ax: Optional[Axes3D] = None,
    n_std: float = 2.0,
    alpha: float = 0.3,
) -> Axes3D:
    """
    Plot multiple 3D Gaussians.

    Args:
        means: List of centers [N x [3]]
        covariances: List of covariance matrices [N x [3, 3]]
        colors: List of colors
        ax: Matplotlib 3D axes
        n_std: Number of standard deviations
        alpha: Transparency

    Returns:
        Matplotlib 3D axes
    """
    if ax is None:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

    if colors is None:
        # Generate colors
        cmap = plt.cm.get_cmap('tab10')
        colors = [cmap(i % 10) for i in range(len(means))]

    for mean, cov, color in zip(means, covariances, colors):
        plot_3d_ellipsoid(mean, cov, ax=ax, n_std=n_std, color=color, alpha=alpha)

    return ax


def visualize_projection(
    gaussian_3d_mean: Union[np.ndarray, torch.Tensor],
    gaussian_3d_cov: Union[np.ndarray, torch.Tensor],
    gaussian_2d_mean: Union[np.ndarray, torch.Tensor],
    gaussian_2d_cov: Union[np.ndarray, torch.Tensor],
    camera_position: Optional[Union[np.ndarray, torch.Tensor]] = None,
    figsize: Tuple[int, int] = (14, 6),
) -> plt.Figure:
    """
    Visualize the projection of a 3D Gaussian to 2D.

    Args:
        gaussian_3d_mean: 3D center [3]
        gaussian_3d_cov: 3D covariance [3, 3]
        gaussian_2d_mean: 2D projected center [2]
        gaussian_2d_cov: 2D projected covariance [2, 2]
        camera_position: Camera position for visualization [3]
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)

    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    plot_3d_ellipsoid(gaussian_3d_mean, gaussian_3d_cov, ax=ax1, alpha=0.4)

    if camera_position is not None:
        if isinstance(camera_position, torch.Tensor):
            camera_position = camera_position.detach().cpu().numpy()
        ax1.scatter(*camera_position, color='red', s=100, marker='^', label='Camera')
        ax1.legend()

    ax1.set_title('3D Gaussian')

    # 2D plot
    ax2 = fig.add_subplot(122)
    plot_2d_gaussian(gaussian_2d_mean, gaussian_2d_cov, ax=ax2, alpha=0.4)
    ax2.set_title('2D Projected Gaussian')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


class GaussianVisualizer:
    """
    Interactive Gaussian visualizer for Jupyter notebooks.
    """

    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        self.figsize = figsize

    def plot_2d_gaussian(self, mean, cov, **kwargs):
        """Plot a 2D Gaussian."""
        fig, ax = plt.subplots(figsize=self.figsize)
        plot_2d_gaussian(mean, cov, ax=ax, **kwargs)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.show()
        return fig

    def plot_3d_ellipsoid(self, mean, cov, **kwargs):
        """Plot a 3D Gaussian ellipsoid."""
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        plot_3d_ellipsoid(mean, cov, ax=ax, **kwargs)
        plt.show()
        return fig

    def compare_gaussians(
        self,
        means: List,
        covs: List,
        labels: Optional[List[str]] = None,
        dim: int = 2,
    ):
        """Compare multiple Gaussians side by side."""
        n = len(means)
        colors = plt.cm.tab10(np.linspace(0, 1, n))

        if dim == 2:
            fig, ax = plt.subplots(figsize=self.figsize)
            for i, (mean, cov) in enumerate(zip(means, covs)):
                label = labels[i] if labels else f'Gaussian {i+1}'
                plot_2d_gaussian(mean, cov, ax=ax, color=colors[i], label=label)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
            for i, (mean, cov) in enumerate(zip(means, covs)):
                plot_3d_ellipsoid(mean, cov, ax=ax, color=colors[i])

        plt.show()
        return fig
