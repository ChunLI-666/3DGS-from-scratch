"""
Interactive Plotting Tools for Jupyter Notebooks

This module provides interactive widgets for exploring
Gaussian parameters in real-time.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Callable, Tuple

# Check if ipywidgets is available
try:
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    WIDGETS_AVAILABLE = True
except ImportError:
    WIDGETS_AVAILABLE = False

from .gaussian_viz import plot_2d_gaussian, plot_3d_ellipsoid


def create_interactive_gaussian(dim: int = 2):
    """
    Create an interactive Gaussian visualization with sliders.

    Args:
        dim: 2 for 2D Gaussian, 3 for 3D ellipsoid

    Returns:
        Interactive widget
    """
    if not WIDGETS_AVAILABLE:
        print("ipywidgets not available. Install with: pip install ipywidgets")
        return None

    if dim == 2:
        return _create_interactive_2d_gaussian()
    else:
        return _create_interactive_3d_gaussian()


def _create_interactive_2d_gaussian():
    """Create interactive 2D Gaussian."""
    # Create sliders
    mean_x = widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description='Mean X:')
    mean_y = widgets.FloatSlider(value=0, min=-5, max=5, step=0.1, description='Mean Y:')
    var_x = widgets.FloatSlider(value=1, min=0.1, max=5, step=0.1, description='Var X:')
    var_y = widgets.FloatSlider(value=1, min=0.1, max=5, step=0.1, description='Var Y:')
    correlation = widgets.FloatSlider(value=0, min=-0.9, max=0.9, step=0.1, description='Correlation:')
    n_std = widgets.FloatSlider(value=2, min=1, max=4, step=0.5, description='N Std:')

    output = widgets.Output()

    def update(change=None):
        with output:
            clear_output(wait=True)

            # Build covariance matrix
            cov_xy = correlation.value * np.sqrt(var_x.value * var_y.value)
            cov = np.array([
                [var_x.value, cov_xy],
                [cov_xy, var_y.value]
            ])
            mean = np.array([mean_x.value, mean_y.value])

            fig, axes = plt.subplots(1, 2, figsize=(14, 6))

            # Ellipse view
            plot_2d_gaussian(mean, cov, ax=axes[0], n_std=n_std.value, alpha=0.4)
            axes[0].set_xlim(-8, 8)
            axes[0].set_ylim(-8, 8)
            axes[0].grid(True, alpha=0.3)
            axes[0].set_title('2D Gaussian Ellipse')
            axes[0].set_xlabel('x')
            axes[0].set_ylabel('y')

            # Heatmap view
            from .gaussian_viz import plot_2d_gaussian_heatmap
            plot_2d_gaussian_heatmap(mean, cov, ax=axes[1])
            axes[1].set_title('2D Gaussian Heatmap')

            plt.tight_layout()
            plt.show()

            # Print covariance matrix
            print(f"Covariance Matrix:\n{cov}")

    # Connect callbacks
    for slider in [mean_x, mean_y, var_x, var_y, correlation, n_std]:
        slider.observe(update, names='value')

    # Layout
    controls = widgets.VBox([
        widgets.HBox([mean_x, mean_y]),
        widgets.HBox([var_x, var_y]),
        widgets.HBox([correlation, n_std]),
    ])

    # Initial plot
    update()

    return widgets.VBox([controls, output])


def _create_interactive_3d_gaussian():
    """Create interactive 3D Gaussian ellipsoid."""
    # Position sliders
    pos_x = widgets.FloatSlider(value=0, min=-3, max=3, step=0.1, description='Pos X:')
    pos_y = widgets.FloatSlider(value=0, min=-3, max=3, step=0.1, description='Pos Y:')
    pos_z = widgets.FloatSlider(value=0, min=-3, max=3, step=0.1, description='Pos Z:')

    # Scale sliders
    scale_x = widgets.FloatSlider(value=1, min=0.1, max=3, step=0.1, description='Scale X:')
    scale_y = widgets.FloatSlider(value=1, min=0.1, max=3, step=0.1, description='Scale Y:')
    scale_z = widgets.FloatSlider(value=1, min=0.1, max=3, step=0.1, description='Scale Z:')

    # Rotation sliders (Euler angles)
    rot_x = widgets.FloatSlider(value=0, min=0, max=360, step=5, description='Rot X (deg):')
    rot_y = widgets.FloatSlider(value=0, min=0, max=360, step=5, description='Rot Y (deg):')
    rot_z = widgets.FloatSlider(value=0, min=0, max=360, step=5, description='Rot Z (deg):')

    output = widgets.Output()

    def euler_to_rotation_matrix(rx, ry, rz):
        """Convert Euler angles (degrees) to rotation matrix."""
        rx, ry, rz = np.radians([rx, ry, rz])

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])

        return Rz @ Ry @ Rx

    def update(change=None):
        with output:
            clear_output(wait=True)

            mean = np.array([pos_x.value, pos_y.value, pos_z.value])
            scaling = np.array([scale_x.value, scale_y.value, scale_z.value])
            R = euler_to_rotation_matrix(rot_x.value, rot_y.value, rot_z.value)

            # Build covariance: Σ = R @ S @ S.T @ R.T
            S = np.diag(scaling)
            cov = R @ S @ S.T @ R.T

            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')

            plot_3d_ellipsoid(mean, cov, ax=ax, alpha=0.4)

            # Set axis limits
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_zlim(-5, 5)

            ax.set_title('3D Gaussian Ellipsoid')
            plt.tight_layout()
            plt.show()

            print(f"Mean: {mean}")
            print(f"Scaling: {scaling}")
            print(f"Covariance:\n{cov}")

    # Connect callbacks
    for slider in [pos_x, pos_y, pos_z, scale_x, scale_y, scale_z, rot_x, rot_y, rot_z]:
        slider.observe(update, names='value')

    # Layout
    controls = widgets.VBox([
        widgets.HTML('<b>Position</b>'),
        widgets.HBox([pos_x, pos_y, pos_z]),
        widgets.HTML('<b>Scale</b>'),
        widgets.HBox([scale_x, scale_y, scale_z]),
        widgets.HTML('<b>Rotation (Euler Angles)</b>'),
        widgets.HBox([rot_x, rot_y, rot_z]),
    ])

    update()

    return widgets.VBox([controls, output])


def create_parameter_sliders(
    param_names: list,
    param_ranges: list,
    callback: Callable,
    descriptions: Optional[list] = None,
):
    """
    Create a generic set of parameter sliders.

    Args:
        param_names: List of parameter names
        param_ranges: List of (min, max, default) tuples
        callback: Function to call when parameters change
        descriptions: Optional descriptions for each parameter

    Returns:
        Interactive widget
    """
    if not WIDGETS_AVAILABLE:
        print("ipywidgets not available. Install with: pip install ipywidgets")
        return None

    sliders = {}
    for i, (name, (min_val, max_val, default)) in enumerate(zip(param_names, param_ranges)):
        desc = descriptions[i] if descriptions else name
        sliders[name] = widgets.FloatSlider(
            value=default,
            min=min_val,
            max=max_val,
            step=(max_val - min_val) / 100,
            description=desc,
        )

    output = widgets.Output()

    def update(change=None):
        with output:
            clear_output(wait=True)
            params = {name: slider.value for name, slider in sliders.items()}
            callback(params)

    for slider in sliders.values():
        slider.observe(update, names='value')

    controls = widgets.VBox(list(sliders.values()))
    update()

    return widgets.VBox([controls, output])
