"""
Visualization functions for vectorial electromagnetic fields.

MPL 2.0 License

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_stokes_parameters(field, figsize=(16, 4), show=False, **kwargs):
    """
    Plot the Stokes parameters of the vectorial field.

    Parameters
    ----------
    field : VectorialMonochromaticField
        The vectorial field to visualize
    figsize : tuple, optional
        Figure size (default: (16, 4))
    show : bool, optional
        Whether to display the plot (default: False)
    **kwargs
        Additional arguments passed to imshow

    Returns
    -------
    tuple
        (figure, axes) containing the matplotlib figure and axes
    """
    S = field.get_stokes_parameters()

    fig, axes = plt.subplots(1, 4, figsize=figsize)

    # Common imshow parameters
    imshow_kwargs = {
        'extent': [field.x[0], field.x[-1], field.y[0], field.y[-1]],
        'origin': 'lower',
        **kwargs
    }

    # S0 - Total intensity
    im0 = axes[0].imshow(S['S0'], **imshow_kwargs)
    axes[0].set_title(r'$S_0$ (Intensity)', fontsize=12)
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    # S1 - Linear polarization (horizontal-vertical)
    im1 = axes[1].imshow(S['S1'], **imshow_kwargs)
    axes[1].set_title(r'$S_1$ (H-V)', fontsize=12)
    axes[1].set_xlabel('x (m)')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    # S2 - Linear polarization (+45/-45)
    im2 = axes[2].imshow(S['S2'], **imshow_kwargs)
    axes[2].set_title(r'$S_2$ (+45/-45)', fontsize=12)
    axes[2].set_xlabel('x (m)')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    # S3 - Circular polarization (RCP-LCP)
    im3 = axes[3].imshow(S['S3'], **imshow_kwargs)
    axes[3].set_title(r'$S_3$ (R-L)', fontsize=12)
    axes[3].set_xlabel('x (m)')
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes


def plot_polarization_map(field, subsample=16, show=False, figsize=(8, 8)):
    """
    Plot a polarization map showing the polarization state across the field.

    Parameters
    ----------
    field : VectorialMonochromaticField
        The vectorial field to visualize
    subsample : int, optional
        Subsampling factor for the visualization (default: 16)
    show : bool, optional
        Whether to display the plot (default: False)
    figsize : tuple, optional
        Figure size (default: (8, 8))

    Returns
    -------
    tuple
        (figure, axes) containing the matplotlib figure and axes
    """
    from .util.backend_functions import backend as bd
    from .util.backend_functions import backend_name

    # Get field components
    if backend_name == 'cupy':
        Ex = field.Ex.get()
        Ey = field.Ey.get()
    elif backend_name == 'jax':
        Ex = field.Ex.block_until_ready()
        Ey = field.Ey.block_until_ready()
        Ex = np.array(Ex)
        Ey = np.array(Ey)
    else:
        Ex = field.Ex
        Ey = field.Ey

    # Subsample for visualization
    x = field.x[::subsample]
    y = field.y[::subsample]
    Ex_sub = Ex[::subsample, ::subsample]
    Ey_sub = Ey[::subsample, ::subsample]

    # Compute polarization parameters
    amplitude = np.sqrt(np.abs(Ex_sub)**2 + np.abs(Ey_sub)**2)
    phase_diff = np.angle(Ey_sub) - np.angle(Ex_sub)

    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        Ex_norm = Ex_sub / (amplitude + 1e-20)
        Ey_norm = Ey_sub / (amplitude + 1e-20)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot intensity as background
    intensity = np.abs(Ex_sub)**2 + np.abs(Ey_sub)**2
    im = ax.imshow(intensity,
                   extent=[field.x[0], field.x[-1], field.y[0], field.y[-1]],
                   origin='lower',
                   cmap='gray')
    plt.colorbar(im, ax=ax, label='Intensity')

    # Overlay polarization ellipses
    X, Y = np.meshgrid(x, y)

    # Draw polarization ellipses
    ax.quiver(X, Y, np.real(Ex_norm), np.real(Ey_norm),
              color='red', scale=20, width=0.005, headwidth=4)

    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title('Polarization State Map')

    if show:
        plt.show()

    return fig, ax


def plot_field_components(field, figsize=(14, 4), show=False, **kwargs):
    """
    Plot the three components of the vectorial field (|Ex|, |Ey|, |Ez|).

    Parameters
    ----------
    field : VectorialMonochromaticField
        The vectorial field to visualize
    figsize : tuple, optional
        Figure size (default: (14, 4))
    show : bool, optional
        Whether to display the plot (default: False)
    **kwargs
        Additional arguments passed to imshow

    Returns
    -------
    tuple
        (figure, axes) containing the matplotlib figure and axes
    """
    from .util.backend_functions import backend as bd
    from .util.backend_functions import backend_name

    # Get intensity of each component
    I_x = bd.real(bd.conj(field.Ex) * field.Ex)
    I_y = bd.real(bd.conj(field.Ey) * field.Ey)
    I_z = bd.real(bd.conj(field.Ez) * field.Ez)

    if backend_name == 'cupy':
        I_x = I_x.get()
        I_y = I_y.get()
        I_z = I_z.get()
    elif backend_name == 'jax':
        I_x = np.array(I_x.block_until_ready())
        I_y = np.array(I_y.block_until_ready())
        I_z = np.array(I_z.block_until_ready())
    else:
        I_x = np.array(I_x)
        I_y = np.array(I_y)
        I_z = np.array(I_z)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    imshow_kwargs = {
        'extent': [field.x[0], field.x[-1], field.y[0], field.y[-1]],
        'origin': 'lower',
        **kwargs
    }

    im0 = axes[0].imshow(I_x, **imshow_kwargs)
    axes[0].set_title(r'$|E_x|^2$', fontsize=12)
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(I_y, **imshow_kwargs)
    axes[1].set_title(r'$|E_y|^2$', fontsize=12)
    axes[1].set_xlabel('x (m)')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(I_z, **imshow_kwargs)
    axes[2].set_title(r'$|E_z|^2$', fontsize=12)
    axes[2].set_xlabel('x (m)')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()

    if show:
        plt.show()

    return fig, axes
