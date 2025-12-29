"""
Vectorial Angular Spectrum Method.

Implements Maxwell-consistent propagation for vector fields.
"""

import numpy as np

def propagate_vector_angular_spectrum(field, z):
    """
    Propagate a VectorField using vectorial angular spectrum method.

    Parameters
    ----------
    field : VectorField
        Input electromagnetic field
    z : float
        Propagation distance

    Returns
    -------
    VectorField
        Propagated field
    """
    # Retrieve field parameters
    Ex = field.Ex
    Ey = field.Ey
    wavelength = field.wavelength
    x = field.x
    y = field.y

    # Grid parameters
    nx = len(x)
    ny = len(y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    # Compute spatial frequencies (kx, ky)
    # numpy.fft.fftfreq returns frequencies in cycles/unit_distance
    fx = np.fft.fftfreq(nx, d=dx)
    fy = np.fft.fftfreq(ny, d=dy)
    
    # Meshgrid for frequencies matching the field shape (ny, nx)
    # Default meshgrid indexing='xy' -> KX varies along axis 1, KY along axis 0
    KX, KY = np.meshgrid(2 * np.pi * fx, 2 * np.pi * fy)

    # Compute kz component of the wave vector
    # kz = sqrt(k^2 - kx^2 - ky^2)
    k = 2 * np.pi / wavelength
    kz_sq = k**2 - KX**2 - KY**2
    
    # Handle evanescent waves (kz_sq < 0) safely
    # numpy.sqrt on complex array returns 1j*sqrt(abs(val)) for negative real inputs,
    # resulting in decaying exponentials exp(-|kz|z) which is physically correct.
    kz = np.sqrt(kz_sq.astype(complex))

    # Fourier transform of transverse components
    Ex_k = np.fft.fft2(Ex)
    Ey_k = np.fft.fft2(Ey)

    # Compute Ez in Fourier domain to enforce div(E) = 0 -> k . E = 0
    # kz * Ez_k = -(kx * Ex_k + ky * Ey_k)
    # Handle singularity at kz=0 by setting Ez_k=0 (purely transverse propagation)
    with np.errstate(divide='ignore', invalid='ignore'):
        Ez_k = -(KX * Ex_k + KY * Ey_k) / kz
    
    Ez_k[np.abs(kz) < 1e-9] = 0.0

    # Optical transfer function (propagator)
    H = np.exp(1j * kz * z)

    # Propagate components
    Ex_k_propagated = Ex_k * H
    Ey_k_propagated = Ey_k * H
    Ez_k_propagated = Ez_k * H

    # Inverse Fourier transform
    Ex_propagated = np.fft.ifft2(Ex_k_propagated)
    Ey_propagated = np.fft.ifft2(Ey_k_propagated)
    Ez_propagated = np.fft.ifft2(Ez_k_propagated)

    # Return new VectorField with propagated components
    # We use type(field) to preserve subclassing if any
    new_field = type(field)(Ex_propagated, Ey_propagated, wavelength, x, y)
    
    # Attach computed Ez component (not part of standard VectorField init)
    new_field.Ez = Ez_propagated
    
    return new_field
