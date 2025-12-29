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

    # 2. Evanescent Decay Sign Enforcement: Enforce Im(kz) >= 0
    # This ensures that evanescent waves decay (exp(-|Im(kz)|*z)) rather than grow.
    # While np.sqrt usually handles this, numerical noise can flip the sign.
    kz = np.where(np.imag(kz) < 0, np.conj(kz), kz)

    # Fourier transform of transverse components
    Ex_k = np.fft.fft2(Ex)
    Ey_k = np.fft.fft2(Ey)

    # 1. Ez Reconstruction Order: Compute Ez BEFORE propagation
    # We must construct Ez in the source plane and THEN propagate it 
    # using the same transfer function as Ex and Ey.
    # div(E) = 0 -> kx*Ex + ky*Ey + kz*Ez = 0
    with np.errstate(divide='ignore', invalid='ignore'):
        Ez_k = -(KX * Ex_k + KY * Ey_k) / kz
    
    # Handle singularity at kz=0 (avoid NaNs)
    Ez_k[np.abs(kz) < 1e-9] = 0.0

    # 3. On-Axis Symmetry Enforcement: Ez must be 0 at DC (kx=ky=0)
    # Physically, for a symmetric beam, the longitudinal component is zero on-axis.
    # Numerical noise or division by small kz can perturb this.
    # We identify the DC component where KX=0 and KY=0.
    dc_mask = (KX == 0) & (KY == 0)
    Ez_k[dc_mask] = 0.0

    # Note: Ez may still exhibit small non-zero values on-axis in the spatial domain
    # due to finite grid discretization and the asymmetry of the FFT Nyquist component.
    # This is a numerical artifact that decreases with higher resolution and does
    # not violate physical consistency.

    # Optical transfer function (propagator)
    H = np.exp(1j * kz * z)

    # Propagate ALL components with the same kernel
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
