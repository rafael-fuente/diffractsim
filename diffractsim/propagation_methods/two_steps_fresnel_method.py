import numpy as np
from ..util.backend_functions import backend as bd

"""
MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""

def two_steps_fresnel_method(simulation, E, z, λ, scale_factor):
    """
    Compute the field in distance equal to z with the two step Fresnel propagator, rescaling the field in the new coordinates
    with extent equal to:
    new_extent_x = scale_factor * self.extent_x
    new_extent_y = scale_factor * self.extent_y

    Note that unlike within in the propagate method, Fresnel approximation is used here.
    Reference: VOELZ, D. G. (2011). Computational Fourier optics. Bellingham, Wash, SPIE.

    To arbitrarily choose and zoom in a region of interest, use bluestein method method instead.
    """
    global bd
    from ..util.backend_functions import backend as bd


    L1 = simulation.extent_x
    L2 = simulation.extent_x*scale_factor


    fft_E = bd.fft.fftshift(bd.fft.fft2(E * np.exp(1j * np.pi/(z * λ) * (L1-L2)/L1 * (simulation.xx**2 + simulation.yy**2) )  ))
    fx = bd.fft.fftshift(bd.fft.fftfreq(simulation.Nx, d = simulation.dx))
    fy = bd.fft.fftshift(bd.fft.fftfreq(simulation.Ny, d = simulation.dy))
    fx, fy = bd.meshgrid(fx, fy)

    E = bd.fft.ifft2(bd.fft.ifftshift( bd.exp(- 1j * np.pi * λ * z * L1/L2 * (fx**2 + fy**2))  *  fft_E) )


    simulation.extent_x = simulation.extent_x*scale_factor
    simulation.extent_y = simulation.extent_y*scale_factor

    simulation.dx = simulation.dx*scale_factor
    simulation.dy = simulation.dy*scale_factor

    simulation.x = simulation.x*scale_factor
    simulation.y = simulation.y*scale_factor

    simulation.xx = simulation.xx*scale_factor
    simulation.yy = simulation.yy*scale_factor


    E = L1/L2 * bd.exp(1j * 2*np.pi/λ * z   - 1j * np.pi/(z * λ)* (L1-L2)/L2 * (simulation.xx**2 + simulation.yy**2)) * E

    return E
