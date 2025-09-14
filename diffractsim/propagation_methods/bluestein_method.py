from ..util.backend_functions import backend as bd
from ..util.bluestein_FFT import bluestein_fft2, bluestein_fftfreq

"""
MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""

def bluestein_method(simulation, E, z, λ, x_interval, y_interval):
    """
    Compute the field in distance equal to z with the Bluestein method. 
    Bluestein method is the more versatile one as the dimensions of the output plane can be arbitrarily chosen by using 
    the arguments x_interval and y_interval

    Parameters
    ----------

    x_interval: A length-2 sequence [x1, x2] giving the x outplut plane range
    y_interval: A length-2 sequence [y1, y2] giving the y outplut plane range

    Reference: 
    Hu, Y., Wang, Z., Wang, X. et al. Efficient full-path optical calculation of scalar and vector diffraction using the Bluestein method. 
    Light Sci Appl 9, 119 (2020).
    """

    global bd
    from ..util.backend_functions import backend as bd

    
    E = bluestein_fft2(E * bd.exp(1j * 2*bd.pi/λ /(2*z) *(simulation.xx**2 + simulation.yy**2)), 
                        x_interval[0] / (z*λ), x_interval[1] / (z*λ), 1/simulation.dx, 
                        y_interval[0] / (z*λ), y_interval[1] / (z*λ), 1/simulation.dy)

    dfx = 1/(simulation.Nx*simulation.dx)
    dfy = 1/(simulation.Ny*simulation.dy)

    fx_zfft = bluestein_fftfreq(x_interval[0]/ (z*λ),x_interval[1]/ (z*λ), simulation.Nx)
    fy_zfft = bluestein_fftfreq(y_interval[0]/ (z*λ),y_interval[1]/ (z*λ), simulation.Ny)
    dfx_zfft = fx_zfft[1]-fx_zfft[0]
    dfy_zfft = fy_zfft[1]-fy_zfft[0]

    fxx, fyy = bd.meshgrid(fx_zfft, fy_zfft)
    x_shift = simulation.x[0]
    y_shift = simulation.y[0]
    factor = (simulation.dx*simulation.dy * bd.exp(-1j*2*bd.pi*x_shift*fxx -1j*2*bd.pi*y_shift*fyy))

    x = fx_zfft*(z*λ)
    y = fy_zfft*(z*λ)

    xx, yy = bd.meshgrid(x, y)
    return x,y, E*factor * bd.exp(1j*bd.pi/(λ*z)  * (xx**2 + yy**2)  +   1j*2*bd.pi/λ * z ) / (1j*z*λ)


