from ..util.backend_functions import backend as bd
from ..util.scaled_FT import scaled_fourier_transform

"""
MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""

def PSF_convolution(simulation, E, λ, PSF, scale_factor = 1):
    """
    Convolve the field with a the given coherent point spread function (PSF) sampled in spatial simulation coordinates.

    Note: the angular spectrum propagation can be exactly reproduced with this method by using as PSF the Rayleigh-Sommerfeld kernel:
    PSF = 1 / (λ) * (1/(k * r) - 1j)  * (exp(j * k * r)* z/r ) where k = 2 * pi / λ  and r = sqrt(x**2 + y**2 + z**2)
    (Also called free space propagation impulse response function)
    """


    global bd
    from ..util.backend_functions import backend as bd

    nn_, mm_ = bd.meshgrid(bd.arange(simulation.Nx)-simulation.Nx//2, bd.arange(simulation.Ny)-simulation.Ny//2)
    factor = ((simulation.dx *simulation.dy)* bd.exp(bd.pi*1j * (nn_ + mm_)))


    E_f = factor*bd.fft.fftshift(bd.fft.fft2(E))
    
    #Definte the ATF function, representing the Fourier transform of the PSF.
    H = factor*bd.fft.fftshift(bd.fft.fft2(PSF))


    if scale_factor == 1:
        return bd.fft.ifft2(bd.fft.ifftshift(E_f*H /factor ))

    else:
        fx = bd.fft.fftshift(bd.fft.fftfreq(simulation.Nx, d = simulation.x[1]-simulation.x[0]))
        fy = bd.fft.fftshift(bd.fft.fftfreq(simulation.Ny, d = simulation.y[1]-simulation.y[0]))
        fxx, fyy = bd.meshgrid(fx, fy)
        extent_fx = (fx[1]-fx[0])*simulation.Nx
        simulation.xx, simulation.yy, E = scaled_fourier_transform(fxx, fyy, E_f*H,  λ = -1, scale_factor = simulation.extent_x/extent_fx * scale_factor, mesh = True)
        simulation.x = simulation.x*scale_factor
        simulation.y = simulation.y*scale_factor
        simulation.dx = simulation.dx*scale_factor
        simulation.dy = simulation.dy*scale_factor
        simulation.extent_x = simulation.extent_x*scale_factor
        simulation.extent_y = simulation.extent_y*scale_factor
        return E



def apply_transfer_function(simulation, E, λ, H, scale_factor = 1):
    """
    Apply amplitude transfer function ATF (H) to the field in the frequency domain sampled in FFT simulation coordinates

    Note: the angular spectrum method amplitude transfer function equivalent is: H = exp(1j * kz * z)
    """

    
    global bd
    from ..util.backend_functions import backend as bd
    import matplotlib.pyplot as plt

    if scale_factor == 1:
        E_f = bd.fft.fftshift(bd.fft.fft2(E))
        return bd.fft.ifft2(bd.fft.ifftshift(E_f*H ))

    else:
        fx = bd.fft.fftshift(bd.fft.fftfreq(simulation.Nx, d = simulation.x[1]-simulation.x[0]))
        fy = bd.fft.fftshift(bd.fft.fftfreq(simulation.Ny, d = simulation.y[1]-simulation.y[0]))
        fxx, fyy = bd.meshgrid(fx, fy)

        nn_, mm_ = bd.meshgrid(bd.arange(simulation.Nx)-simulation.Nx//2, bd.arange(simulation.Ny)-simulation.Ny//2)
        factor = ((simulation.dx *simulation.dy)* bd.exp(bd.pi*1j * (nn_ + mm_)))

        E_f = factor*bd.fft.fftshift(bd.fft.fft2(E))

        extent_fx = (fx[1]-fx[0])*simulation.Nx
        simulation.xx, simulation.yy, E = scaled_fourier_transform(fxx, fyy, E_f*H,  λ = -1, scale_factor = simulation.extent_x/extent_fx * scale_factor, mesh = True)
        simulation.x = simulation.x*scale_factor
        simulation.y = simulation.y*scale_factor
        simulation.dx = simulation.dx*scale_factor
        simulation.dy = simulation.dy*scale_factor
        simulation.extent_x = simulation.extent_x*scale_factor
        simulation.extent_y = simulation.extent_y*scale_factor
        return E
