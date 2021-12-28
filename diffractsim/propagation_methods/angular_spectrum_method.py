import numpy as np
from ..util.backend_functions import backend as bd

def angular_spectrum_method(simulation, E, z, λ, scale_factor = 1):
    """
    compute the field in distance equal to z with the angular spectrum method. 
    The ouplut plane coordinates must be the same than the input, so scale_factor = 1. Otherwise you can use two_steps_fresnel_method
    Reference: https://rafael-fuente.github.io/simulating-diffraction-patterns-with-the-angular-spectrum-method-and-python.html
    """
    global bd
    from ..util.backend_functions import backend as bd

    # compute angular spectrum
    fft_c = bd.fft.fft2(E)
    c = bd.fft.fftshift(fft_c)

    kx = 2*bd.pi*bd.fft.fftshift(bd.fft.fftfreq(simulation.Nx, d = simulation.dx))
    ky = 2*bd.pi*bd.fft.fftshift(bd.fft.fftfreq(simulation.Ny, d = simulation.dy))
    kx, ky = bd.meshgrid(kx, ky)

    argument = (2 * bd.pi / λ) ** 2 - kx ** 2 - ky ** 2

    #Calculate the propagating and the evanescent (complex) modes
    tmp = bd.sqrt(bd.abs(argument))
    kz = bd.where(argument >= 0, tmp, 1j*tmp)

    # propagate the angular spectrum a distance z
    E = bd.fft.ifft2(bd.fft.ifftshift(c * bd.exp(1j * kz * z)))

    return E
