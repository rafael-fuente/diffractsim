import diffractsim
diffractsim.set_backend("CUDA")
import numpy as np
from diffractsim import PolychromaticField, ApertureFromImage, cf, mm, cm, CircularAperture

"""
MPL 2.0 License

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""


def get_colors_at_image_plane(F, radius, M,  zi, z0):
    from diffractsim.util.backend_functions import backend as bd
    import numpy as np
    import time
    import progressbar
    """
    Assuming an incoherent optical system with linear response and assuming the system is only diffraction-limited by
    the exit pupil of the system, compute the field at its image plane

    
    Parameters
    ----------

    radius: exit pupil radius

    zi: distance from the image plane to the exit pupil
    z0: distance from the exit pupil to the current position

    M: magnification factor of the optical system
    (If the optical system is a single lens, magnification = - zi/z0)

    Reference:
    Introduction to Fourier Optics J. Goodman, Frequency Analysis of Optical Imaging Systems
    
    """
    pupil = CircularAperture(radius)
    F.z += zi + z0




    for j in range(len(F.optical_elements)):
        F.E = F.E * F.optical_elements[j].get_transmittance(F.xx, F.yy, 0)

    # if the magnification is negative, the image is inverted
    if M < 0:
        F.E = bd.flip(F.E)
    M_abs = bd.abs(M)

    F.E = F.E/M_abs

    Ip = F.E * bd.conjugate(F.E)
    
    fft_c = bd.fft.fft2(Ip)
    c = bd.fft.fftshift(fft_c)

    fx = bd.fft.fftshift(bd.fft.fftfreq(F.Nx, d = F.x[1]-F.x[0]))/M_abs
    fy = bd.fft.fftshift(bd.fft.fftfreq(F.Ny, d = F.y[1]-F.y[0]))/M_abs
    fx, fy = bd.meshgrid(fx, fy)
    fp = bd.sqrt(fx**2 + fy**2)

    bar = progressbar.ProgressBar()

    # We compute the pattern of each wavelength separately, and associate it to small spectrum interval dλ = (780- 380)/spectrum_divisions . We approximately the final colour
    # by summing the contribution of each small spectrum interval converting its intensity distribution to a RGB space.
    
    sRGB_linear = bd.zeros((3, F.Nx * F.Ny))

    t0 = time.time()

    for i in bar(range(F.spectrum_divisions)):
        #Definte the OTF function, representing the Fourier transform of the circular pupil function.

        fc = radius / (F.λ_list_samples[i]* nm  * zi) # coherent cutoff frequency

        H = pupil.get_optical_transfer_function(fx, fy, zi, F.λ_list_samples[i]* nm )
        #H = bd.where(fp < 2 * fc, 2/bd.pi * (bd.arccos(fp / (2*fc)) - fp / (2*fc) * bd.sqrt(1 - (fp / (2*fc))**2)) , bd.zeros_like(fp))
        Iλ = bd.abs(bd.fft.ifft2(bd.fft.ifftshift(c*H)))

        XYZ = F.cs.spec_partition_to_XYZ(bd.outer(Iλ, F.spec_partitions[i]),i)
        sRGB_linear += F.cs.XYZ_to_sRGB_linear(XYZ)

    if bd != np:
        bd.cuda.Stream.null.synchronize()

    F.xx = M_abs * F.xx
    F.yy = M_abs * F.yy
    F.x = M_abs * F.x
    F.y = M_abs * F.y
    F.dx = M_abs * F.dx
    F.dy = M_abs * F.dy

    rgb = F.cs.sRGB_linear_to_sRGB(sRGB_linear)
    rgb = (rgb.T).reshape((F.Ny, F.Nx, 3))
    print ("Computation Took", time.time() - t0)
    return rgb




from diffractsim import MonochromaticField, nm, mm, cm, um



zi = 40*cm # distance from the image plane to the exit pupil
z0 = 40*cm # distance from the exit pupil to the current simulation plane
pupil_radius = 20*cm # exit pupil radius

#(If the optical system is a single lens, magnification = - zi/z0)
M = -zi/z0
NA = pupil_radius/z0 #numerical aperture

#maximum resolvable distance by Rayleigh criteria for λ ∼ 550 nm:
print('\n Maximum resolvable distance by Rayleigh criteria: {} nm'.format("%.0f"  % (0.61*550/NA)))



#set up the simulation

F = PolychromaticField(
    spectrum=2*M**2* cf.illuminant_d65, extent_x= 14 * um, extent_y= 14 * um, Nx=2048, Ny=2048,
    spectrum_size = 180, spectrum_divisions = 30
)
F.add(ApertureFromImage( "./apertures/horse.png",  image_size=(14 * um, 14 * um), simulation = F))

#propagate the light assuming the source is spatially incoherent
rgb = get_colors_at_image_plane(F,radius = pupil_radius, zi = zi, z0 = z0, M = M)

F.plot_colors(rgb, figsize=(5, 5))
