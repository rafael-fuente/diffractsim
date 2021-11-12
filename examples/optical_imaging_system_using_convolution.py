import diffractsim
diffractsim.set_backend("CPU")

def propagate_to_image_plane(F, radius, zi, z0):
    from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
    from scipy.interpolate import interp2d
    import numpy as np
    """
    zi: distance from the image plane to the lens
    z0: distance from the lens the current position
    zi and z0 should satisfy the equation 1/zi + 1/z0 = 1/f 
    where f is the focal distance of the lens
    radius: radius of the lens pupil
    """
    F.z += zi + z0
    
    #magnification factor
    M = zi/z0
    fun = interp2d(
                np.linspace(-F.extent_x / 2, F.extent_x / 2, F.E.shape[1]),
                np.linspace(-F.extent_y / 2, F.extent_y / 2, F.E.shape[0]),
                F.E,
                kind="cubic",)
    
    F.E = fun(np.linspace(
              -F.extent_x / 2 , F.extent_x / 2 , F.E.shape[1])/M, 
              np.linspace(-F.extent_y / 2 , F.extent_y / 2 ,F.E.shape[0])/M )/M
    F.E = np.flip(F.E)

    fft_c = fft2(F.E)
    c = fftshift(fft_c)

    fx = np.linspace(
        -1/2 * F.Nx // 2 / (F.extent_x / 2),
        1/2 * F.Nx // 2 / (F.extent_x / 2),
        F.Nx,
    )
    fy = np.linspace(
        -1/2 * F.Ny // 2 / (F.extent_y / 2),
        1/2 * F.Ny // 2 / (F.extent_y / 2),
        F.Ny,
    )
    fx, fy = np.meshgrid(fx, fy)
    fp = np.sqrt(fx**2 + fy**2)

    
    #Definte the OTF function, representing the Fourier transform of the circular pupil function.
    H = np.select(
        [fp * zi* F.λ < radius , True], [1, 0]
    )
    F.E = ifft2(ifftshift(c*H))

    # compute Field Intensity
    F.I = np.real(F.E * np.conjugate(F.E))  


from diffractsim import MonochromaticField, nm, mm, cm

F = MonochromaticField(
    wavelength=488 * nm, extent_x=1.5 * mm, extent_y=1.5 * mm, Nx=2048, Ny=2048,intensity = 0.2
)

F.add_aperture_from_image(
    "./apertures/QWT.png",  image_size = (1.0 * mm, 1.0 * mm)
)

propagate_to_image_plane(F,radius = 6*mm, zi = 50*cm, z0 = 50*cm)


rgb = F.get_colors()
F.plot(rgb, figsize=(5, 5), xlim=[-0.4,0.4], ylim=[-0.4,0.4])
