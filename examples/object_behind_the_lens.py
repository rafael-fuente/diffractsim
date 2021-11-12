import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, nm, mm, cm

F = MonochromaticField(
    wavelength=488 * nm, extent_x=19. * mm, extent_y=19. * mm, Nx=2000, Ny=2000,intensity = 0.2
)


F.add_gaussian_beam(4*mm)
F.add_lens(f = 60*cm)
F.propagate(30*cm)

F.add_aperture_from_image(
    "./apertures/QWT.png", image_size = (10. * mm, 10. * mm)
)
F.propagate(30*cm)


rgb = F.get_colors()
F.plot(rgb, xlim=[-6.0,6.0], ylim=[-6.0,6.0])
