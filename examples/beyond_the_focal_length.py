import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, nm, mm, cm

F = MonochromaticField(
    wavelength=488 * nm, extent_x=27. * mm, extent_y=27. * mm, Nx=2000, Ny=2000,intensity = 0.2
)

F.add_aperture_from_image(
    "./apertures/QWT.png", image_size = (17* mm,17* mm)
)

F.add_lens(f = 50*cm)
F.propagate(100*cm)


rgb = F.get_colors()
F.plot(rgb, xlim=[-8,8], ylim=[-8,8])
