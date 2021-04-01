import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, nm, mm, cm

F = MonochromaticField(
    wavelength=632.8 * nm, extent_x=3.6 * mm, extent_y=3.6 * mm, Nx=900, Ny=900
)

F.add_aperture_from_image(
    "./apertures/hexagon.jpg", pad=(9 * mm, 9 * mm), Nx=1500, Ny=1500
)

F.add_lens(f = 80*cm) # Just remove this command to see the pattern without lens
F.propagate(80*cm)


rgb = F.get_colors()
F.plot(rgb, xlim=[-7,7], ylim=[-7,7])
