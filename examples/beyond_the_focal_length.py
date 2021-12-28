import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, ApertureFromImage, Lens, nm, mm, cm

F = MonochromaticField(
    wavelength=488 * nm, extent_x=27. * mm, extent_y=27. * mm, Nx=2000, Ny=2000,intensity = 0.2
)

F.add(ApertureFromImage("./apertures/QWT.png", image_size=(15 * mm, 15 * mm), simulation = F))

F.add(Lens(f = 50*cm))
F.propagate(100*cm)

rgb = F.get_colors()
F.plot_colors(rgb, xlim=[-8*mm,8*mm], ylim=[-8*mm,8*mm])