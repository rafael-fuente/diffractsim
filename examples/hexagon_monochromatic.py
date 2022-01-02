import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, ApertureFromImage, mm, nm, cm

F = MonochromaticField(
    wavelength=632.8 * nm, extent_x=18 * mm, extent_y=18 * mm, Nx=1024, Ny=1024
)

F.add(ApertureFromImage("./apertures/hexagon.jpg", image_size=(5.6 * mm, 5.6 * mm), simulation = F))


F.propagate(80*cm)
rgb = F.get_colors()
F.plot_colors(rgb, xlim=[-7* mm, 7* mm], ylim=[-7* mm, 7* mm])
