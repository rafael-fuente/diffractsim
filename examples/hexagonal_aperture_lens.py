import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, ApertureFromImage,Lens, nm, mm, cm

F = MonochromaticField(
    wavelength=632.8 * nm, extent_x=21.6 * mm, extent_y=21.6 * mm, Nx=900, Ny=900
)

F.add(ApertureFromImage("./apertures/hexagon.jpg", image_size=(3.6* mm, 3.6* mm), simulation = F))

F.add(Lens(f = 80*cm)) # Just remove this command to see the pattern without lens
F.propagate(80*cm)


rgb = F.get_colors()
F.plot_colors(rgb, xlim=[-7*mm,7*mm], ylim=[-7*mm,7*mm])
