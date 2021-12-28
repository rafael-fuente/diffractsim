import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticField, ApertureFromImage, Lens, cf, mm, cm

F = PolychromaticField(
    spectrum=2.0 * cf.illuminant_d65,
    extent_x=20.0 * mm,
    extent_y=20.0 * mm,
    Nx=1000,
    Ny=1000,
    spectrum_size = 200, spectrum_divisions = 50  # increase these values to improve color resolution
)

F.add(ApertureFromImage("./apertures/hexagon_grating.jpg", image_size=(15 * mm, 15 * mm), simulation = F))

F.add(Lens(f = 100*cm))
F.propagate(z= 100*cm)

rgb =F.get_colors()
F.plot_colors(rgb, xlim=[-9*mm, 9*mm], ylim=[-9*mm, 9*mm])
