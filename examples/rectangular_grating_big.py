import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticField, ApertureFromImage, cf, mm, cm

F = PolychromaticField(
    spectrum=1 * cf.illuminant_d65,
    extent_x=16.0 * mm,
    extent_y=16.0 * mm,
    Nx=1500,
    Ny=1500,
)

F.add(ApertureFromImage("./apertures/rectangular_grating.jpg",image_size = (10 * mm, 10 * mm), simulation = F))

F.propagate(30*cm)

rgb = F.get_colors()
F.plot_colors(rgb, xlim=[-8*mm, 8*mm], ylim=[-8*mm, 8*mm])
