import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticField, ApertureFromImage, cf, mm, cm

F = PolychromaticField(
    spectrum=10 * cf.illuminant_d65,
    extent_x=30.0 * mm,
    extent_y=30.0 * mm,
    Nx=2048,
    Ny=2048,
)

F.add(ApertureFromImage("./apertures/rectangular_grating.jpg",image_size = (1.2 * mm,1.2 * mm), simulation = F))

F.propagate(100*cm)

rgb = F.get_colors()
F.plot_colors(rgb, xlim=[-8*mm, 8*mm], ylim=[-8*mm, 8*mm])
