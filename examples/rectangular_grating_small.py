import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticField, cf, mm, cm

F = PolychromaticField(
    spectrum=10 * cf.illuminant_d65,
    extent_x=5.0 * mm,
    extent_y=5.0 * mm,
    Nx=1000,
    Ny=1000,
)

F.add_aperture_from_image(
    "./apertures/rectangular_grating_small.jpg",
    pad=(15 * mm, 15 * mm),
    Nx=1600,
    Ny=1600,
)
rgb = F.compute_colors_at(100*cm)
F.plot(rgb, xlim=[-7, 7], ylim=[-7, 7])
