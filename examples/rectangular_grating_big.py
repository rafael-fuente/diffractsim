import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticField, cf, mm, cm

F = PolychromaticField(
    spectrum=1 * cf.illuminant_d65,
    extent_x=10.0 * mm,
    extent_y=10.0 * mm,
    Nx=300,
    Ny=300,
)

F.add_aperture_from_image(
    "./apertures/rectangular_grating_big.jpg", pad=(10 * mm, 10 * mm), Nx=1400, Ny=1400
)
rgb = F.compute_colors_at(30*cm)
F.plot(rgb, xlim=[-8, 8], ylim=[-8, 8])
