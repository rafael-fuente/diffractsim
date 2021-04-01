import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticField, cf, mm, cm

F = PolychromaticField(
    spectrum=1.5 * cf.illuminant_d65,
    extent_x=15 * mm,
    extent_y=15 * mm,
    Nx=1200,
    Ny=1200,
)

F.add_aperture_from_image(
    "./apertures/text.jpg", pad=(5 * mm, 5 * mm), Nx=1400, Ny=1400
)
rgb = F.compute_colors_at(z=150*cm)
F.plot(rgb, xlim=[-10, 10], ylim=[-10, 10])
