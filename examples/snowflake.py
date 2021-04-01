import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticField, cf, mm, cm

F = PolychromaticField(
    spectrum=6 * cf.illuminant_d65, extent_x=12 * mm, extent_y=12 * mm, Nx=800, Ny=800
)

F.add_aperture_from_image(
    "./apertures/snowflake.jpg", pad=(7.5 * mm, 7.5 * mm), Nx=1500, Ny=1500
)
rgb = F.compute_colors_at(z=100*cm)
F.plot(rgb, xlim=[-8, 8], ylim=[-8, 8])
