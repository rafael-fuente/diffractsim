import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticField, cf, mm, cm

F = PolychromaticField(
    spectrum=6 * cf.illuminant_d65, extent_x=20  * mm, extent_y=20  * mm, Nx=2048, Ny=2048
)

F.add_aperture_from_image(
    "./apertures/snowflake.jpg", image_size =(12 * mm, 12 * mm)
)
rgb = F.compute_colors_at(z=100*cm)
F.plot(rgb, xlim=[-8, 8], ylim=[-8, 8])
