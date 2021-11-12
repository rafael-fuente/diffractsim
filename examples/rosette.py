import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticField, cf, mm, cm

F = PolychromaticField(
    spectrum=7 * cf.illuminant_d65, extent_x=25 * mm, extent_y=25 * mm, Nx=1500, Ny=1500,
    spectrum_size = 200, spectrum_divisions = 40  # increase these values to improve color resolution
)

F.add_aperture_from_image(
    "./apertures/rosette.png", image_size=(17.0 * mm, 17.0 * mm)
)

F.add_lens(f = 120*cm)
rgb = F.compute_colors_at(z=120*cm)
F.plot(rgb, xlim=[-8, 8], ylim=[-8, 8])
