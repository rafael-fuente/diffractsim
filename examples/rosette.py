import diffractsim
diffractsim.set_backend("CUDA") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticField, cf, mm, cm

F = PolychromaticField(
    spectrum=7 * cf.illuminant_d65, extent_x=16 * mm, extent_y=16 * mm, Nx=800, Ny=800,
    spectrum_size = 200, spectrum_divisions = 40  # increase these values to improve color resolution
)

F.add_aperture_from_image(
    "./apertures/rosette.png", pad=(7.0 * mm, 7.0 * mm), Nx=1500, Ny=1500
)

F.add_lens(f = 120*cm)
rgb = F.compute_colors_at(z=120*cm)
F.plot(rgb, xlim=[-8, 8], ylim=[-8, 8])
