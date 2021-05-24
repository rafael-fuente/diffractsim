import diffractsim
diffractsim.set_backend("CUDA") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticField, cf, mm, cm

F = PolychromaticField(
    spectrum=2.0 * cf.illuminant_d65,
    extent_x=15.0 * mm,
    extent_y=15.0 * mm,
    Nx=1000,
    Ny=1000,
    spectrum_size = 200, spectrum_divisions = 50  # increase these values to improve color resolution
)

F.add_aperture_from_image(
    "./apertures/hexagon_grating.jpg", pad=(8 * mm, 8 * mm), Nx=1500, Ny=1500
)

F.add_lens(f = 100*cm)
rgb = F.compute_colors_at(100*cm,)
F.plot(rgb, xlim=[-9, 9], ylim=[-9, 9])
