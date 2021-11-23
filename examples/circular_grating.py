import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticField, cf, mm, cm

F = PolychromaticField(
    spectrum=3.5 * cf.illuminant_d65,
    extent_x=26.0 * mm,
    extent_y=26.0 * mm,
    Nx=1200,
    Ny=1200,
    spectrum_size = 200, spectrum_divisions = 40  # increase these values to improve color resolution
)

F.add_aperture_from_image(
    "./apertures/circular_grating.jpg", image_size=(14 * mm, 14 * mm)
)
rgb = F.compute_colors_at(80*cm)
F.plot_colors(rgb, xlim=[-9*mm, 9*mm], ylim=[-9*mm, 9*mm])
