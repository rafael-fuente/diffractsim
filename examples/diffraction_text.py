import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticField, cf, mm, cm

F = PolychromaticField(
    spectrum=1.5 * cf.illuminant_d65,
    extent_x=20 * mm,
    extent_y=20 * mm,
    Nx=1600,
    Ny=1600,
)

F.add_aperture_from_image(
    "./apertures/diffraction_text.jpg", image_size=(15 * mm, 15 * mm)
)
rgb = F.compute_colors_at(z=150*cm)
F.plot(rgb, xlim=[-10, 10], ylim=[-10, 10])
