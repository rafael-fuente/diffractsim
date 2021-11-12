import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticField, cf, mm, cm

F = PolychromaticField(
    spectrum=10 * cf.illuminant_d65,
    extent_x=30.0 * mm,
    extent_y=30.0 * mm,
    Nx=2048,
    Ny=2048,
)

F.add_aperture_from_image(
    "./apertures/rectangular_grating.jpg",image_size = (1.2 * mm,1.2 * mm)
)
rgb = F.compute_colors_at(100*cm)
F.plot(rgb, xlim=[-8, 8], ylim=[-8, 8])
