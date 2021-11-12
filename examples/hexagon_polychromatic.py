import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticField, cf, mm, cm

F = PolychromaticField(
    spectrum=2 * cf.illuminant_d65, extent_x=18 * mm, extent_y=18 * mm, Nx=1500, Ny=1500
)

F.add_aperture_from_image(
    "./apertures/hexagon.jpg", image_size=(5.6 * mm, 5.6 * mm)
)

rgb = F.compute_colors_at(z=80*cm)
F.plot(rgb, xlim=[-7, 7], ylim=[-7, 7])
