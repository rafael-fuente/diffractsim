import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticField, cf, nm, mm, cm

F = PolychromaticField(
    spectrum = 4*cf.illuminant_d65, 
    extent_x=5. * mm, extent_y=5. * mm, 
    Nx=1500, Ny=1500
)


F.add_aperture_from_image(
    "./apertures/bahtinov_mask.jpg", pad=(10 * mm, 10 * mm), Nx=1520, Ny=1520
)


F.add_lens(f = 30*cm)

rgb = F.compute_colors_at(z=30*cm)
F.plot(rgb, xlim=[-6, 6], ylim=[-6, 6])
