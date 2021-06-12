import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticField, cf, nm, mm, cm

F = PolychromaticField(
    spectrum = 6*cf.illuminant_d65, 
    extent_x=5. * mm, extent_y=5. * mm, 
    Nx=700, Ny=700,spectrum_size = 300, spectrum_divisions = 100
)


F.add_aperture_from_image(
    "./apertures/tribahtinov_mask.jpg", pad=(18 * mm, 18 * mm), Nx=2500, Ny=2500
)


F.add_lens(f = 60*cm)

rgb = F.compute_colors_at(z=60*cm)
F.plot(rgb, xlim=[-9, 9], ylim=[-9, 9], figsize = (10,10))
