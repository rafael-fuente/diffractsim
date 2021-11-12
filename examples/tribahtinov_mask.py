import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticField, cf, nm, mm, cm

F = PolychromaticField(
    spectrum = 6*cf.illuminant_d65, 
    extent_x=24. * mm, extent_y=24. * mm, 
    Nx=2048, Ny=2048,spectrum_size = 240, spectrum_divisions = 80
)


F.add_aperture_from_image(
    "./apertures/tribahtinov_mask.jpg", image_size=(5 * mm, 5 * mm)
)


F.add_lens(f = 60*cm)

rgb = F.compute_colors_at(z=60*cm)
F.plot(rgb, xlim=[-9, 9], ylim=[-9, 9], figsize = (10,10))
