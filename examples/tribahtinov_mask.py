import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticField, Lens, ApertureFromImage, cf, nm, mm, cm

F = PolychromaticField(
    spectrum = 6*cf.illuminant_d65, 
    extent_x=24. * mm, extent_y=24. * mm, 
    Nx=2048, Ny=2048,spectrum_size = 240, spectrum_divisions = 80
)

F.add(ApertureFromImage("./apertures/tribahtinov_mask.jpg", image_size=(5 * mm, 5 * mm), simulation = F))

F.add(Lens(f = 60*cm))
F.propagate(60*cm)

rgb =F.get_colors()
F.plot_colors(rgb, xlim=[-9*mm, 9*mm], ylim=[-9*mm, 9*mm], figsize = (10,10))
