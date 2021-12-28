import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import PolychromaticField,ApertureFromImage, cf, mm, cm

F = PolychromaticField(
    spectrum=1.5 * cf.illuminant_d65,
    extent_x=25.0 * mm,
    extent_y=25.0 * mm,
    Nx=1500,
    Ny=1500,
    spectrum_size = 200, spectrum_divisions = 40  # increase these values to improve color resolution
)

F.add(ApertureFromImage("./apertures/rings.jpg", image_size = (12.0 * mm,12.0 * mm), simulation = F))


F.propagate(z=150*cm)
rgb = F.get_colors()
F.plot_colors(rgb, xlim=[-8*mm, 8*mm], ylim=[-8*mm, 8*mm])
