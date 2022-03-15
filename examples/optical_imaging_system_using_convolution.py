import diffractsim
diffractsim.set_backend("CPU")
from diffractsim import MonochromaticField,ApertureFromImage, nm, mm, cm, CircularAperture

F = MonochromaticField(
    wavelength=488 * nm, extent_x=1.5 * mm, extent_y=1.5 * mm, Nx=2048, Ny=2048,intensity = 0.2
)

F.add(ApertureFromImage("./apertures/QWT.png",  image_size = (1.0 * mm, 1.0 * mm), simulation = F))

F.propagate_to_image_plane(pupil = CircularAperture(radius = 6*mm) , zi = 50*cm, z0 = 50*cm, scale_factor = 2)

rgb = F.get_colors()
F.plot_colors(rgb, figsize=(5, 5), xlim=[-0.4*mm,0.4*mm], ylim=[-0.4*mm,0.4*mm])
