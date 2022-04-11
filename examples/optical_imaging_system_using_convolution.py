import diffractsim
diffractsim.set_backend("CPU")
from diffractsim import MonochromaticField,ApertureFromImage, nm, mm, cm,um, CircularAperture

zi = 50*cm # distance from the image plane to the lens
z0 = 50*cm # distance from the lens to the current position
M = -zi/z0 # magnification factor
radius = 6*mm
NA = radius  / z0  #numerical aperture

#print diffraction limit
print('\n Maximum object resolvable distance by Rayleigh criteria: {} mm'.format("%.3f"  % (0.61*488*nm/NA /mm)))


F = MonochromaticField(
    wavelength=488 * nm, extent_x=1.5 * mm, extent_y=1.5 * mm, Nx=2048, Ny=2048,intensity = 0.2
)

F.add(ApertureFromImage("./apertures/QWT.png",  image_size = (1.0 * mm, 1.0 * mm), simulation = F))

F.propagate_to_image_plane(pupil = CircularAperture(radius = 6*mm) , M = M, zi = zi, z0 = z0)
rgb = F.get_colors()
F.plot_colors(rgb, figsize=(5, 5), xlim=[-0.5*mm,0.5*mm], ylim=[-0.5*mm,0.5*mm])
