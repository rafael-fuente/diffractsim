import diffractsim
diffractsim.set_backend("CUDA") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, ApertureFromImage, Lens, nm, mm, cm, um


zi = 50*cm # distance from the image plane to the lens
z0 = 50*cm # distance from the lens to the current position
M = zi/z0 # magnification factor
radius = 6*mm


# set up simulation
F = MonochromaticField(
    wavelength=488 * nm, extent_x=1.5 * mm, extent_y=1.5 * mm, Nx=2048, Ny=2048,intensity = 0.2
)

F.add(ApertureFromImage("./apertures/QWT.png",  image_size = (1.0 * mm, 1.0 * mm), simulation = F))

F.scale_propagate(z0, scale_factor = 30)
#zi and z0 must satisfy the thin les equation 1/zi + 1/z0 = 1/f 
F.add(Lens(f = zi*z0/(zi+z0), radius = radius))
F.scale_propagate(zi, scale_factor = M/(30))

#image at z = 100*cm
rgb = F.get_colors()
F.plot_colors(rgb, figsize=(5, 5), xlim=[-0.5*mm,0.5*mm], ylim=[-0.5*mm,0.5*mm])
