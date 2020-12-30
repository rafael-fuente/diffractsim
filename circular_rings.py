from polychromatic_simulator import *

F = PolychromaticField(spectrum = 1.5*cf.illuminant_d65, extent_x = 12.0*mm , extent_y = 12.0*mm, Nx= 1200, Ny= 1200)

F.add_aperture_from_image("./apertures/circular_rings.jpg", pad = (9*mm,9*mm), Nx= 1500, Ny= 1500)
rgb = F.compute_colors_at(z = 1.5)
F.plot(rgb, xlim = [-8,8], ylim = [-8,8])