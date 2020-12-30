from monochromatic_simulator import *

F = MonochromaticField(wavelength =632.8 *nm, extent_x = 5.6*mm , extent_y = 5.6*mm, Nx= 500, Ny= 500)

F.add_aperture_from_image(Path("./apertures/hexagon.jpg"), pad = (10 *mm, 10*mm),  Nx = 1400, Ny = 1400)

rgb = F.compute_colors_at(z = 0.8)
F.plot(rgb, xlim = [-7,7], ylim = [-7,7])
