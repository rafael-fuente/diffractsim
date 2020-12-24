from polychromatic_simulator import *

# Lx, y Ly en mm        
nm = 1e-9
mm = 1e-3

Lx = 5.60000*mm
Ly = 5.60000*mm
Nx= 600
Ny= 600

F = PolychromaticField(spectrum = 2*cf.illuminant_d65, extent_x = Lx , extent_y = Ly, Nx= Nx, Ny= Ny)

F.add_aperture_from_image("hexagon.jpg", pad = (11.34*mm,7.2*mm), Nx = int(np.round(1.1*800*16/9)), Ny = 800)
#rgb = F.compute_colors_at(z = 2, spectrum_divisions = 1,grid_divisions = 10)

F.image_gen(simulation_name = "hexagon", time = 10, max_distance= 1.5, figsize=(16/9 *10,10), xlim = [-13.689,  13.689], ylim = [-14/2,  14/2], fps = 60)