from simulator import PolychromaticField, cf, mm
from pathlib import Path


F = PolychromaticField(
    spectrum=2 * cf.illuminant_d65, extent_x=5.6 * mm, extent_y=5.6 * mm, Nx=500, Ny=500
)

F.add_aperture_from_image(
    Path("./apertures/hexagon.jpg"), pad=(10 * mm, 10 * mm), Nx=1400, Ny=1400
)

rgb = F.compute_colors_at(z=0.8, spectrum_divisions=40, grid_divisions=10)
F.plot(rgb, xlim=[-7, 7], ylim=[-7, 7])
