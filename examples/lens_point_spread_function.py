import diffractsim
from diffractsim import MonochromaticField,Lens, nm, um
# Change the string to "CUDA" to use GPU acceleration
diffractsim.set_backend("CPU")

# PARAMETERS
wlen = 1000 * nm
Nx = Ny = 2048
dx = dy = 400 * nm
focal_distance = 10 * Nx * dx  # 10 times lens diameter

# SETUP AND PROPAGATION
F = MonochromaticField(
    wavelength=wlen,
    extent_x=Nx * dx,
    extent_y=Nx * dx,
    Nx=Nx,
    Ny=Ny)

F.add(Lens(f=focal_distance, radius = Nx * dx/2))
F.propagate(focal_distance)

I = F.get_intensity()
F.plot_intensity(I, square_root = True, xlim = [-30*um, 30*um] , ylim = [-30*um, 30*um],  units = um, grid = True, figsize = (14,5), slice_y_pos = 0*um)
