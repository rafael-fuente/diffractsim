import diffractsim
from diffractsim import MonochromaticField, nm
print(diffractsim.__file__)
import pylab as pl
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

F.add_lens(f=focal_distance)
F.propagate(focal_distance)

# POST PROCESSING
# Find indices closest to (0,0)
idx_x = pl.argmin(abs(F.x-0))
idx_y = pl.argmin(abs(F.y-0))
# Plot PSF cross-sectionis
pl.plot(F.x * 1e6, F.I[idx_x, :], label='PSF in x', lw=2)
pl.plot(F.y * 1e6, F.I[:, idx_y], label='PSF in y', lw=2)
pl.xlabel('x (um)')
pl.xlim(-30, 30)
pl.legend()
pl.show()

# Image plot of PSF
extent = 1e6 * pl.array([F.x.min(), F.x.max(), F.y.min(), F.y.max()])
pl.imshow(F.I, extent=extent)
pl.xlim(-30, 30)
pl.ylim(-30, 30)
pl.xlabel('x (um)')
pl.ylabel('y (um)')
pl.colorbar()
pl.show()
