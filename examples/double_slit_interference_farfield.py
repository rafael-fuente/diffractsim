import diffractsim
diffractsim.set_backend("CPU")

from diffractsim import MonochromaticField, nm, mm, cm, RectangularSlit

# Initialize field with polarization support (pol='x')
F = MonochromaticField(wavelength=632.8*nm, extent_x=15.*mm, extent_y=15.*mm, Nx=1024, Ny=1024, pol='x')

D = 1 * mm  # slits separation
F.add(RectangularSlit(width=0.2*mm, height=1.5*mm, x0=-D/2, y0=0) + RectangularSlit(width=0.2*mm, height=1.5*mm, x0=D/2, y0=0))

# Compute and plot far-field diffraction
alpha, beta, radiant_intensity_percos = F.get_farfield()
F.plot_farfield(alpha, beta, radiant_intensity_percos, grid=True, alpha_lim=[-0.01,0.01], beta_lim=[-0.01,0.01])
F.plot_farfield_spherical_coordinates(alpha, beta, radiant_intensity_percos, theta_lim=0.5)
