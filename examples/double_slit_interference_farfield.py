import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, nm, mm, cm, RectangularSlit

F = MonochromaticField(
    wavelength = 632.8 * nm, extent_x=20. * mm, extent_y=20. * mm, Nx=2048, Ny=2048, intensity =2.
)


D = 1 * mm  #slits separation
F.add(RectangularSlit(width = 0.2*mm, height = 1.5*mm, x0 = -D/2 , y0 = 0)   +   RectangularSlit(width = 0.2*mm, height = 1.5*mm, x0 = D/2, y0 = 0))

α, β, radiant_intensity_percos = F.get_farfield()

F.plot_farfield(α, β, radiant_intensity_percos ,  grid = True,  alpha_lim=[-0.01,0.01], beta_lim=[-0.01,0.01])

F.plot_farfield_spherical_coordinates(α, β, radiant_intensity_percos ,  theta_lim=0.5)