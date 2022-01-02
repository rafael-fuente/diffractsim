import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, nm, mm, cm, RectangularSlit

F = MonochromaticField(
    wavelength = 632.8 * nm, extent_x=20. * mm, extent_y=20. * mm, Nx=2048, Ny=2048, intensity =2.
)


D = 1 * mm  #slits separation
F.add(RectangularSlit(width = 0.2*mm, height = 1.5*mm, x0 = -D/2 , y0 = 0)   +   RectangularSlit(width = 0.2*mm, height = 1.5*mm, x0 = D/2, y0 = 0))


# plot the double slit
rgb = F.get_colors()
F.plot_colors(rgb) 

# propagate the field and scale the viewing extent five times: (new_extent_x = old_extent_x * 5 = 100* mm)
F.scale_propagate(400*cm, scale_factor = 5)


# plot the double slit diffraction pattern
rgb = F.get_colors()
F.plot_colors(rgb) 
