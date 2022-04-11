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



# propagate the field and scale the viewing extent four times: (new_extent_x = old_extent_x * 4 = 80* mm)
#F.scale_propagate(400*cm, scale_factor = 4)
F.zoom_propagate(400*cm, x_interval = [-40. * mm, 40. * mm], y_interval = [-40*mm, 40*mm])



# plot the double slit diffraction pattern colors
rgb = F.get_colors()
F.plot_colors(rgb) 



# plot the intensity
I = F.get_intensity()
F.plot_intensity(I, square_root = True, units = mm, grid = True, figsize = (14,5), slice_y_pos = 0*mm)
