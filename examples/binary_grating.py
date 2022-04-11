import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, nm, mm, cm, BinaryGrating, Lens

F = MonochromaticField(
    wavelength = 632.8 * nm, extent_x=3. * mm, extent_y=3. * mm, Nx=2048, Ny=2048, intensity =2.
)


F.add(BinaryGrating(width = 0.5*mm, height = 0.5*mm, period = 0.05 *mm))


# plot the grating
I = F.get_intensity()
F.plot_intensity(I, grid = True, units = mm)

F.add(Lens(f = 30*cm))
F.zoom_propagate(30*cm, x_interval = [-8*mm, 8*mm], y_interval = [-8*mm,8*mm])

# plot the diffraction pattern
I = F.get_intensity()
F.plot_intensity(I, square_root = True, units = mm, grid = True, figsize = (14,5), slice_y_pos = 0*mm)
