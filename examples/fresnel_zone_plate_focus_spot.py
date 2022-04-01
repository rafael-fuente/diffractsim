import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, nm, mm, cm,um, FZP

F = MonochromaticField(
    wavelength = 980 * nm, extent_x= 0.8 * mm, extent_y=0.8 * mm, Nx=2048, Ny=2048, intensity =0.01
)

# add a Fresnel zone plate lens (FZP) with focal length = 1*mm
F.add(FZP(f = 1*mm,λ = 980 * nm, radius = 200*um))


#plot phase shift of the FZP
E = F.get_field()
F.plot_phase(E, grid = True, units = mm)



#propagate to the focal plane and zoom in the focus spot of the lens
F.zoom_propagate(1*mm, [-40* um, 40* um], [-40* um, 40* um])

# plot the phase in the focal plane
E = F.get_field()
F.plot_phase(E, grid = True, units = um)


# plot the intensity in the focal plane
I = F.get_intensity()
F.plot_intensity(I, square_root = True, units = um, grid = True)

