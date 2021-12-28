import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, nm, mm, cm, CircularAperture, Lens

F = MonochromaticField(
    wavelength = 543 * nm, extent_x=13. * mm, extent_y=13. * mm, Nx=2000, Ny=2000, intensity =0.01
)

F.add(CircularAperture(radius = 0.7*mm))

F.add(Lens(f = 100*cm)) # Just remove this command to see the pattern without lens
F.propagate(100*cm)

rgb = F.get_colors()
F.plot_colors(rgb, xlim=[-3*mm,3*mm], ylim=[-3*mm,3*mm])
