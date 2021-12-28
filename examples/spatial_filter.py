import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, GaussianBeam, Lens, CircularAperture, SpatialNoise, nm, mm, cm

F = MonochromaticField(
    wavelength=632.8 * nm, extent_x=25. * mm, extent_y=25. * mm, Nx=2000, Ny=2000,intensity = 0.1
)


F.add(GaussianBeam(w0 = 0.7*mm))
F.add(SpatialNoise(noise_radius = 0.8*mm, f_mean = 1/(0.2*mm), f_spread = 1/(0.3*mm), A = 1, N= 50))

rgb = F.get_colors()
F.plot_colors(rgb, xlim=[-2.5*mm,2.5*mm], ylim=[-2.5*mm,2.5*mm])

F.add(Lens(f = 50*cm))
F.propagate(50*cm)

rgb = F.get_colors()
F.plot_colors(rgb, xlim=[-2.5*mm,2.5*mm], ylim=[-2.5*mm,2.5*mm])


F.add(CircularAperture(0.28*mm))
F.propagate(50*cm)
F.add(Lens(f = 50*cm))
F.propagate(30*cm)


rgb = F.get_colors()
F.plot_colors(rgb, xlim=[-2.5*mm,2.5*mm], ylim=[-2.5*mm,2.5*mm])
