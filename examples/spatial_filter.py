import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, nm, mm, cm

F = MonochromaticField(
    wavelength=632.8 * nm, extent_x=25. * mm, extent_y=25. * mm, Nx=2000, Ny=2000,intensity = 0.1
)


F.add_gaussian_beam(0.7*mm)
F.add_spatial_noise(noise_radius = 2.2*mm, f_mean = 1/(0.2*mm), f_size = 1/(0.5*mm), A = 0.2, N= 50)

rgb = F.get_colors()
F.plot(rgb, xlim=[-2.5,2.5], ylim=[-2.5,2.5])

F.add_lens(f = 50*cm)
F.propagate(50*cm)
F.add_circular_slit( 0, 0, 0.28*mm)
F.propagate(50*cm)
F.add_lens(f = 50*cm)
F.propagate(30*cm)



rgb = F.get_colors()
F.plot(rgb, xlim=[-2.5,2.5], ylim=[-2.5,2.5])
