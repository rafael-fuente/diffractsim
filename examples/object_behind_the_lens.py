import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, GaussianBeam,Lens,ApertureFromImage, nm, mm, cm

F = MonochromaticField(
    wavelength=488 * nm, extent_x=19. * mm, extent_y=19. * mm, Nx=2000, Ny=2000,intensity = 0.2
)

F.add(GaussianBeam(4*mm))
F.add(Lens(f = 60*cm))
F.propagate(30*cm)

F.add(ApertureFromImage("./apertures/QWT.png", image_size = (10. * mm, 10. * mm), simulation = F))
F.propagate(30*cm)


rgb = F.get_colors()
F.plot_colors(rgb, xlim=[-6.0*mm,6.0*mm], ylim=[-6.0*mm,6.0*mm])
