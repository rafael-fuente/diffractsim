import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, ApertureFromImage, Lens, nm, mm, cm

F = MonochromaticField(
    wavelength=488 * nm, extent_x=18. * mm, extent_y=18. * mm, Nx=2048, Ny=2048,intensity = 0.2
)

F.add(ApertureFromImage("./apertures/QWT.png",  image_size =(14 * mm, 14 * mm), simulation = F))


#image at z = 0*cm
rgb = F.get_colors()
F.plot_colors(rgb, xlim=[-5.0*mm,5.0*mm], ylim=[-5.0*mm,5.0*mm])


F.propagate(400*cm)

F.add(Lens(f = 200*cm, radius = 20*mm))

F.propagate(400*cm)

#image at z = 100*cm
rgb = F.get_colors()
F.plot_colors(rgb, xlim=[-5.0*mm,5.0*mm], ylim=[-5.0*mm,5.0*mm])
