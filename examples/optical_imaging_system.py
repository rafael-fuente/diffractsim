import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, nm, mm, cm

F = MonochromaticField(
    wavelength=488 * nm, extent_x=18. * mm, extent_y=18. * mm, Nx=2048, Ny=2048,intensity = 0.2
)

F.add_aperture_from_image(
    "./apertures/QWT.png",  image_size =(14 * mm, 14 * mm)
)

#image at z = 0*cm
rgb = F.get_colors()
F.plot_colors(rgb, xlim=[-5.0*mm,5.0*mm], ylim=[-5.0*mm,5.0*mm])


F.propagate(50*cm)

F.add_lens(f = 25*cm, radius = 6*mm)

F.propagate(50*cm)

#image at z = 100*cm
rgb = F.get_colors()
F.plot_colors(rgb, xlim=[-5.0*mm,5.0*mm], ylim=[-5.0*mm,5.0*mm])
