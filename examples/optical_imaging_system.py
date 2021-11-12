import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, nm, mm, cm

F = MonochromaticField(
    wavelength=488 * nm, extent_x=18. * mm, extent_y=18. * mm, Nx=2048, Ny=2048,intensity = 0.2
)

F.add_aperture_from_image(
    "./apertures/QWT.png",  image_size =(14 * mm, 14 * mm), Nx=2300, Ny=2300
)

#image at z = 0*cm
rgb = F.get_colors()
F.plot(rgb, xlim=[-5.0,5.0], ylim=[-5.0,5.0])


F.propagate(50*cm)

F.add_lens(f = 25*cm)
F.add_circular_slit( 0, 0, 6*mm) # we model the entrance pupil of the lens as a circular aperture

F.propagate(50*cm)

#image at z = 100*cm
rgb = F.get_colors()
F.plot(rgb, xlim=[-5.0,5.0], ylim=[-5.0,5.0])
