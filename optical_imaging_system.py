from diffractsim import MonochromaticField, nm, mm, cm

F = MonochromaticField(
    wavelength=488 * nm, extent_x=14. * mm, extent_y=14. * mm, Nx=2000, Ny=2000,power = 0.2
)

F.add_aperture_from_image(
    "./apertures/QWT.png",  pad=(2 * mm, 2 * mm), Nx=2300, Ny=2300
)

F.propagate(50*cm)
F.add_lens(f = 25*cm)
F.propagate(50*cm)


rgb = F.get_colors()
F.plot(rgb, xlim=[-5.0,5.0], ylim=[-5.0,5.0])
