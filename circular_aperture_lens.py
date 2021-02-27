from diffractsim import MonochromaticField, nm, mm, cm

F = MonochromaticField(
    wavelength = 543 * nm, extent_x=13. * mm, extent_y=13. * mm, Nx=3000, Ny=3000, power =0.01
)

F.add_circular_slit(0,0, 0.7*mm)

F.add_lens(f = 100*cm)
F.propagate(100*cm)


rgb = F.get_colors()
F.plot(rgb, xlim=[-3,3], ylim=[-3,3])
