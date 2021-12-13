import diffractsim
diffractsim.set_backend("CUDA") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, mm, nm, cm

F = MonochromaticField(
    wavelength=632.8 * nm, extent_x=18 * mm, extent_y=18 * mm, Nx=2000, Ny=2000
)

F.add_aperture_from_image(
    "./examples/apertures/hexagon.jpg", image_size=(5.6 * mm, 5.6 * mm)
)

#F.propagate(80*cm)

F.scale_propagate(z = 2240*cm, scale_factor = 5.2)

F.plot_intensity(square_root = True)
#rgb = F.get_colors()
#F.plot_colors(rgb)#, xlim=[-7* mm, 7* mm], ylim=[-7* mm, 7* mm])
