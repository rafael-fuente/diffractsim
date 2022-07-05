import diffractsim
diffractsim.set_backend("CUDA") #Change the string to "CUDA" to use GPU acceleration
# Note: this example is highly recommendeded to run with CUDA

from diffractsim import MonochromaticField, ApertureFromImage, Lens, nm, mm, cm

F = MonochromaticField(
    wavelength=488 * nm, extent_x=20. * mm, extent_y=20. * mm, Nx=2048, Ny=2048,intensity = 0.2
)

F.add(ApertureFromImage("./apertures/QWT.png",  image_size =(15 * mm, 15 * mm), simulation = F))
F.add(Lens(f = 50*cm))


longitudinal_profile_rgb, longitudinal_profile_E, extent = F.get_longitudinal_profile( start_distance = 0*cm , end_distance = 100 *cm , steps = 80) 
#plot colors
F.plot_longitudinal_profile_colors(longitudinal_profile_rgb = longitudinal_profile_rgb, extent = extent)
print(longitudinal_profile_rgb.shape)