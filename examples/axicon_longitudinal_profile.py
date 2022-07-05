import diffractsim
diffractsim.set_backend("CUDA") #Change the string to "CUDA" to use GPU acceleration
# Note: this example is highly recommendeded to run with CUDA

from diffractsim import MonochromaticField, nm, mm, cm, Axicon, bd


F = MonochromaticField(
    wavelength = 578 * nm, extent_x=20. * mm, extent_y=20. * mm, Nx=2048, Ny=2048, intensity =0.005
)

"""
The axicon that creates a beam with an approximate Bessel function profile.
"""

F.add(Axicon(period = 0.06*mm, radius = 5*mm))


end_distance = 90 *cm
steps = 80 
longitudinal_profile_rgb, longitudinal_profile_E, extent = F.get_longitudinal_profile( start_distance = 0*cm , end_distance = end_distance , steps = steps) 

#measure the position of the intensity maximum
longitudinal_profile_E = longitudinal_profile_E[:, 1024]
index = bd.argmax(bd.abs(longitudinal_profile_E))
print('\n Effective focal distance of the axicon: {} cm'.format("%.1f"  % (index * end_distance / steps / cm)))

#plot colors
F.plot_longitudinal_profile_colors(longitudinal_profile_rgb = longitudinal_profile_rgb, extent = extent)