import diffractsim

diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration
# Note: this example is highly recommendeded to run with CUDA

from diffractsim import MonochromaticField, Lens, nm, mm, cm, GaussianBeam

F = MonochromaticField(
    wavelength=630 * nm, extent_x=20. * mm, extent_y=20. * mm, Nx=256, Ny=256,intensity = 1
)
F.add(GaussianBeam(1*mm))
F.add(
    Lens(
        diffractsim.bd.Inf,
        # equivalent OPD to a 50cm lens: -1/(2*f) r^2
        aberration=lambda x,y: -1/(2*50*cm) * (x**2+y**2)
    )
)

longitudinal_profile_rgb, longitudinal_profile_E, extent = F.get_longitudinal_profile( start_distance = 0*cm , end_distance = 100 *cm , steps = 80) 
F.plot_longitudinal_profile_colors(longitudinal_profile_rgb = longitudinal_profile_rgb, extent = extent)
print(longitudinal_profile_rgb.shape)

