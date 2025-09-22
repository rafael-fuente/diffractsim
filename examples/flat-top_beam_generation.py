"""
Rotational symmetric beam-shaping example:
Generating a phase profile to shape a Gaussian beam to a flat-top profile

The method is described in: 
"Experimental demonstration of a beam shaping non-imaging metasurface," Opt. Express 33, 19119-19129 (2025)
https://doi.org/10.1364/OE.559542
"""


import jax
jax.config.update("jax_enable_x64", True)
import numpy as np


import diffractsim
from diffractsim import MonochromaticField, nm, mm, cm,um, RectangularSlit
diffractsim.set_backend("JAX") 
from diffractsim.holography.rotational_symmetric_phase_design import RotationalPhaseDesign
λ = 1.0*um  # 
z = 10*cm # distacne
Nx, Ny = 2048,2048 #resolution
extent_x = extent_y = W = 12*mm # extent of the design
extent_input, extent_target = extent_x/2, extent_x/2



#Specify target and source intensity distributions

def target_intensity(t):
    # Target function. Flat-Top beam with beam waist = w_E

    w_E = 1500*um
    n = 20
    E = np.exp(-(t/w_E)**n)
    return E**2

def source_intensity(r):
    # Source function. Gaussian beam with beam waist = w_I

    w_I = 2000*um 
    E = np.exp(-(r/w_I)**2)
    return E**2


# create the phase design
RPD = RotationalPhaseDesign(λ, z, extent_input, extent_target,  Nx, Ny)
RPD.set_source_intensity(source_intensity)
RPD.set_target_intensity(target_intensity)
RPD.get_phase_fun()
RPD.save_design_phase_as_image('flat-top_hologram.png')
RPD.save_design_phase_as_file('flat-top_hologram.npy')




#################################
# Simulate the generated design #
#################################

from diffractsim import SLM, load_image_as_function, load_file_as_function, load_phase_as_function

F = MonochromaticField(
    wavelength=1.0*um, extent_x=extent_x, extent_y=extent_y, Nx=Ny, Ny=Nx, intensity = 0.001
)

# set source intensity as gaussian beam
F.E = np.sqrt(source_intensity(np.sqrt(F.xx**2 + F.yy**2)))

I = F.get_intensity()
F.plot_intensity(I, square_root = True, units = mm, grid = True, figsize = (14,5), slice_y_pos = 0*mm)



# add the SLM represeting the computed phase function
F.add(SLM(
      phase_mask_function = load_phase_as_function("flat-top_hologram.npy", 
      x_interval = (-extent_x/2, +extent_x/2),   y_interval = (-extent_y/2, +extent_y/2)), 
      size_x =extent_x, size_y =extent_y, 
      simulation = F)
)

# propagate field 10*cm
F.propagate(z)


I = F.get_intensity()
F.plot_intensity(I, units = mm, grid = True, figsize = (14,5), slice_y_pos = 0*mm)





#visualize the longitudinal profile
F = MonochromaticField(wavelength=λ, extent_x=extent_x, extent_y=extent_y, Nx=Nx, Ny=Ny, intensity = 0.001)
F.rr = np.sqrt(F.xx**2 + F.yy**2)
F.E = np.sqrt(source_intensity(F.rr))*np.exp(1j*(RPD.Φ_fun(F.rr)))

longitudinal_profile_rgb, longitudinal_profile_E, extent = F.get_longitudinal_profile( start_distance = 0*cm , end_distance = z , steps = 80) 
F.plot_longitudinal_profile_intensity(longitudinal_profile_E, extent)