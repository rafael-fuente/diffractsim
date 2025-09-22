import numpy as np
from ..util.backend_functions import backend as jnp

from ..util.file_handling import load_graymap_image_as_array, save_phase_mask_as_image
from ..util.image_handling import rescale_img_to_custom_coordinates
from ..monochromatic_simulator import MonochromaticField
from ..propagation_methods import two_steps_fresnel_method

from pathlib import Path
from PIL import Image
from ..util.constants import *
import progressbar
from scipy import integrate
from scipy.integrate import simpson
from scipy import interpolate

from ..util.image_handling import load_image_as_function



"""

MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.

"""



class RotationalPhaseDesign():
    def __init__(self, wavelength, z, extent_input, extent_target,  Nx, Ny):
        """        

        Class for design the phase mask required to reconstruct an rotational symmetric intensity at a distance z
        Method from the paper:
        "Experimental demonstration of a beam shaping non-imaging metasurface," Opt. Express 33, 19119-19129 (2025)
        https://doi.org/10.1364/OE.559542 

        Parameters
        ----------
        input_fun, target_fun: 1D functions with I(r) and E(t) as defined in the paper
        extent_input, extent_target: float values with the interval over which E(t) and I(r) are evaluated
        λ, z: wavelength in the propagating medium and target profile distance
        integration_points: number of samples to use for the numerical integration
        """

        self.z = z
        self.Nx = Nx
        self.Ny = Ny
        self.λ = wavelength
        self.extent_input = extent_input
        self.extent_target = extent_target

        self.dx = 2*extent_input/Nx
        self.dy = 2*extent_input/Ny

        self.x = self.dx*(np.arange(Nx)-Nx//2)
        self.y = self.dy*(np.arange(Ny)-Ny//2)
        self.xx, self.yy = np.meshgrid(self.x, self.y)

    def set_source_intensity(self, source_function):
        self.source_function = source_function

    def set_target_intensity(self, target_function):
        self.target_function = target_function


    def get_phase_fun(self, integration_points=500000):

        r = np.linspace(0,self.extent_input, integration_points) #input r coordinates
        I = self.source_function(r) #incident intensity profile
        t = np.linspace(0,self.extent_target, integration_points) #target t coordinates
        E = self.target_function(t) #target intensity profile
        PI,PE = simpson(r*I, r), simpson(t*E, t) # compute total power
        E = PI/PE*E  # scale the target profile to conserve energy
        int_I = np.array(integrate.cumulative_trapezoid(r*I, r, initial=0) )
        int_E = np.array(integrate.cumulative_trapezoid(t*E, t, initial=0) )
        int_E, idx = np.unique(int_E, return_index=True) # remove repeated values
        t = t[idx]
        
        int_E_inv_fun = interpolate.interp1d(int_E, t, kind="linear",bounds_error=False ,fill_value = (t.min(), t.max()))
        t = int_E_inv_fun(int_I) 
        T_fun = interpolate.interp1d(r, t, kind="cubic",bounds_error=False, fill_value = (t.min(), t.max()))
        t =  T_fun(r)
        
        dΦ_dr = (t - r) / np.sqrt(self.z**2 +  (t - r)**2 ) 
        Φ = (2*np.pi / self.λ)  * integrate.cumulative_trapezoid(dΦ_dr, r, initial=0) 
        Φ_fun = interpolate.interp1d(r, Φ - Φ[0], kind="cubic",bounds_error=False , fill_value = Φ.max())


        self.Φ_fun = Φ_fun
        self.target_scale = PI/ PE
    def save_design_phase_as_image(self, name, phase_mask_format = 'hsv'):

        
        Φ= self.Φ_fun(np.sqrt(self.xx**2 + self.yy**2))
        save_phase_mask_as_image(name, (Φ- Φ.min()) % (2*np.pi)   -  np.pi, phase_mask_format = phase_mask_format)
        
    def save_design_phase_as_file(self, name):

        np.save(name, self.Φ_fun(np.sqrt(self.xx**2 + self.yy**2)))

