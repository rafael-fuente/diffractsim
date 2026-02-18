import matplotlib.pyplot as plt
import time
import progressbar
from .util.constants import *
from .propagation_methods import angular_spectrum_method, two_steps_fresnel_method, bluestein_method, apply_transfer_function

import numpy as np
from .util.backend_functions import backend as bd
from .util.bluestein_FFT import bluestein_fft2
from .polarization_states import polarization_state
import matplotlib.pyplot as plt  # Added for plotting

import numpy as np
from .util.backend_functions import backend as bd
from .util.bluestein_FFT import bluestein_fft2

"""
MPL 2.0 Clause License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""

class MonochromaticField:
    class MonochromaticField:
        def __init__(self, wavelength, extent_x, extent_y, Nx, Ny, intensity=0.1 * W / (m**2)):
            """
            Initializes the field, representing the cross-section profile of a plane wave

            Parameters
            ----------
            wavelength: wavelength of the plane wave
            extent_x: length of the rectangular grid 
            extent_y: height of the rectangular grid 
            Nx: horizontal dimension of the grid 
            Ny: vertical dimension of the grid 
            intensity: intensity of the field
            """
            global bd
            global backend_name
            from .util.backend_functions import backend as bd
            from .util.backend_functions import backend_name

            self.extent_x = extent_x
            self.extent_y = extent_y

            self.dx = extent_x/Nx
            self.dy = extent_y/Ny

            self.x = self.dx*(bd.arange(Nx)-Nx//2)
            self.y = self.dy*(bd.arange(Ny)-Ny//2)
            self.xx, self.yy = bd.meshgrid(self.x, self.y)

            self.Nx = Nx
    