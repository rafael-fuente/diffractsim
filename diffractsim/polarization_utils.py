"""
MPL 2.0 Clause License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""

import matplotlib.pyplot as plt
import time
import progressbar
from .util.constants import *
from .propagation_methods import angular_spectrum_method, two_steps_fresnel_method, bluestein_method, apply_transfer_function

import numpy as np
from .util.backend_functions import backend as bd
from .util.bluestein_FFT import bluestein_fft2
from .vectorial_field import VectorialField  # NEW IMPORT

class MonochromaticField:
    def __init__(self,  wavelength, extent_x, extent_y, Nx, Ny, intensity = 0.1 * W / (m**2)):
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
        self.Ny = Ny
        self.E = bd.ones((self.Ny, self.Nx)) * bd.sqrt(intensity)
        self.λ = wavelength
        self.z = 0
        self.cs = cf.ColourSystem(clip_method = 0)

    def add(self, optical_element):

        self.E = optical_element.get_E(self.E, self.xx, self.yy, self.λ)


    def propagate(self, z, scale_factor = 1):
        """
        Compute the field in 

# Placeholder for polarization calculations
def calculate_polarization(field):
    # Example: Calculate the Jones vector
    E_x = field.E.real
    E_y = field.E.imag
    Jx = E_x / bd.sqrt(E_x**2 + E_y**2)
    Jy = E_y / bd.sqrt(E_x**2 + E_y**2)
    return Jx, Jy

# Placeholder for polarization analysis
def analyze_polarization(jones_vector):
    # Example: Calculate the degree of polarization
    P = 2 * bd.abs(bd.cross(jones_vector[0], jones_vector[1])) / (bd.norm(jones_vector[0]) + bd.norm(jones_vector[1]))
    return P

# Placeholder for polarization visualization
def plot_polarization(jones_vector):
    # Example: Plot the Jones vector

... [truncated]
    plt.quiver(jones_vector[0], jones_vector[1])
    plt.xlabel('E_x')
    plt.ylabel('E_y')
    plt.title('Jones Vector')
    plt.show()

# Placeholder for polarization transformation
def transform_polarization(field, transformation_matrix):
    # Example: Transform the Jones vector using a matrix
    Jx, Jy = calculate_polarization(field)
    Jx_transformed = bd.dot(transformation_matrix[0], [Jx, Jy])
    Jy_transformed = bd.dot(transformation_matrix[1], [Jx, Jy])
    return Jx_transformed, Jy_transformed
```

# Placeholder for polarization calculations
        pass