from . import colour_functions as cf
import matplotlib.pyplot as plt
import time
import progressbar
from .util.constants import *
from .propagation_methods import angular_spectrum_method, two_steps_fresnel_method, bluestein_method
from .monochromatic_simulator import MonochromaticField

import numpy as np
from .util.backend_functions import backend as bd

"""
MPL 2.0 License

Copyright (c) 2025, Implementation by AI Assistant for rafael-fuente/diffractsim
Based on MonochromaticField class by Rafael de la Fuente
"""


class VectorialMonochromaticField(MonochromaticField):
    def __init__(self, wavelength, extent_x, extent_y, Nx, Ny, intensity=0.1 * W / (m**2)):
        """
        Initializes a vectorial electromagnetic field with three components (Ex, Ey, Ez).
        
        Parameters
        ----------
        wavelength: wavelength of the field
        extent_x: length of the rectangular grid
        extent_y: height of the rectangular grid
        Nx: horizontal dimension of the grid
        Ny: vertical dimension of the grid
        intensity: total intensity of the field (distributed across components)
        """
        # Initialize base class
        super().__init__(wavelength, extent_x, extent_y, Nx, Ny, intensity)
        
        # Replace scalar E with vector components
        # Initialize with x-polarized field by default
        self.Ex = self.E.copy()
        self.Ey = bd.zeros((self.Ny, self.Nx), dtype=complex)
        self.Ez = bd.zeros((self.Ny, self.Nx), dtype=complex)
        
        # Remove scalar E to avoid confusion
        del self.E
        
    def set_linear_polarization(self, angle=0):
        """
        Set field to have linear polarization at specified angle.
        
        Parameters
        ----------
        angle: polarization angle in radians (0 = horizontal, π/2 = vertical)
        """
        amplitude = bd.sqrt(bd.mean(bd.abs(self.Ex)**2 + bd.abs(self.Ey)**2))
        self.Ex = amplitude * bd.cos(angle) * bd.ones((self.Ny, self.Nx), dtype=complex)
        self.Ey = amplitude * bd.sin(angle) * bd.ones((self.Ny, self.Nx), dtype=complex)
        self.Ez = bd.zeros((self.Ny, self.Nx), dtype=complex)
        
    def set_circular_polarization(self, handedness='right'):
        """
        Set field to have circular polarization.
        
        Parameters
        ----------
        handedness: 'right' or 'left' circular polarization
        """
        amplitude = bd.sqrt(bd.mean(bd.abs(self.Ex)**2 + bd.abs(self.Ey)**2)) / bd.sqrt(2)
        self.Ex = amplitude * bd.ones((self.Ny, self.Nx), dtype=complex)
        
        if handedness == 'right':
            self.Ey = amplitude * 1j * bd.ones((self.Ny, self.Nx), dtype=complex)
        else:  # left
            self.Ey = -amplitude * 1j * bd.ones((self.Ny, self.Nx), dtype=complex)
            
        self.Ez = bd.zeros((self.Ny, self.Nx), dtype=complex)
        
    def set_elliptical_polarization(self, a, b, angle=0):
        """
        Set field to have elliptical polarization.
        
        Parameters
        ----------
        a: semi-major axis amplitude
        b: semi-minor axis amplitude
        angle: rotation angle of ellipse in radians
        """
        # Normalize amplitudes
        norm = bd.sqrt(a**2 + b**2)
        a_norm = a / norm
        b_norm = b / norm
        
        amplitude = bd.sqrt(bd.mean(bd.abs(self.Ex)**2 + bd.abs(self.Ey)**2))
        
        # Jones vector for elliptical polarization
        self.Ex = amplitude * a_norm * bd.cos(angle) * bd.ones((self.Ny, self.Nx), dtype=complex)
        self.Ey = amplitude * (a_norm * bd.sin(angle) + 1j * b_norm) * bd.ones((self.Ny, self.Nx), dtype=complex)
        self.Ez = bd.zeros((self.Ny, self.Nx), dtype=complex)
        
    def add(self, optical_element):
        """
        Apply optical element to all field components.
        """
        # Apply element to each component
        self.Ex = optical_element.get_E(self.Ex, self.xx, self.yy, self.λ)
        self.Ey = optical_element.get_E(self.Ey, self.xx, self.yy, self.λ)
        self.Ez = optical_element.get_E(self.Ez, self.xx, self.yy, self.λ)
        
    def propagate(self, z, scale_factor=1):
        """
        Propagate all field components using angular spectrum method.
        """
        self.z += z
        self.Ex = angular_spectrum_method(self, self.Ex, z, self.λ, scale_factor=scale_factor)
        self.Ey = angular_spectrum_method(self, self.Ey, z, self.λ, scale_factor=scale_factor)
        self.Ez = angular_spectrum_method(self, self.Ez, z, self.λ, scale_factor=scale_factor)
        
    def scale_propagate(self, z, scale_factor):
        """
        Propagate and rescale all field components.
        """
        self.z += z
        self.x, self.y, self.Ex = two_steps_fresnel_method(self, self.Ex, z, self.λ, scale_factor)
        _, _, self.Ey = two_steps_fresnel_method(self, self.Ey, z, self.λ, scale_factor)
        _, _, self.Ez = two_steps_fresnel_method(self, self.Ez, z, self.λ, scale_factor)
        
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.xx, self.yy = bd.meshgrid(self.x, self.y)
        self.extent_x = self.Nx * self.dx
        self.extent_y = self.Ny * self.dy
        
    def zoom_propagate(self, z, x_interval, y_interval):
        """
        Propagate with zoom using Bluestein method for all components.
        """
        self.z += z
        self.x, self.y, self.Ex = bluestein_method(self, self.Ex, z, self.λ, x_interval, y_interval)
        _, _, self.Ey = bluestein_method(self, self.Ey, z, self.λ, x_interval, y_interval)
        _, _, self.Ez = bluestein_method(self, self.Ez, z, self.λ, x_interval, y_interval)
        
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.xx, self.yy = bd.meshgrid(self.x, self.y)
        self.extent_x = self.Nx * self.dx
        self.extent_y = self.Ny * self.dy
        
    def get_intensity(self):
        """
        Compute total intensity from all field components.
        I = |Ex|² + |Ey|² + |Ez|²
        """
        return bd.real(
            self.Ex * bd.conjugate(self.Ex) +
            self.Ey * bd.conjugate(self.Ey) +
            self.Ez * bd.conjugate(self.Ez)
        )
        
    def get_jones_vector(self, x_idx=None, y_idx=None):
        """
        Get Jones vector at specified point (or center if not specified).
        
        Returns
        -------
        jones_vector: complex array of shape (2,) containing [Ex, Ey]
        """
        if x_idx is None:
            x_idx = self.Nx // 2
        if y_idx is None:
            y_idx = self.Ny // 2
            
        return bd.array([self.Ex[y_idx, x_idx], self.Ey[y_idx, x_idx]])
        
    def get_stokes_parameters(self):
        """
        Compute Stokes parameters (S0, S1, S2, S3) for the field.
        
        S0 = |Ex|² + |Ey|²  (total intensity)
        S1 = |Ex|² - |Ey|²  (linear polarization at 0° vs 90°)
        S2 = 2 Re(Ex Ey*)    (linear polarization at 45° vs 135°)
        S3 = 2 Im(Ex Ey*)    (circular polarization)
        
        Returns
        -------
        S0, S1, S2, S3: 2D arrays of Stokes parameters
        """
        S0 = bd.abs(self.Ex)**2 + bd.abs(self.Ey)**2
        S1 = bd.abs(self.Ex)**2 - bd.abs(self.Ey)**2
        S2 = 2 * bd.real(self.Ex * bd.conj(self.Ey))
        S3 = 2 * bd.imag(self.Ex * bd.conj(self.Ey))
        
        return S0, S1, S2, S3
        
    def get_degree_of_polarization(self):
        """
        Compute degree of polarization (0 = unpolarized, 1 = fully polarized).
        
        DOP = sqrt(S1² + S2² + S3²) / S0
        """
        S0, S1, S2, S3 = self.get_stokes_parameters()
        return bd.sqrt(S1**2 + S2**2 + S3**2) / (S0 + 1e-10)  # avoid division by zero
        
    def __add__(self, Field):
        """
        Interfere VectorialMonochromaticField with another VectorialMonochromaticField.
        """
        if ((self.extent_x == Field.extent_x) and (self.extent_y == Field.extent_y) and 
            (self.Nx == Field.Nx) and (self.Ny == Field.Ny) and (self.λ == Field.λ)):
            
            mixed_field = VectorialMonochromaticField(self.λ, self.extent_x, self.extent_y, 
                                                     self.Nx, self.Ny)
            mixed_field.Ex = self.Ex + Field.Ex
            mixed_field.Ey = self.Ey + Field.Ey
            mixed_field.Ez = self.Ez + Field.Ez
            return mixed_field
        else:
            raise ValueError(
                "The wavelength, dimensions and sampling of the interfering fields must be identical")
