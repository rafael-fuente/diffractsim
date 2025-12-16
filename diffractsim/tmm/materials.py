"""
Material definitions for TMM calculations.

MPL 2.0 Clause License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""

import numpy as np
from ..util.backend_functions import backend as bd


class Material:
    """
    Base class for optical materials.
    
    Parameters
    ----------
    n_func : callable, optional
        Function n(λ) returning refractive index
    k_func : callable, optional
        Function k(λ) returning extinction coefficient
    """
    
    def __init__(self, n_func=None, k_func=None):
        self.n_func = n_func if n_func else (lambda λ: 1.0)
        self.k_func = k_func if k_func else (lambda λ: 0.0)
    
    def get_index(self, wavelength):
        """Get complex refractive index."""
        return self.n_func(wavelength) + 1j * self.k_func(wavelength)
    
    def get_extinction(self, wavelength):
        """Get extinction coefficient."""
        return self.k_func(wavelength)


class DrudeMaterial(Material):
    """
    Drude model for metals.
    
    Parameters
    ----------
    ωp : float
        Plasma frequency (rad/s)
    γ : float
        Damping constant (rad/s)
    ε_inf : float, optional
        High-frequency permittivity. Default is 1.0.
    """
    
    def __init__(self, ωp, γ, ε_inf=1.0):
        self.ωp = ωp
        self.γ = γ
        self.ε_inf = ε_inf
        # c = 299792458 m/s
        self.c = 299792458.0
    
    def get_index(self, wavelength):
        """
        Get complex refractive index from Drude model.
        
        ε(ω) = ε_inf - ωp² / (ω² + iγω)
        n = sqrt(ε)
        """
        # Angular frequency
        ω = 2 * np.pi * self.c / wavelength
        
        # Drude permittivity
        ε = self.ε_inf - (self.ωp**2) / (ω**2 + 1j * self.γ * ω)
        
        # Complex refractive index (use numpy sqrt for complex numbers)
        n_complex = np.sqrt(ε)
        
        return n_complex
    
    def get_extinction(self, wavelength):
        """Get extinction coefficient."""
        n_complex = self.get_index(wavelength)
        return np.imag(n_complex)

