"""
Layer class for TMM calculations.

MPL 2.0 Clause License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""

import numpy as np
from ..util.backend_functions import backend as bd


class Layer:
    """
    Represents a single layer in a multilayer stack.
    
    Parameters
    ----------
    n : float, callable, or Material
        Refractive index. Can be:
        - float: constant index
        - callable: function n(λ) returning complex index
        - Material: Material object with n(λ) and k(λ) methods
    k : float, callable, optional
        Extinction coefficient (for lossy media). Default is 0.
        Can be float or callable k(λ).
    d : float, optional
        Layer thickness in meters. If None, layer is semi-infinite (substrate).
    name : str, optional
        Layer name for identification.
    """
    
    def __init__(self, n, k=0.0, d=None, name=""):
        self._n = n
        self._k = k
        self.d = d
        self.name = name
    
    def get_index(self, wavelength):
        """
        Get complex refractive index at given wavelength.
        
        Parameters
        ----------
        wavelength : float or array
            Wavelength(s) in meters
            
        Returns
        -------
        complex or array
            Complex refractive index n + ik
        """
        # Handle n
        if callable(self._n):
            n_val = self._n(wavelength)
        elif hasattr(self._n, 'get_index'):
            # Material object
            n_val = self._n.get_index(wavelength)
        else:
            n_val = self._n
        
        # Handle k
        if callable(self._k):
            k_val = self._k(wavelength)
        elif hasattr(self._k, 'get_extinction'):
            k_val = self._k.get_extinction(wavelength)
        else:
            k_val = self._k
        
        return n_val + 1j * k_val
    
    def is_semi_infinite(self):
        """Check if layer is semi-infinite (substrate)."""
        return self.d is None

