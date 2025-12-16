"""
Stack class for TMM calculations of multilayer structures.

MPL 2.0 Clause License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""

import numpy as np
from ..util.backend_functions import backend as bd
from .layer import Layer
from .fresnel import fresnel_coefficients


class Stack:
    """
    Multilayer stack for TMM calculations.
    
    Parameters
    ----------
    layers : list of Layer
        List of layers from incident medium to substrate.
        First layer is incident medium, last is substrate (semi-infinite).
    """
    
    def __init__(self, layers):
        if not layers:
            raise ValueError("Stack must have at least one layer (incident medium)")
        
        self.layers = layers
        
        # Validate: last layer should be semi-infinite
        if not self.layers[-1].is_semi_infinite():
            import warnings
            warnings.warn("Last layer should typically be semi-infinite (substrate)")
    
    def solve(self, wavelength, θ_incident=0.0, polarization="s"):
        """
        Solve TMM for given wavelength and angle.
        
        Parameters
        ----------
        wavelength : float
            Wavelength in meters
        θ_incident : float, optional
            Angle of incidence in radians. Default is 0 (normal incidence).
        polarization : str, optional
            "s" for s-polarization, "p" for p-polarization, "mix" for unpolarized.
            Default is "s".
            
        Returns
        -------
        dict
            Dictionary with keys:
            - 'R': Reflectance (power reflection coefficient)
            - 'T': Transmittance (power transmission coefficient)
            - 'A': Absorbance (power absorption coefficient)
            - 'r': Complex reflection coefficient
            - 't': Complex transmission coefficient
        """
        if len(self.layers) < 2:
            # Single interface
            n1 = self.layers[0].get_index(wavelength)
            n2 = self.layers[1].get_index(wavelength) if len(self.layers) > 1 else n1
            
            r, t, _ = fresnel_coefficients(n1, n2, θ_incident, polarization)
            
            # Power coefficients
            if polarization.lower() == "mix":
                # Average of s and p
                r_s, t_s, _ = fresnel_coefficients(n1, n2, θ_incident, "s")
                r_p, t_p, _ = fresnel_coefficients(n1, n2, θ_incident, "p")
                R = 0.5 * (bd.abs(r_s)**2 + bd.abs(r_p)**2)
                T = 0.5 * (bd.abs(t_s)**2 + bd.abs(t_p)**2)
                r = 0.5 * (r_s + r_p)
                t = 0.5 * (t_s + t_p)
            else:
                R = bd.abs(r)**2
                T = bd.abs(t)**2
            
            A = 1.0 - R - T
            
            return {
                'R': R,
                'T': T,
                'A': A,
                'r': r,
                't': t
            }
        
        # Multi-layer stack: use transfer matrix method
        return self._solve_multilayer(wavelength, θ_incident, polarization)
    
    def _solve_multilayer(self, wavelength, θ_incident, polarization):
        """
        Solve multilayer stack using transfer matrix method.
        
        Uses standard TMM formulation where transfer matrix relates
        forward and backward propagating waves.
        """
        # Get indices for all layers
        n_list = [layer.get_index(wavelength) for layer in self.layers]
        
        # Initialize angles using Snell's law
        θ_list = [θ_incident]
        for i in range(len(n_list) - 1):
            sin_θ = (n_list[i] / n_list[i+1]) * bd.sin(θ_list[i])
            θ_list.append(bd.arcsin(sin_θ))
        
        # Build transfer matrix from incident to substrate
        # Start with identity matrix
        M = bd.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex)
        
        # Process each interface and layer
        for i in range(len(self.layers) - 1):
            n_i = n_list[i]
            n_j = n_list[i+1]
            θ_i = θ_list[i]
            θ_j = θ_list[i+1]
            
            # Fresnel coefficients at interface i -> i+1
            r_ij, t_ij, _ = fresnel_coefficients(n_i, n_j, θ_i, polarization)
            
            # Interface transfer matrix
            # Standard form: relates fields on left and right of interface
            M_interface = (1.0 / t_ij) * bd.array([
                [1.0, r_ij],
                [r_ij, 1.0]
            ])
            
            M = M @ M_interface
            
            # Propagation matrix through layer j (if finite thickness)
            # Only propagate if this is not the last layer
            if i < len(self.layers) - 2 and self.layers[i+1].d is not None:
                d = self.layers[i+1].d
                k_z = 2 * np.pi * n_j * bd.cos(θ_j) / wavelength
                phi = k_z * d
                
                # Propagation matrix: phase advance for forward, phase retard for backward
                M_prop = bd.array([
                    [bd.exp(-1j * phi), 0.0],
                    [0.0, bd.exp(1j * phi)]
                ])
                
                M = M @ M_prop
        
        # Extract reflection and transmission coefficients
        # Transfer matrix M relates: [E_forward_inc, E_backward_inc]^T = M [E_forward_sub, 0]^T
        # Since E_backward_sub = 0 (no reflection from substrate)
        # r = M[1,0] / M[0,0], t = 1 / M[0,0]
        r = M[1, 0] / M[0, 0]
        t = 1.0 / M[0, 0]
        
        # Power coefficients (reflectance and transmittance)
        n_inc = n_list[0]
        n_sub = n_list[-1]
        cos_θ_inc = bd.cos(θ_list[0])
        cos_θ_sub = bd.cos(θ_list[-1])
        
        R = bd.abs(r)**2
        
        # Transmittance: account for impedance mismatch
        if polarization.lower() == "s":
            # s-polarization: T = |t|² * Re(n_sub * cos(θ_sub)) / Re(n_inc * cos(θ_inc))
            T = bd.abs(t)**2 * bd.real(n_sub * cos_θ_sub) / bd.real(n_inc * cos_θ_inc)
        elif polarization.lower() == "p":
            # p-polarization: T = |t|² * Re(n_sub * cos(θ_sub) / n_sub²) / Re(n_inc * cos(θ_inc) / n_inc²)
            # Simplified: T = |t|² * Re(cos(θ_sub) / n_sub) / Re(cos(θ_inc) / n_inc)
            T = bd.abs(t)**2 * bd.real(cos_θ_sub / n_sub) / bd.real(cos_θ_inc / n_inc)
        else:  # mix
            # Average of s and p
            result_s = self._solve_multilayer(wavelength, θ_incident, "s")
            result_p = self._solve_multilayer(wavelength, θ_incident, "p")
            R = 0.5 * (result_s['R'] + result_p['R'])
            T = 0.5 * (result_s['T'] + result_p['T'])
            r = 0.5 * (result_s['r'] + result_p['r'])
            t = 0.5 * (result_s['t'] + result_p['t'])
        
        A = 1.0 - R - T
        
        return {
            'R': R,
            'T': T,
            'A': A,
            'r': r,
            't': t
        }
    
    def get_field_profile(self, wavelength, θ_incident=0.0, polarization="s", z_points=None):
        """
        Compute field profile E(z) through the stack.
        
        Parameters
        ----------
        wavelength : float
            Wavelength in meters
        θ_incident : float, optional
            Angle of incidence in radians
        polarization : str, optional
            "s" or "p" polarization
        z_points : array, optional
            z positions to sample (meters). If None, auto-generate.
            
        Returns
        -------
        dict
            Dictionary with 'z', 'E', 'H' field profiles
        """
        # This is a placeholder - full implementation would compute fields at each z
        # For now, return structure
        if z_points is None:
            # Generate z points through stack
            z_total = sum([layer.d for layer in self.layers[:-1] if layer.d is not None])
            z_points = np.linspace(0, z_total, 100)
        
        # TODO: Implement full field reconstruction
        # This requires back-propagating from substrate and forward from incident
        
        return {
            'z': z_points,
            'E': np.zeros_like(z_points, dtype=complex),
            'H': np.zeros_like(z_points, dtype=complex)
        }

