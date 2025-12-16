"""
Surface Plasmon Polariton (SPP) calculations.

MPL 2.0 Clause License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""

import numpy as np
from ..util.backend_functions import backend as bd
from .stack import Stack
from .layer import Layer
from .fresnel import fresnel_coefficients


def spp_dispersion_relation(ε_m, ε_d, k0):
    """
    Calculate SPP wavevector from dispersion relation.
    
    Parameters
    ----------
    ε_m : complex
        Permittivity of metal
    ε_d : complex or float
        Permittivity of dielectric
    k0 : float
        Free-space wavevector (2π/λ)
        
    Returns
    -------
    k_spp : complex
        SPP wavevector
    """
    # SPP dispersion: k_spp = k0 * sqrt(ε_m * ε_d / (ε_m + ε_d))
    k_spp = k0 * bd.sqrt(ε_m * ε_d / (ε_m + ε_d))
    return k_spp


def spp_effective_index(ε_m, ε_d):
    """
    Calculate SPP effective index.
    
    Parameters
    ----------
    ε_m : complex
        Permittivity of metal
    ε_d : complex or float
        Permittivity of dielectric
        
    Returns
    -------
    n_eff : complex
        Effective index n_eff = k_spp / k0
    """
    n_eff = bd.sqrt(ε_m * ε_d / (ε_m + ε_d))
    return n_eff


def kretschmann_configuration(prism_n, metal_layer, dielectric_n, wavelength, angle_range=None):
    """
    Kretschmann configuration: prism-metal-dielectric.
    
    Parameters
    ----------
    prism_n : float or callable
        Prism refractive index
    metal_layer : Layer
        Metal layer (should have finite thickness)
    dielectric_n : float or callable
        Dielectric (substrate) refractive index
    wavelength : float or array
        Wavelength(s) in meters
    angle_range : tuple, optional
        (θ_min, θ_max, n_points) for angle sweep. If None, single angle calculation.
        
    Returns
    -------
    dict
        Results dictionary with reflectance, transmittance, etc.
    """
    # Create stack: prism (incident) - metal - dielectric (substrate)
    layers = [
        Layer(prism_n, name="prism"),
        metal_layer,
        Layer(dielectric_n, name="dielectric")
    ]
    
    stack = Stack(layers)
    
    if angle_range is None:
        # Single angle (use resonance angle estimate)
        # For now, use normal incidence
        result = stack.solve(wavelength, θ_incident=0.0, polarization="p")
        return result
    else:
        # Angle sweep
        θ_min, θ_max, n_points = angle_range
        angles = np.linspace(θ_min, θ_max, n_points)
        
        R = []
        T = []
        A = []
        
        for θ in angles:
            result = stack.solve(wavelength, θ_incident=θ, polarization="p")
            R.append(result['R'])
            T.append(result['T'])
            A.append(result['A'])
        
        # Find resonance (minimum R)
        R_array = np.array(R)
        resonance_idx = np.argmin(R_array)
        resonance_angle = angles[resonance_idx]
        
        return {
            'angles': angles,
            'R': np.array(R),
            'T': np.array(T),
            'A': np.array(A),
            'resonance_angle': resonance_angle,
            'min_R': R_array[resonance_idx]
        }


def single_interface_spp(metal_n, dielectric_n, wavelength):
    """
    Single interface SPP (metal-dielectric).
    
    Parameters
    ----------
    metal_n : complex
        Metal complex refractive index
    dielectric_n : float or complex
        Dielectric refractive index
    wavelength : float
        Wavelength in meters
        
    Returns
    -------
    dict
        SPP properties including effective index, penetration depths
    """
    k0 = 2 * np.pi / wavelength
    
    # Permittivities
    ε_m = metal_n**2
    ε_d = dielectric_n**2 if isinstance(dielectric_n, complex) or np.iscomplexobj(dielectric_n) else dielectric_n**2
    
    # SPP effective index
    n_eff = spp_effective_index(ε_m, ε_d)
    k_spp = k0 * n_eff
    
    # Penetration depths
    # In metal: δ_m = 1 / Im(k_z_m)
    # In dielectric: δ_d = 1 / Im(k_z_d)
    k_z_m = bd.sqrt(ε_m * k0**2 - k_spp**2)
    k_z_d = bd.sqrt(ε_d * k0**2 - k_spp**2)
    
    δ_m = 1.0 / bd.imag(k_z_m) if bd.imag(k_z_m) > 0 else np.inf
    δ_d = 1.0 / bd.imag(k_z_d) if bd.imag(k_z_d) > 0 else np.inf
    
    return {
        'n_eff': n_eff,
        'k_spp': k_spp,
        'penetration_depth_metal': δ_m,
        'penetration_depth_dielectric': δ_d,
        'wavelength': wavelength
    }

