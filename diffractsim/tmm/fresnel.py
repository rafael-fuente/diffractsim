"""
Fresnel coefficients for s and p polarizations.

MPL 2.0 Clause License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""

import numpy as np
from ..util.backend_functions import backend as bd


def fresnel_coefficients(n1, n2, θ1, polarization="s"):
    """
    Calculate Fresnel reflection and transmission coefficients.
    
    Parameters
    ----------
    n1 : complex or array
        Refractive index of incident medium
    n2 : complex or array
        Refractive index of transmitted medium
    θ1 : float or array
        Angle of incidence in radians
    polarization : str, optional
        "s" for s-polarization (TE), "p" for p-polarization (TM). Default is "s".
        
    Returns
    -------
    r : complex or array
        Reflection coefficient
    t : complex or array
        Transmission coefficient
    θ2 : float or array
        Angle of refraction (Snell's law)
    """
    # Convert to arrays for broadcasting
    n1 = bd.asarray(n1)
    n2 = bd.asarray(n2)
    θ1 = bd.asarray(θ1)
    
    # Snell's law: n1*sin(θ1) = n2*sin(θ2)
    sin_θ2 = (n1 / n2) * bd.sin(θ1)
    
    # Handle total internal reflection
    cos_θ2 = bd.sqrt(1 - sin_θ2**2 + 0j)  # Complex square root for evanescent waves
    
    cos_θ1 = bd.cos(θ1)
    
    if polarization.lower() == "s":
        # s-polarization (TE): E perpendicular to plane of incidence
        r = (n1 * cos_θ1 - n2 * cos_θ2) / (n1 * cos_θ1 + n2 * cos_θ2)
        t = (2 * n1 * cos_θ1) / (n1 * cos_θ1 + n2 * cos_θ2)
    
    elif polarization.lower() == "p":
        # p-polarization (TM): E parallel to plane of incidence
        # Note: At normal incidence, this should match s-polarization
        r = (n1 * cos_θ2 - n2 * cos_θ1) / (n1 * cos_θ2 + n2 * cos_θ1)
        t = (2 * n1 * cos_θ1) / (n1 * cos_θ2 + n2 * cos_θ1)
    
    else:
        raise ValueError(f'polarization must be "s" or "p", got "{polarization}"')
    
    θ2 = bd.arcsin(sin_θ2)
    
    return r, t, θ2

