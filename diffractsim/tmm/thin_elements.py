"""
Thin optical elements: polarizers and retarders (Jones matrices).

MPL 2.0 Clause License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""

import numpy as np
from ..util.backend_functions import backend as bd


class IdealPolarizer:
    """
    Ideal linear polarizer (Jones matrix).
    
    Parameters
    ----------
    angle : float
        Transmission axis angle in radians (0 = x-axis, π/2 = y-axis)
    """
    
    def __init__(self, angle=0.0):
        self.angle = angle
    
    def get_jones_matrix(self):
        """
        Get Jones matrix for ideal polarizer.
        
        Returns
        -------
        array
            2x2 Jones matrix
        """
        θ = self.angle
        cos_θ = bd.cos(θ)
        sin_θ = bd.sin(θ)
        
        # Jones matrix for ideal linear polarizer
        # J = [[cos²(θ), cos(θ)sin(θ)], [cos(θ)sin(θ), sin²(θ)]]
        J = bd.array([
            [cos_θ**2, cos_θ * sin_θ],
            [cos_θ * sin_θ, sin_θ**2]
        ])
        
        return J
    
    def apply(self, field):
        """
        Apply polarizer to VectorialField.
        
        Parameters
        ----------
        field : VectorialField
            Input field
            
        Returns
        -------
        VectorialField
            Output field after polarizer
        """
        from ..vectorial_field import VectorialField
        
        # Convert to xy basis if needed
        if field.basis != "xy":
            field = field.to_basis("xy")
        
        J = self.get_jones_matrix()
        
        # Apply Jones matrix: [Ex_out, Ey_out] = J @ [Ex_in, Ey_in]
        Ex_flat = field.Ex.flatten()
        Ey_flat = field.Ey.flatten()
        
        # Vectorized application
        # Use numpy-style stacking that works with all backends
        if hasattr(bd, 'stack'):
            E_in = bd.stack([Ex_flat, Ey_flat], axis=0)  # Shape: (2, N)
        else:
            # Fallback for backends without stack
            E_in = bd.array([Ex_flat, Ey_flat])  # Shape: (2, N)
        
        # Apply matrix multiplication
        if hasattr(bd, 'tensordot'):
            E_out = bd.tensordot(J, E_in, axes=1)  # Shape: (2, N)
        else:
            # Fallback: use matrix multiplication
            E_out = bd.dot(J, E_in)  # Shape: (2, N)
        
        Ex_out = E_out[0].reshape(field.shape)
        Ey_out = E_out[1].reshape(field.shape)
        
        return VectorialField(
            Ex_out,
            Ey_out,
            field.λ,
            field.dx,
            field.dy,
            basis="xy"
        )


class IdealRetarder:
    """
    Ideal waveplate/retarder (Jones matrix).
    
    Parameters
    ----------
    retardance : float
        Retardance in radians (e.g., π/2 for quarter-wave, π for half-wave)
    axis_angle : float
        Fast axis angle in radians (0 = x-axis)
    """
    
    def __init__(self, retardance, axis_angle=0.0):
        self.retardance = retardance
        self.axis_angle = axis_angle
    
    def get_jones_matrix(self):
        """
        Get Jones matrix for ideal retarder.
        
        Returns
        -------
        array
            2x2 Jones matrix
        """
        δ = self.retardance
        θ = self.axis_angle
        
        cos_θ = bd.cos(θ)
        sin_θ = bd.sin(θ)
        
        # Rotation matrix
        R = bd.array([
            [cos_θ, sin_θ],
            [-sin_θ, cos_θ]
        ])
        
        # Retarder matrix in its principal axes
        R_ret = bd.array([
            [bd.exp(-1j * δ / 2), 0],
            [0, bd.exp(1j * δ / 2)]
        ])
        
        # Rotate back
        R_inv = bd.array([
            [cos_θ, -sin_θ],
            [sin_θ, cos_θ]
        ])
        
        # Jones matrix: R_inv @ R_ret @ R
        J = R_inv @ R_ret @ R
        
        return J
    
    def apply(self, field):
        """
        Apply retarder to VectorialField.
        
        Parameters
        ----------
        field : VectorialField
            Input field
            
        Returns
        -------
        VectorialField
            Output field after retarder
        """
        from ..vectorial_field import VectorialField
        
        # Convert to xy basis if needed
        if field.basis != "xy":
            field = field.to_basis("xy")
        
        J = self.get_jones_matrix()
        
        # Apply Jones matrix
        Ex_flat = field.Ex.flatten()
        Ey_flat = field.Ey.flatten()
        
        # Use numpy-style stacking that works with all backends
        if hasattr(bd, 'stack'):
            E_in = bd.stack([Ex_flat, Ey_flat], axis=0)
        else:
            E_in = bd.array([Ex_flat, Ey_flat])
        
        # Apply matrix multiplication
        if hasattr(bd, 'tensordot'):
            E_out = bd.tensordot(J, E_in, axes=1)
        else:
            E_out = bd.dot(J, E_in)
        
        Ex_out = E_out[0].reshape(field.shape)
        Ey_out = E_out[1].reshape(field.shape)
        
        return VectorialField(
            Ex_out,
            Ey_out,
            field.λ,
            field.dx,
            field.dy,
            basis="xy"
        )


# Convenience functions
def quarter_wave_plate(axis_angle=0.0):
    """Create quarter-wave plate (λ/4 retarder)."""
    return IdealRetarder(np.pi / 2, axis_angle)


def half_wave_plate(axis_angle=0.0):
    """Create half-wave plate (λ/2 retarder)."""
    return IdealRetarder(np.pi, axis_angle)

