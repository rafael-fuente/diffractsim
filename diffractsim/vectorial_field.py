"""
MPL 2.0 Clause License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""

import numpy as np
from .util.backend_functions import backend as bd
from .util.constants import *


class VectorialField:
    """
    Vectorial electromagnetic field with Ex and Ey components.
    
    Supports basis transforms between (x,y) and (s,p) polarization bases.
    
    Parameters
    ----------
    Ex : array-like
        Electric field x-component (complex)
    Ey : array-like
        Electric field y-component (complex)
    wavelength : float
        Wavelength in meters
    dx : float
        Spatial sampling in x-direction (meters)
    dy : float
        Spatial sampling in y-direction (meters)
    basis : str, optional
        Basis representation: "xy" (default) or "sp" (s-polarization, p-polarization)
    """
    
    def __init__(self, Ex, Ey, wavelength, dx, dy, basis="xy"):
        """
        Initialize VectorialField with validation.
        """
        # Convert to backend arrays
        Ex = bd.asarray(Ex)
        Ey = bd.asarray(Ey)
        
        # Validate shapes match
        if Ex.shape != Ey.shape:
            raise ValueError(f"Ex and Ey must have the same shape. Got Ex: {Ex.shape}, Ey: {Ey.shape}")
        
        # Validate basis
        if basis not in ["xy", "sp"]:
            raise ValueError(f'basis must be "xy" or "sp", got "{basis}"')
        
        self.Ex = Ex
        self.Ey = Ey
        self.λ = wavelength
        self.dx = dx
        self.dy = dy
        self.basis = basis
        self.shape = Ex.shape
        
        # Store grid dimensions
        if len(Ex.shape) == 2:
            self.Ny, self.Nx = Ex.shape
        else:
            self.Nx = Ex.shape[0] if len(Ex.shape) == 1 else Ex.shape[-1]
            self.Ny = Ex.shape[0] if len(Ex.shape) == 1 else Ex.shape[-2]
    
    def to_basis(self, target_basis, angle=0.0):
        """
        Transform field to target basis.
        
        Parameters
        ----------
        target_basis : str
            Target basis: "xy" or "sp"
        angle : float, optional
            Angle for s/p basis (radians). For s/p basis, angle is the angle of incidence
            in the plane of incidence. Default is 0.
            
        Returns
        -------
        VectorialField
            New VectorialField in target basis
        """
        if target_basis == self.basis:
            return VectorialField(self.Ex.copy(), self.Ey.copy(), self.λ, self.dx, self.dy, self.basis)
        
        if self.basis == "xy" and target_basis == "sp":
            # Transform from (x,y) to (s,p)
            # s-polarization: perpendicular to plane of incidence (y-direction for normal incidence)
            # p-polarization: parallel to plane of incidence (x-direction for normal incidence)
            # For angle=0 (normal incidence), s = y, p = x
            cos_a = bd.cos(angle)
            sin_a = bd.sin(angle)
            
            # Rotation matrix from (x,y) to (s,p)
            # Es = -sin(θ)*Ex + cos(θ)*Ey
            # Ep = cos(θ)*Ex + sin(θ)*Ey
            Es = -sin_a * self.Ex + cos_a * self.Ey
            Ep = cos_a * self.Ex + sin_a * self.Ey
            
            return VectorialField(Es, Ep, self.λ, self.dx, self.dy, basis="sp")
        
        elif self.basis == "sp" and target_basis == "xy":
            # Transform from (s,p) to (x,y)
            cos_a = bd.cos(angle)
            sin_a = bd.sin(angle)
            
            # Inverse rotation
            Ex = -sin_a * self.Ex + cos_a * self.Ey
            Ey = cos_a * self.Ex + sin_a * self.Ey
            
            return VectorialField(Ex, Ey, self.λ, self.dx, self.dy, basis="xy")
        
        else:
            raise ValueError(f"Unknown basis transformation: {self.basis} -> {target_basis}")
    
    def intensity(self):
        """
        Compute total intensity: |Ex|² + |Ey|²
        
        Returns
        -------
        array
            Intensity distribution
        """
        return bd.real(self.Ex * bd.conj(self.Ex) + self.Ey * bd.conj(self.Ey))
    
    def stokes(self):
        """
        Compute Stokes parameters.
        
        Returns
        -------
        dict
            Dictionary with keys 'S0', 'S1', 'S2', 'S3'
        """
        # Convert to xy basis if needed
        if self.basis == "sp":
            field_xy = self.to_basis("xy")
            Ex = field_xy.Ex
            Ey = field_xy.Ey
        else:
            Ex = self.Ex
            Ey = self.Ey
        
        # Stokes parameters
        S0 = bd.real(Ex * bd.conj(Ex) + Ey * bd.conj(Ey))  # Total intensity
        S1 = bd.real(Ex * bd.conj(Ex) - Ey * bd.conj(Ey))  # Linear polarization (0°/90°)
        S2 = bd.real(Ex * bd.conj(Ey) + Ey * bd.conj(Ex))  # Linear polarization (45°/135°)
        S3 = bd.imag(Ex * bd.conj(Ey) - Ey * bd.conj(Ex))  # Circular polarization
        
        return {
            'S0': S0,
            'S1': S1,
            'S2': S2,
            'S3': S3
        }
    
    def __add__(self, other):
        """
        Add two VectorialFields (interference).
        
        Parameters
        ----------
        other : VectorialField
            Field to add
            
        Returns
        -------
        VectorialField
            Sum of fields
        """
        if not isinstance(other, VectorialField):
            raise TypeError("Can only add VectorialField to VectorialField")
        
        if self.shape != other.shape:
            raise ValueError(f"Fields must have same shape. Got {self.shape} and {other.shape}")
        
        if abs(self.λ - other.λ) > 1e-12:
            raise ValueError(f"Fields must have same wavelength. Got {self.λ} and {other.λ}")
        
        if abs(self.dx - other.dx) > 1e-12 or abs(self.dy - other.dy) > 1e-12:
            raise ValueError("Fields must have same spatial sampling")
        
        # Convert to same basis if needed
        if self.basis != other.basis:
            other = other.to_basis(self.basis)
        
        return VectorialField(
            self.Ex + other.Ex,
            self.Ey + other.Ey,
            self.λ,
            self.dx,
            self.dy,
            self.basis
        )
    
    def __mul__(self, scalar):
        """
        Multiply field by scalar (amplitude scaling).
        
        Parameters
        ----------
        scalar : complex or float
            Scaling factor
            
        Returns
        -------
        VectorialField
            Scaled field
        """
        return VectorialField(
            scalar * self.Ex,
            scalar * self.Ey,
            self.λ,
            self.dx,
            self.dy,
            self.basis
        )
    
    def __rmul__(self, scalar):
        """Right multiplication."""
        return self.__mul__(scalar)

