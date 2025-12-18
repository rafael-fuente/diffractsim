"""
Vectorial Electromagnetic Field Simulator for Diffractsim

This module extends diffractsim to support vectorial EM field representation
with polarization support. Implements Jones calculus for polarization state
manipulation and propagation.

References:
- Jones, R. C. (1941). A new calculus for the treatment of optical systems.
  Journal of the Optical Society of America, 31(7), 488-493.
- Born, M., & Wolf, E. (1999). Principles of optics. Cambridge University Press.

MPL 2.0 License
Copyright (c) 2025
"""

from . import colour_functions as cf
import numpy as np
from .util.constants import *
from .propagation_methods import angular_spectrum_method
from .util.backend_functions import backend as bd


class JonesVector:
    """
    Represents a Jones vector for polarization state.

    A Jones vector (Ex, Ey) represents the complex amplitudes of the
    x and y components of the electric field.

    Common polarization states:
    - Horizontal: (1, 0)
    - Vertical: (0, 1)
    - Diagonal (+45°): (1/√2, 1/√2)
    - Anti-diagonal (-45°): (1/√2, -1/√2)
    - Right circular: (1/√2, -i/√2)
    - Left circular: (1/√2, i/√2)
    """

    # Predefined polarization states
    HORIZONTAL = np.array([1, 0], dtype=complex)
    VERTICAL = np.array([0, 1], dtype=complex)
    DIAGONAL = np.array([1, 1], dtype=complex) / np.sqrt(2)
    ANTI_DIAGONAL = np.array([1, -1], dtype=complex) / np.sqrt(2)
    RIGHT_CIRCULAR = np.array([1, -1j], dtype=complex) / np.sqrt(2)
    LEFT_CIRCULAR = np.array([1, 1j], dtype=complex) / np.sqrt(2)

    def __init__(self, Ex, Ey):
        """
        Initialize a Jones vector.

        Parameters
        ----------
        Ex : complex
            Complex amplitude of x-component
        Ey : complex
            Complex amplitude of y-component
        """
        self.vector = np.array([Ex, Ey], dtype=complex)

    @classmethod
    def from_angle(cls, theta, phase_diff=0):
        """
        Create a Jones vector from polarization angle.

        Parameters
        ----------
        theta : float
            Polarization angle in radians (0 = horizontal)
        phase_diff : float
            Phase difference between Ey and Ex in radians
        """
        Ex = np.cos(theta)
        Ey = np.sin(theta) * np.exp(1j * phase_diff)
        return cls(Ex, Ey)

    @property
    def Ex(self):
        return self.vector[0]

    @property
    def Ey(self):
        return self.vector[1]

    def normalize(self):
        """Return normalized Jones vector."""
        norm = np.sqrt(np.abs(self.Ex)**2 + np.abs(self.Ey)**2)
        if norm > 0:
            return JonesVector(self.Ex / norm, self.Ey / norm)
        return self

    def intensity(self):
        """Return total intensity."""
        return np.abs(self.Ex)**2 + np.abs(self.Ey)**2

    def stokes_parameters(self):
        """
        Convert to Stokes parameters (S0, S1, S2, S3).

        Returns
        -------
        tuple
            (S0, S1, S2, S3) Stokes parameters
        """
        Ex, Ey = self.Ex, self.Ey
        S0 = np.abs(Ex)**2 + np.abs(Ey)**2
        S1 = np.abs(Ex)**2 - np.abs(Ey)**2
        S2 = 2 * np.real(Ex * np.conj(Ey))
        S3 = 2 * np.imag(Ex * np.conj(Ey))
        return S0, S1, S2, S3


class JonesMatrix:
    """
    Jones matrix for optical elements.

    A 2x2 complex matrix that transforms Jones vectors.
    """

    def __init__(self, matrix):
        """
        Initialize Jones matrix.

        Parameters
        ----------
        matrix : array_like
            2x2 complex matrix
        """
        self.matrix = np.array(matrix, dtype=complex).reshape(2, 2)

    @classmethod
    def linear_polarizer(cls, angle=0):
        """
        Linear polarizer at given angle.

        Parameters
        ----------
        angle : float
            Transmission axis angle in radians
        """
        c, s = np.cos(angle), np.sin(angle)
        matrix = np.array([
            [c**2, c*s],
            [c*s, s**2]
        ])
        return cls(matrix)

    @classmethod
    def half_wave_plate(cls, angle=0):
        """
        Half-wave plate (λ/2) with fast axis at given angle.
        """
        c, s = np.cos(2*angle), np.sin(2*angle)
        matrix = np.array([
            [c, s],
            [s, -c]
        ])
        return cls(matrix)

    @classmethod
    def quarter_wave_plate(cls, angle=0):
        """
        Quarter-wave plate (λ/4) with fast axis at given angle.
        """
        c, s = np.cos(angle), np.sin(angle)
        matrix = np.exp(1j * np.pi/4) * np.array([
            [c**2 + 1j*s**2, (1-1j)*c*s],
            [(1-1j)*c*s, s**2 + 1j*c**2]
        ])
        return cls(matrix)

    @classmethod
    def waveplate(cls, retardance, angle=0):
        """
        General waveplate with arbitrary retardance.

        Parameters
        ----------
        retardance : float
            Phase retardance in radians
        angle : float
            Fast axis angle in radians
        """
        c, s = np.cos(angle), np.sin(angle)
        phase = np.exp(1j * retardance / 2)
        matrix = np.array([
            [c**2 * phase + s**2 / phase, c*s*(phase - 1/phase)],
            [c*s*(phase - 1/phase), s**2 * phase + c**2 / phase]
        ])
        return cls(matrix)

    def __matmul__(self, other):
        """Matrix multiplication with another Jones matrix or vector."""
        if isinstance(other, JonesMatrix):
            return JonesMatrix(self.matrix @ other.matrix)
        elif isinstance(other, JonesVector):
            result = self.matrix @ other.vector
            return JonesVector(result[0], result[1])
        else:
            raise TypeError(f"Cannot multiply JonesMatrix with {type(other)}")


class VectorialField:
    """
    Vectorial electromagnetic field with polarization support.

    Extends the scalar MonochromaticField to support full vectorial
    field representation with Ex and Ey components.
    """

    def __init__(self, wavelength, extent_x, extent_y, Nx, Ny,
                 intensity=0.1 * W / (m**2), polarization=None):
        """
        Initialize vectorial field.

        Parameters
        ----------
        wavelength : float
            Wavelength of the field
        extent_x : float
            Physical width of the simulation domain
        extent_y : float
            Physical height of the simulation domain
        Nx : int
            Number of grid points in x
        Ny : int
            Number of grid points in y
        intensity : float
            Initial intensity
        polarization : JonesVector or str, optional
            Initial polarization state. Can be:
            - JonesVector instance
            - String: 'horizontal', 'vertical', 'diagonal',
                      'anti_diagonal', 'right_circular', 'left_circular'
            - None (defaults to horizontal)
        """
        global bd
        from .util.backend_functions import backend as bd

        self.extent_x = extent_x
        self.extent_y = extent_y
        self.dx = extent_x / Nx
        self.dy = extent_y / Ny

        self.x = self.dx * (bd.arange(Nx) - Nx // 2)
        self.y = self.dy * (bd.arange(Ny) - Ny // 2)
        self.xx, self.yy = bd.meshgrid(self.x, self.y)

        self.Nx = Nx
        self.Ny = Ny
        self.λ = wavelength
        self.z = 0
        self.cs = cf.ColourSystem(clip_method=0)

        # Set polarization
        if polarization is None:
            self.polarization = JonesVector(1, 0)  # Horizontal
        elif isinstance(polarization, str):
            pol_map = {
                'horizontal': JonesVector.HORIZONTAL,
                'vertical': JonesVector.VERTICAL,
                'diagonal': JonesVector.DIAGONAL,
                'anti_diagonal': JonesVector.ANTI_DIAGONAL,
                'right_circular': JonesVector.RIGHT_CIRCULAR,
                'left_circular': JonesVector.LEFT_CIRCULAR,
            }
            if polarization.lower() in pol_map:
                self.polarization = JonesVector(
                    pol_map[polarization.lower()][0],
                    pol_map[polarization.lower()][1]
                )
            else:
                raise ValueError(f"Unknown polarization: {polarization}")
        elif isinstance(polarization, JonesVector):
            self.polarization = polarization
        else:
            raise TypeError("polarization must be JonesVector, string, or None")

        # Initialize field components
        amplitude = bd.sqrt(intensity)
        base_field = bd.ones((Ny, Nx), dtype=complex) * amplitude
        self.Ex = base_field * self.polarization.Ex
        self.Ey = base_field * self.polarization.Ey

    @property
    def E(self):
        """Return scalar field (total amplitude) for compatibility."""
        return bd.sqrt(bd.abs(self.Ex)**2 + bd.abs(self.Ey)**2)

    def add(self, optical_element):
        """
        Apply an optical element to the field.

        Parameters
        ----------
        optical_element : DiffractiveElement or JonesMatrix
            Element to apply
        """
        if isinstance(optical_element, JonesMatrix):
            # Apply Jones matrix to each spatial point
            Ex_new = optical_element.matrix[0, 0] * self.Ex + optical_element.matrix[0, 1] * self.Ey
            Ey_new = optical_element.matrix[1, 0] * self.Ex + optical_element.matrix[1, 1] * self.Ey
            self.Ex = Ex_new
            self.Ey = Ey_new
        else:
            # Standard diffractive element - apply to both components
            self.Ex = optical_element.get_E(self.Ex, self.xx, self.yy, self.λ)
            self.Ey = optical_element.get_E(self.Ey, self.xx, self.yy, self.λ)

    def propagate(self, z, scale_factor=1):
        """
        Propagate the vectorial field using angular spectrum method.

        Parameters
        ----------
        z : float
            Propagation distance
        scale_factor : float
            Output scaling factor
        """
        self.z += z
        # Propagate each polarization component independently
        self.Ex = angular_spectrum_method(self, self.Ex, z, self.λ, scale_factor=scale_factor)
        self.Ey = angular_spectrum_method(self, self.Ey, z, self.λ, scale_factor=scale_factor)

    def get_intensity(self):
        """Return total intensity."""
        return bd.real(self.Ex * bd.conj(self.Ex) + self.Ey * bd.conj(self.Ey))

    def get_intensity_x(self):
        """Return x-polarized intensity."""
        return bd.real(self.Ex * bd.conj(self.Ex))

    def get_intensity_y(self):
        """Return y-polarized intensity."""
        return bd.real(self.Ey * bd.conj(self.Ey))

    def get_stokes_parameters(self):
        """
        Compute Stokes parameters at each spatial point.

        Returns
        -------
        tuple
            (S0, S1, S2, S3) arrays with same shape as field
        """
        S0 = bd.abs(self.Ex)**2 + bd.abs(self.Ey)**2
        S1 = bd.abs(self.Ex)**2 - bd.abs(self.Ey)**2
        S2 = 2 * bd.real(self.Ex * bd.conj(self.Ey))
        S3 = 2 * bd.imag(self.Ex * bd.conj(self.Ey))
        return S0, S1, S2, S3

    def get_degree_of_polarization(self):
        """
        Compute degree of polarization at each point.

        Returns
        -------
        ndarray
            Degree of polarization (0 = unpolarized, 1 = fully polarized)
        """
        S0, S1, S2, S3 = self.get_stokes_parameters()
        return bd.sqrt(S1**2 + S2**2 + S3**2) / (S0 + 1e-20)

    def get_ellipticity(self):
        """
        Compute polarization ellipticity at each point.

        Returns
        -------
        ndarray
            Ellipticity (-1 = left circular, 0 = linear, +1 = right circular)
        """
        S0, S1, S2, S3 = self.get_stokes_parameters()
        return S3 / (bd.sqrt(S1**2 + S2**2 + S3**2) + 1e-20)

    def get_colors(self):
        """Compute RGB colors based on total intensity."""
        I = self.get_intensity()
        rgb = self.cs.wavelength_to_sRGB(self.λ / nm, 10 * I.ravel()).T.reshape(
            (self.Ny, self.Nx, 3)
        )
        return rgb

    def __add__(self, other):
        """Superpose two vectorial fields."""
        if not isinstance(other, VectorialField):
            raise TypeError("Can only add VectorialField instances")

        if not (self.extent_x == other.extent_x and
                self.extent_y == other.extent_y and
                self.Nx == other.Nx and
                self.Ny == other.Ny and
                self.λ == other.λ):
            raise ValueError("Fields must have matching dimensions and wavelength")

        result = VectorialField(self.λ, self.extent_x, self.extent_y,
                               self.Nx, self.Ny)
        result.Ex = self.Ex + other.Ex
        result.Ey = self.Ey + other.Ey
        return result


# Convenience functions for common polarization operations
def create_polarized_beam(wavelength, extent_x, extent_y, Nx, Ny,
                         polarization='horizontal', intensity=0.1 * W / (m**2)):
    """
    Create a polarized beam.

    Parameters
    ----------
    wavelength : float
        Wavelength
    extent_x, extent_y : float
        Spatial extents
    Nx, Ny : int
        Grid dimensions
    polarization : str or JonesVector
        Polarization state
    intensity : float
        Beam intensity

    Returns
    -------
    VectorialField
        Initialized vectorial field
    """
    return VectorialField(wavelength, extent_x, extent_y, Nx, Ny,
                         intensity=intensity, polarization=polarization)


def apply_polarization_element(field, element_type, angle=0, **kwargs):
    """
    Apply a polarization optical element to a vectorial field.

    Parameters
    ----------
    field : VectorialField
        Input field
    element_type : str
        One of: 'polarizer', 'half_wave', 'quarter_wave', 'waveplate'
    angle : float
        Element orientation angle in radians
    **kwargs
        Additional arguments (e.g., retardance for general waveplate)

    Returns
    -------
    VectorialField
        Modified field (in-place)
    """
    if element_type == 'polarizer':
        matrix = JonesMatrix.linear_polarizer(angle)
    elif element_type == 'half_wave':
        matrix = JonesMatrix.half_wave_plate(angle)
    elif element_type == 'quarter_wave':
        matrix = JonesMatrix.quarter_wave_plate(angle)
    elif element_type == 'waveplate':
        retardance = kwargs.get('retardance', np.pi)
        matrix = JonesMatrix.waveplate(retardance, angle)
    else:
        raise ValueError(f"Unknown element type: {element_type}")

    field.add(matrix)
    return field
