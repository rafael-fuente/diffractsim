"""
MPL 2.0 Clause License

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.

Vectorial EM Field Model for diffractsim
Supports vector representation of electromagnetic fields with Jones vector polarization
and transfer matrix method integration.
"""

from . import colour_functions as cf
import numpy as np
from .util.backend_functions import backend as bd
from .util.backend_functions import backend_name


class JonesVector:
    """
    Represents the polarization state of an electromagnetic wave using Jones calculus.

    The Jones vector describes the amplitude and phase of the Ex and Ey components
    of the electric field in the basis (x, y).

    Parameters
    ----------
    Ex : complex
        Complex amplitude of the x-component
    Ey : complex
        Complex amplitude of the y-component

    Examples
    --------
    >>> # Horizontal polarization
    >>> jv = JonesVector(1, 0)
    >>> # Vertical polarization
    >>> jv = JonesVector(0, 1)
    >>> # Right-hand circular polarization
    >>> jv = JonesVector(1, -1j)
    >>> # Left-hand circular polarization
    >>> jv = JonesVector(1, 1j)
    >>> # 45 degree linear polarization
    >>> jv = JonesVector(1, 1)
    """

    def __init__(self, Ex, Ey):
        self.Ex = complex(Ex)
        self.Ey = complex(Ey)

    @classmethod
    def horizontal(cls):
        """Create a horizontally polarized Jones vector"""
        return cls(1, 0)

    @classmethod
    def vertical(cls):
        """Create a vertically polarized Jones vector"""
        return cls(0, 1)

    @classmethod
    def linear(cls, angle_deg):
        """Create a linearly polarized Jones vector at specified angle

        Parameters
        ----------
        angle_deg : float
            Polarization angle in degrees (0 = horizontal, 90 = vertical)
        """
        angle_rad = np.deg2rad(angle_deg)
        return cls(np.cos(angle_rad), np.sin(angle_rad))

    @classmethod
    def right_circular(cls):
        """Create a right-hand circularly polarized Jones vector"""
        return cls(1, -1j)

    @classmethod
    def left_circular(cls):
        """Create a left-hand circularly polarized Jones vector"""
        return cls(1, 1j)

    @property
    def intensity(self):
        """Return the total intensity (normalized)"""
        return abs(self.Ex)**2 + abs(self.Ey)**2

    @property
    def normalized(self):
        """Return normalized Jones vector"""
        I = self.intensity
        if I > 0:
            return JonesVector(self.Ex / np.sqrt(I), self.Ey / np.sqrt(I))
        return JonesVector(0, 0)

    def __repr__(self):
        return f"JonesVector(Ex={self.Ex:.3f}, Ey={self.Ey:.3f})"


class JonesMatrix:
    """
    Represents a 2x2 Jones matrix for describing polarization transformations.

    Jones matrices can represent various optical elements like wave plates,
    polarizers, rotators, etc.

    Parameters
    ----------
    matrix : array-like, shape (2, 2)
        2x2 complex matrix
    """

    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=complex)

    @classmethod
    def identity(cls):
        """Create identity matrix (no polarization change)"""
        return cls([[1, 0], [0, 1]])

    @classmethod
    def linear_polarizer(cls, angle_deg):
        """Create a linear polarizer matrix

        Parameters
        ----------
        angle_deg : float
            Transmission axis angle in degrees
        """
        theta = np.deg2rad(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        return cls([[cos_t**2, cos_t*sin_t],
                    [cos_t*sin_t, sin_t**2]])

    @classmethod
    def phase_retarder(cls, delta, theta=0):
        """Create a phase retarder matrix (wave plate)

        Parameters
        ----------
        delta : float
            Phase retardation in radians
        theta : float
            Fast axis angle in degrees (default: 0)
        """
        theta = np.deg2rad(theta)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        exp_delta = np.exp(1j * delta)

        # Matrix elements for general wave plate
        a = cos_t**2 + exp_delta * sin_t**2
        b = (1 - exp_delta) * cos_t * sin_t
        c = b
        d = sin_t**2 + exp_delta * cos_t**2

        return cls([[a, b], [c, d]])

    @classmethod
    def quarter_wave_plate(cls, theta=0):
        """Create a quarter-wave plate (lambda/4)"""
        return cls.phase_retarder(np.pi/2, theta)

    @classmethod
    def half_wave_plate(cls, theta=0):
        """Create a half-wave plate (lambda/2)"""
        return cls.phase_retarder(np.pi, theta)

    @classmethod
    def rotator(cls, angle_deg):
        """Create an optical rotator matrix

        Parameters
        ----------
        angle_deg : float
            Rotation angle in degrees
        """
        theta = np.deg2rad(angle_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        return cls([[cos_t, sin_t], [-sin_t, cos_t]])

    def apply(self, jones_vector):
        """Apply the Jones matrix to a Jones vector

        Parameters
        ----------
        jones_vector : JonesVector
            Input polarization state

        Returns
        -------
        JonesVector
            Output polarization state after transformation
        """
        Ex_out = self.matrix[0, 0] * jones_vector.Ex + self.matrix[0, 1] * jones_vector.Ey
        Ey_out = self.matrix[1, 0] * jones_vector.Ex + self.matrix[1, 1] * jones_vector.Ey
        return JonesVector(Ex_out, Ey_out)

    def __matmul__(self, other):
        """Matrix multiplication (A @ B)"""
        if isinstance(other, JonesMatrix):
            return JonesMatrix(self.matrix @ other.matrix)
        elif isinstance(other, JonesVector):
            return self.apply(other)
        return NotImplemented

    def __repr__(self):
        return f"JonesMatrix({self.matrix.tolist()})"


class TransferMatrix:
    """
    Represents a transfer matrix for optical systems using the transfer matrix method.

    The transfer matrix method is used to propagate electromagnetic waves through
    layered optical systems. Each layer is represented by a 2x2 transfer matrix
    that relates the forward and backward propagating waves at each interface.

    Parameters
    ----------
    n1 : float
        Refractive index of medium 1
    n2 : float
        Refractive index of medium 2
    d : float
        Thickness of the layer (in meters)
    wavelength : float
        Wavelength of the incident light (in meters)
    angle : float, optional
        Angle of incidence in radians (default: 0)
    """

    def __init__(self, n1, n2, d, wavelength, angle=0):
        self.n1 = n1
        self.n2 = n2
        self.d = d
        self.wavelength = wavelength
        self.angle = angle
        self.k = 2 * np.pi / wavelength

    def get_interface_matrix(self):
        """Get the interface matrix at an angle of incidence

        Returns
        -------
        JonesMatrix
            Interface transmission/reflection matrix
        """
        # Fresnel coefficients for s and p polarizations
        cos_i = np.cos(self.angle)
        # Snell's law: n1*sin(i) = n2*sin(t)
        sin_i = np.sin(self.angle)

        # Handle total internal reflection case
        if self.n1 > self.n2:
            n_ratio = self.n2 / self.n1
            cos_t = np.sqrt(1 - (n_ratio * sin_i)**2)
            if np.isnan(cos_t):
                # Total internal reflection - return identity for transmission
                return JonesMatrix.identity()
        else:
            cos_t = np.sqrt(1 - (self.n1/self.n2 * sin_i)**2)

        # Fresnel equations
        # s-polarization (TE)
        ts = 2 * self.n1 * cos_i / (self.n1 * cos_i + self.n2 * cos_t)
        rs = (self.n1 * cos_i - self.n2 * cos_t) / (self.n1 * cos_i + self.n2 * cos_t)

        # p-polarization (TM)
        tp = 2 * self.n1 * cos_i / (self.n2 * cos_i + self.n1 * cos_t)
        rp = (self.n2 * cos_i - self.n1 * cos_t) / (self.n2 * cos_i + self.n1 * cos_t)

        # Return as Jones matrix for the polarization components
        # Format: [[ts, rp], [rs, tp]] - relates (Es_forward, Ep_forward) to output
        return JonesMatrix([[ts, rp], [rs, tp]])

    def get_propagation_matrix(self):
        """Get the propagation matrix through the layer

        Returns
        -------
        JonesMatrix
            Propagation matrix with phase accumulation
        """
        cos_t = np.sqrt(1 - (self.n1/self.n2 * np.sin(self.angle))**2)
        phase = self.k * self.n2 * self.d * cos_t
        return JonesMatrix([[np.exp(1j * phase), 0], [0, np.exp(-1j * phase)]])

    def get_total_matrix(self):
        """Get the total transfer matrix for the layer

        Returns
        -------
        JonesMatrix
            Total transfer matrix
        """
        interface = self.get_interface_matrix()
        propagation = self.get_propagation_matrix()
        return propagation @ interface


class VectorialMonochromaticField:
    """
    Represents a vectorial electromagnetic field with full polarization information.

    This class extends the scalar MonochromaticField to support vectorial EM field
    modeling, including Jones vector polarization representation and transfer matrix
    method integration.

    Parameters
    ----------
    wavelength : float
        Wavelength of the light (in meters)
    extent_x : float
        Length of the rectangular grid in x direction (in meters)
    extent_y : float
        Height of the rectangular grid in y direction (in meters)
    Nx : int
        Horizontal dimension of the grid
    Ny : int
        Vertical dimension of the grid
    intensity : float, optional
        Intensity of the field (default: 0.1 W/m^2)
    jones_vector : JonesVector, optional
        Initial polarization state (default: horizontal)
    """

    def __init__(self, wavelength, extent_x, extent_y, Nx, Ny,
                 intensity=0.1, jones_vector=None):
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

        # Initialize field amplitude
        amplitude = bd.sqrt(intensity)

        # Jones vector for polarization
        if jones_vector is None:
            self.jones_vector = JonesVector.horizontal()
        else:
            self.jones_vector = jones_vector

        # Create field components based on Jones vector
        # The field is stored as a dictionary with Ex and Ey components
        self.Ex = bd.ones((self.Ny, self.Nx), dtype=complex) * self.jones_vector.Ex * amplitude
        self.Ey = bd.ones((self.Ny, self.Nx), dtype=complex) * self.jones_vector.Ey * amplitude
        self.Ez = bd.zeros((self.Ny, self.Nx), dtype=complex)  # Longitudinal component (for near-field)

        self.z = 0
        self.cs = cf.ColourSystem(clip_method=0)

    def set_polarization(self, jones_vector):
        """Set the polarization state of the field

        Parameters
        ----------
        jones_vector : JonesVector
            New polarization state
        """
        # Get current intensity distribution
        I = self.get_intensity()

        # Update Jones vector
        self.jones_vector = jones_vector

        # Update field components
        amplitude = bd.sqrt(I)
        self.Ex = self.Ex / bd.abs(self.Ex + 1e-20) * jones_vector.Ex * amplitude
        self.Ey = self.Ey / bd.abs(self.Ey + 1e-20) * jones_vector.Ey * amplitude

    def add(self, optical_element):
        """Add a diffractive optical element to the field

        Parameters
        ----------
        optical_element : DiffractiveElement
            Optical element to add
        """
        # Get the scalar field transformation
        E = optical_element.get_E(bd.ones((self.Ny, self.Nx)), self.xx, self.yy, self.λ)

        # Apply to both components
        self.Ex = self.Ex * E
        self.Ey = self.Ey * E

    def add_polarization_element(self, jones_matrix):
        """Add a polarization-transforming optical element

        Parameters
        ----------
        jones_matrix : JonesMatrix
            Polarization transformation matrix
        """
        # Apply Jones matrix to the field components
        # This treats each point as having the same polarization
        Ex_new = jones_matrix.matrix[0, 0] * self.Ex + jones_matrix.matrix[0, 1] * self.Ey
        Ey_new = jones_matrix.matrix[1, 0] * self.Ex + jones_matrix.matrix[1, 1] * self.Ey

        self.Ex = Ex_new
        self.Ey = Ey_new

        # Update Jones vector
        self.jones_vector = jones_matrix.apply(self.jones_vector)

    def propagate(self, z, scale_factor=1):
        """
        Compute the field at distance z using the angular spectrum method.

        For vectorial fields, we propagate the transverse components (Ex, Ey)
        using the scalar propagator and compute the longitudinal component (Ez)
        from the divergence-free condition.

        Parameters
        ----------
        z : float
            Propagation distance (in meters)
        scale_factor : float, optional
            Scale factor for output coordinates (default: 1)
        """
        from .propagation_methods import angular_spectrum_method

        self.z += z

        # Propagate Ex and Ey using angular spectrum method
        self.Ex = angular_spectrum_method(self, self.Ex, z, self.λ, scale_factor)
        self.Ey = angular_spectrum_method(self, self.Ey, z, self.λ, scale_factor)

        # Compute Ez from Maxwell's equations (divergence-free condition)
        # For propagation in z-direction: Ez = -(Ex_x + Ey_y) / kz
        # This is approximate and valid for paraxial approximation
        k = 2 * np.pi / self.λ

        # Compute spatial derivatives (approximation)
        dEx_dx = bd.gradient(self.Ex, axis=1) / self.dx
        dEy_dy = bd.gradient(self.Ey, axis=0) / self.dy

        # Ez from div(E) = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            self.Ez = - (dEx_dx + dEy_dy) / (1j * k)
            self.Ez = bd.nan_to_num(self.Ez, nan=0, posinf=0, neginf=0)

    def scale_propagate(self, z, scale_factor):
        """
        Propagate using two-step Fresnel method with scaling.

        Parameters
        ----------
        z : float
            Propagation distance (in meters)
        scale_factor : float
            Scale factor for output coordinates
        """
        from .propagation_methods import two_steps_fresnel_method

        self.z += z

        self.x, self.y, self.Ex = two_steps_fresnel_method(self, self.Ex, z, self.λ, scale_factor)
        _, _, self.Ey = two_steps_fresnel_method(self, self.Ey, z, self.λ, scale_factor)

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.xx, self.yy = bd.meshgrid(self.x, self.y)
        self.extent_x = self.Nx * self.dx
        self.extent_y = self.Ny * self.dy

        # Clear Ez since it's invalid after scaling
        self.Ez = bd.zeros((self.Ny, self.Nx), dtype=complex)

    def zoom_propagate(self, z, x_interval, y_interval):
        """
        Propagate using Bluestein method with arbitrary output region.

        Parameters
        ----------
        z : float
            Propagation distance (in meters)
        x_interval : list
            [x1, x2] output range in x direction
        y_interval : list
            [y1, y2] output range in y direction
        """
        from .propagation_methods import bluestein_method

        self.z += z

        self.x, self.y, self.Ex = bluestein_method(self, self.Ex, z, self.λ, x_interval, y_interval)
        _, _, self.Ey = bluestein_method(self, self.Ey, z, self.λ, x_interval, y_interval)

        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.xx, self.yy = bd.meshgrid(self.x, self.y)
        self.extent_x = self.Nx * self.dx
        self.extent_y = self.Ny * self.dy

        # Clear Ez
        self.Ez = bd.zeros((self.Ny, self.Nx), dtype=complex)

    def apply_transfer_matrix(self, n1, n2, d, angle=0):
        """
        Apply transfer matrix method for propagation through a layer.

        Parameters
        ----------
        n1 : float
            Refractive index of incident medium
        n2 : float
            Refractive index of layer
        d : float
            Layer thickness (in meters)
        angle : float, optional
            Angle of incidence in radians (default: 0)
        """
        # Get the transfer matrix
        tm = TransferMatrix(n1, n2, d, self.λ, angle)
        jm = tm.get_total_matrix()

        # Apply to field
        self.add_polarization_element(jm)

    def get_field(self):
        """Get the vector field as a dictionary"""
        return {'Ex': self.Ex, 'Ey': self.Ey, 'Ez': self.Ez}

    def get_transverse_field(self):
        """Get the transverse field components (Ex, Ey) as a complex array"""
        return bd.stack([self.Ex, self.Ey], axis=-1)

    def get_intensity(self):
        """
        Compute the total field intensity.

        For vectorial fields: I = |Ex|^2 + |Ey|^2 + |Ez|^2

        Returns
        -------
        array
            Intensity distribution
        """
        return bd.real(bd.conj(self.Ex) * self.Ex +
                       bd.conj(self.Ey) * self.Ey +
                       bd.conj(self.Ez) * self.Ez)

    def get_transverse_intensity(self):
        """
        Compute the transverse field intensity (only Ex, Ey).

        Returns
        -------
        array
            Transverse intensity distribution
        """
        return bd.real(bd.conj(self.Ex) * self.Ex + bd.conj(self.Ey) * self.Ey)

    def get_colors(self):
        """
        Compute RGB colors of the cross-section profile.

        Uses the transverse intensity for color computation.

        Returns
        -------
        array
            RGB color array
        """
        I = self.get_transverse_intensity()

        rgb = self.cs.wavelength_to_sRGB(self.λ / 1e-9, 10 * I.ravel()).T.reshape(
            (self.Ny, self.Nx, 3)
        )
        return rgb

    def get_stokes_parameters(self):
        """
        Compute the Stokes parameters for each point in the field.

        Returns
        -------
        dict
            Dictionary with S0, S1, S2, S3 Stokes parameters

        Note
        ----
        S0 = I = |Ex|^2 + |Ey|^2
        S1 = |Ex|^2 - |Ey|^2
        S2 = 2*Re(Ex*conj(Ey))
        S3 = -2*Im(Ex*conj(Ey))
        """
        I = self.get_transverse_intensity()

        # S1: horizontal - vertical
        S1 = bd.real(bd.conj(self.Ex) * self.Ex - bd.conj(self.Ey) * self.Ey)

        # S2: +45 - -45 degrees
        S2 = 2 * bd.real(bd.conj(self.Ex) * self.Ey)

        # S3: right - left circular
        S3 = -2 * bd.imag(bd.conj(self.Ex) * self.Ey)

        return {'S0': I, 'S1': S1, 'S2': S2, 'S3': S3}

    def get_degree_of_polarization(self):
        """
        Compute the degree of polarization (DOP) for each point.

        Returns
        -------
        array
            DOP distribution (0 = unpolarized, 1 = fully polarized)
        """
        S = self.get_stokes_parameters()
        S0 = S['S0']
        S1 = S['S1']
        S2 = S['S2']
        S3 = S['S3']

        with np.errstate(divide='ignore', invalid='ignore'):
            DOP = bd.sqrt(S1**2 + S2**2 + S3**2) / (S0 + 1e-20)
            DOP = bd.nan_to_num(DOP, nan=0, posinf=1, neginf=1)

        return DOP

    def __add__(self, other):
        """
        Combine two VectorialMonochromaticField instances.

        Parameters
        ----------
        other : VectorialMonochromaticField
            Field to add

        Returns
        -------
        VectorialMonochromaticField
            Combined field
        """
        if not isinstance(other, VectorialMonochromaticField):
            return NotImplemented

        if (self.extent_x == other.extent_x and
            self.extent_y == other.extent_y and
            self.Nx == other.Nx and
            self.Ny == other.Ny and
            self.λ == other.λ):

            new_field = VectorialMonochromaticField(
                self.λ, self.extent_x, self.extent_y, self.Nx, self.Ny
            )
            new_field.Ex = self.Ex + other.Ex
            new_field.Ey = self.Ey + other.Ey
            new_field.Ez = self.Ez + other.Ez
            new_field.jones_vector = JonesVector.horizontal()  # Mixed polarization
            return new_field
        else:
            raise ValueError(
                "The wavelength, dimensions and sampling of the interfering fields must be identical"
            )

    # Visualization methods
    from .visualization.plot_colors import plot_colors as plot_colors_v
    from .visualization.plot_intensity import plot_intensity as plot_intensity_v
    from .visualization.plot_vectorial_field import plot_stokes_parameters as plot_stokes_parameters_v

    plot_colors = plot_colors_v
    plot_intensity = plot_intensity_v
    plot_stokes_parameters = plot_stokes_parameters_v
