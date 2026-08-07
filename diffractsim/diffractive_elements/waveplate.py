import numpy as np
from ..util.backend_functions import backend as bd
from .diffractive_element import DOE

class Waveplate(DOE):
    def __init__(self, phase_delay, fast_axis_angle):
        """
        Creates a waveplate with a given phase delay (in radians) and fast axis angle (in degrees)
        
        Parameters
        ----------
        phase_delay: Phase difference between the fast and slow axes (e.g., pi/2 for QWP, pi for HWP)
        fast_axis_angle: Angle of the fast axis relative to the x-axis (in degrees)
        """
        self.phase_delay = phase_delay
        self.theta = np.radians(fast_axis_angle)

    def apply_to_vectorial_field(self, field):
        """
        Applies the Jones matrix of a waveplate to the vectorial field
        """
        c = bd.cos(self.theta)
        s = bd.sin(self.theta)
        phi = self.phase_delay
        
        # Jones Matrix for a waveplate
        J11 = bd.exp(1j * phi/2) * c**2 + bd.exp(-1j * phi/2) * s**2
        J12 = (bd.exp(1j * phi/2) - bd.exp(-1j * phi/2)) * s * c
        J21 = J12
        J22 = bd.exp(1j * phi/2) * s**2 + bd.exp(-1j * phi/2) * c**2
        
        Ex_new = J11 * field.Ex + J12 * field.Ey
        Ey_new = J21 * field.Ex + J22 * field.Ey
        
        field.Ex = Ex_new
        field.Ey = Ey_new

    def get_transmittance(self, xx, yy, λ):
        return bd.ones_like(xx)

class HalfWavePlate(Waveplate):
    def __init__(self, fast_axis_angle):
        super().__init__(np.pi, fast_axis_angle)

class QuarterWavePlate(Waveplate):
    def __init__(self, fast_axis_angle):
        super().__init__(np.pi/2, fast_axis_angle)
