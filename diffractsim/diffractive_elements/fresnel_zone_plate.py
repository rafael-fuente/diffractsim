import numpy as np
from ..util.backend_functions import backend as bd
from .diffractive_element import DOE

class BinaryFZP(DOE):
    def __init__(self, f, λ, radius = None, aberration = None):
        """
        Creates a Phase Binary Fresnel Zone Plate with a focal length equal to f for a wavelength λ
        """
        global bd
        from ..util.backend_functions import backend as bd

        self.f = f
        self.FZP_λ = λ
        self.radius = radius

    def get_transmittance(self, xx, yy, λ):

        t = 1

        if self.radius != None:
            t = bd.where((xx**2 + yy**2) < self.radius**2, t, bd.zeros_like(xx))

        r_2 = xx**2 + yy**2

        phase_shift =  bd.pi* (bd.sign(((2*bd.pi/self.FZP_λ * (bd.sqrt(self.f**2 + r_2) - self.f))) % (2*bd.pi)  -  bd.pi  ))/2.
        t = t*bd.exp(1j*phase_shift)
        return t



class FZP(DOE):
    def __init__(self, f, λ, radius = None, aberration = None):
        """
        Creates a Phase Blazed (Ideal) Fresnel Zone Plate with a focal length equal to f for a wavelength λ
        """
        global bd
        from ..util.backend_functions import backend as bd

        self.f = f
        self.FZP_λ = λ
        self.radius = radius

    def get_transmittance(self, xx, yy, λ):

        t = 1

        if self.radius != None:
            t = bd.where((xx**2 + yy**2) < self.radius**2, t, bd.zeros_like(xx))

        r_2 = xx**2 + yy**2

        phase_shift = -(2*bd.pi/λ * (bd.sqrt(self.f**2 + r_2) - self.f))
        t = t*bd.exp(1j*phase_shift)
        return t
