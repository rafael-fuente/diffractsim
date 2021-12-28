import numpy as np
from ..util.backend_functions import backend as bd
from .diffractive_element import DOE

class Axicon(DOE):
    def __init__(self, period, radius = None, aberration = None):
        """
        Axicon that creates a beam with an approximate Bessel function profile
        """
        global bd
        from ..util.backend_functions import backend as bd

        self.period = period
        self.radius = radius

    def get_transmittance(self, xx, yy, λ):

        t = 1

        if self.radius != None:
            t = bd.where((xx**2 + yy**2) < self.radius**2, t, bd.zeros_like(xx))

        r = bd.sqrt(xx**2 + yy**2)

        phase_shift = -2*bd.pi*r/self.period
        t = t*bd.exp(1j*phase_shift)
        return t
