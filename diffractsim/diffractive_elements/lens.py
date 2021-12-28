import numpy as np
from ..util.backend_functions import backend as bd
from .diffractive_element import DOE

class Lens(DOE):
    def __init__(self,f, radius = None, aberration = None):
        """
        Creates a thin lens with a focal length equal to f 
        """
        global bd
        from ..util.backend_functions import backend as bd

        self.f = f
        self.aberration = aberration
        self.radius = radius

    def get_transmittance(self, xx, yy, λ):

        t = 1
        if self.aberration != None:
            t = t*bd.exp(2*bd.pi * 1j *aberration(xx, yy))

        if self.radius != None:
            t = bd.where((xx**2 + yy**2) < self.radius**2, t, bd.zeros_like(xx))

        self.t = t

        t = self.t * bd.exp(-1j*bd.pi/(λ*self.f) * (xx**2 + yy**2))
        return t
