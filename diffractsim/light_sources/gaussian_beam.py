import numpy as np
from ..util.backend_functions import backend as bd
from .light_source import LightSource

class GaussianBeam(LightSource):
    def __init__(self, w0):
        """
        Creates a Gaussian beam with waist radius equal to w0
        """
        global bd
        from ..util.backend_functions import backend as bd

        self.w0 = w0

    def get_E(self, E, xx, yy, λ):
        """Returns a Gaussian beam with circular polarization"""
        r2 = xx**2 + yy**2
        amplitude = E*bd.exp(-r2/(self.w0**2))
        # Circular polarization components
        Ex = amplitude * bd.exp(1j * 2*bd.pi/λ * bd.sqrt(r2))
        Ey = amplitude * bd.exp(1j * (2*bd.pi/λ * bd.sqrt(r2) + bd.pi/2))
        return Ex, Ey
