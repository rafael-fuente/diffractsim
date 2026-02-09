import numpy as np
from ..util.backend_functions import backend as bd
from .light_source import LightSource

class GaussianBeam(LightSource):
    def __init__(self, w0, Ex_amplitude=1.0, Ey_amplitude=0.0, phase_diff=0.0):
        """
        Creates a Gaussian beam with waist radius equal to w0
        """
        global bd
        from ..util.backend_functions import backend as bd

        self.w0 = w0
        self.Ex_amplitude = Ex_amplitude
        self.Ey_amplitude = Ey_amplitude
        self.phase_diff = phase_diff

    def get_E(self, E, xx, yy, λ):

        r2 = xx**2 + yy**2 
        E = E*bd.exp(-r2/(self.w0**2))
        return E
