import numpy as np
from ..util.backend_functions import backend as bd
from .light_source import LightSource

class PlaneWave(LightSource):
    def __init__(self, Ex_amplitude=1.0, Ey_amplitude=0.0, phase_diff=0.0):
        """
        Creates a plane wave
        """
        global bd
        from ..util.backend_functions import backend as bd
        self.Ex_amplitude = Ex_amplitude
        self.Ey_amplitude = Ey_amplitude
        self.phase_diff = phase_diff

    def get_E(self, E, xx, yy, λ):
        
        return bd.ones_like(xx) * E
