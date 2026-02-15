import numpy as np
from ..util.backend_functions import backend as bd
from .light_source import LightSource

class PlaneWave(LightSource):
    def __init__(self, Ex_amplitude=1.0, Ey_amplitude=0.0, phase_diff=0.0):
        global bd
        from ..util.backend_functions import backend as bd
        self.Ex_amplitude = Ex_amplitude
        self.Ey_amplitude = Ey_amplitude
        self.phase_diff = phase_diff

    def get_E(self, E, xx, yy, lam):
        return bd.ones_like(xx) * E

    def get_E_components(self, Ex, Ey, xx, yy, lam):
        Ex_mod = self.get_E(Ex, xx, yy, lam)
        Ey_mod = self.get_E(Ey, xx, yy, lam)
        Ex_out = self.Ex_amplitude * Ex_mod
        Ey_out = self.Ey_amplitude * bd.exp(1j * self.phase_diff) * Ey_mod
        return Ex_out, Ey_out
