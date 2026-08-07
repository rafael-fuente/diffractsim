import numpy as np
from ..util.backend_functions import backend as bd
from .light_source import LightSource

class PlaneWave(LightSource):
    def __init__(self):
        """
        Creates a Gaussian beam with waist radius equal to w0
        """
        global bd
        from ..util.backend_functions import backend as bd

    def get_E(self, E, xx, yy, λ):
        """Returns a plane wave with uniform polarization"""
        # Return a vector field with x and y components
        Ex = bd.ones_like(xx) * E
        Ey = bd.zeros_like(xx)
        return Ex, Ey
