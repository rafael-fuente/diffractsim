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

        r2 = xx**2 + yy**2 
        E = E*bd.exp(-r2/(self.w0**2))
        return E
