import numpy as np
from ..util.backend_functions import backend as bd
from .diffractive_element import DOE

class ApertureFromFunction(DOE):
    def __init__(self, function):
        """
        Evaluate a function with arguments 'x : 2D  array' , 'y : 2D array' and 'λ : float' as the amplitude transmittance of the aperture. 
        """
        global bd
        from ..util.backend_functions import backend as bd

        self.function = function

    def get_transmittance(self, xx, yy, λ):

        t = self.function(xx, yy, λ)
        return t
