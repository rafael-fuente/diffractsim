import numpy as np
from ..util.backend_functions import backend as bd
from .diffractive_element import DOE

class RectangularSlit(DOE):
    def __init__(self, width, height, x0 = 0, y0 = 0):

        """
        Creates a slit centered at the point (x0, y0) with width width and height height
        """

        global bd
        from ..util.backend_functions import backend as bd

        self.x0 = x0
        self.y0 = y0

        self.width = width
        self.height = height

    def get_transmittance(self, xx, yy, λ):

        t = bd.where((((xx >= (self.x0 - self.width / 2)) & (xx < (self.x0 + self.width / 2)))
                        & ((yy >= (self.y0 - self.height / 2)) & (yy < (self.y0 + self.height / 2)))),
                        bd.ones_like(xx), bd.zeros_like(xx))

        return t
