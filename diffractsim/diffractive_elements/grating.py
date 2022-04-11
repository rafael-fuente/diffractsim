import numpy as np
from ..util.backend_functions import backend as bd
from .diffractive_element import DOE



class BinaryGrating(DOE):
    def __init__(self, period, width, height, x0 = 0, y0 = 0):
        """
        Creates a binary (amplitude) rectangular grating at the point (x0, y0) with width width and height height
        """
        global bd
        from ..util.backend_functions import backend as bd

        self.period = period
        self.x0 = x0
        self.y0 = y0

        self.width = width
        self.height = height

    def get_transmittance(self, xx, yy, λ):

        t = bd.sign((xx) % (self.period) - self.period/2)
        t = bd.select([t==0, t==1, t==-1], [bd.ones_like(t), bd.ones_like(t),  bd.zeros_like(t)])

        t = t*bd.where((((xx >= (self.x0 - self.width / 2)) & (xx < (self.x0 + self.width / 2)))
                        & ((yy >= (self.y0 - self.height / 2)) & (yy < (self.y0 + self.height / 2)))),
                        bd.ones_like(xx), bd.zeros_like(xx))

        return t



class PhaseGrating(DOE):
    def __init__(self, period, width, height, x0 = 0, y0 = 0):
        """
        Creates a phase grating at the point (x0, y0) with width width and height height
        """
        global bd
        from ..util.backend_functions import backend as bd

        self.period = period
        self.x0 = x0
        self.y0 = y0

        self.width = width
        self.height = height

    def get_transmittance(self, xx, yy, λ):


        t = bd.where((((xx > (self.x0 - self.width / 2)) & (xx < (self.x0 + self.width / 2)))
                        & ((yy > (self.y0 - self.height / 2)) & (yy < (self.y0 + self.height / 2)))),
                        bd.ones_like(xx), bd.zeros_like(xx))

        phase_shift = xx/self.period
        return t*bd.exp(1j*2*bd.pi*phase_shift)

