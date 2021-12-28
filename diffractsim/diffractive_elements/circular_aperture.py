import numpy as np
from ..util.backend_functions import backend as bd
import numpy as np
from .diffractive_element import DOE

class CircularAperture(DOE):
    def __init__(self, radius , x0 = 0, y0 = 0):
        """
        Creates a circular slit centered at the point (x0,y0)
        """
        global bd
        from ..util.backend_functions import backend as bd

        self.x0 = x0
        self.y0 = y0
        self.radius = radius

    def get_transmittance(self, xx, yy, λ):

        t = bd.select(
            [(xx - self.x0) ** 2 + (yy - self.y0) ** 2 < self.radius ** 2, bd.ones_like(xx, dtype=bool)], [bd.ones_like(xx), bd.zeros_like(xx)]
        )

        return t

    def get_coherent_PSF(self,  xx, yy, z, λ):
        """ 
        Get the coherent point spread function (PSF) of the DEO when it acts as the pupil of an imaging system
        Exactly, this method returns the result of the following integral:

        PSF(x,y) = 1 / (z*λ)**2 * ∫∫  t(u, v) * exp(-1j*pi/ (z*λ) *(u*x + v*y)) * du*dv
        """

        if bd == np:
            from scipy import special
        else: 
            from cupyx.scipy import special


        rr = bd.sqrt(xx**2 + yy**2)
        tmp = 2*bd.pi*self.radius*rr/(λ*z)
        tmp = bd.where(tmp < 1e-9, 1e-9, tmp) #avoid division by 0

        PSF = 2 * bd.pi * self.radius**2 * (special.j1(tmp))/ tmp
        PSF = 1 / (z*λ)**2 * PSF
        
        return PSF
