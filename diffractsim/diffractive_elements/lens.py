import numpy as np
from ..util.backend_functions import backend as bd
from .diffractive_element import DOE
from ..util.scaled_FT import scaled_fourier_transform

class Lens(DOE):
    def __init__(self,f, radius = None, aberration = None):
        """
        Creates a thin lens with a focal length equal to f. 

        Radius is a physical circular boundary of the lens.

        Aberration is a function of (x, y) which describes the optical
        path depth aberration of the lens. This is applied along with
        any focal length given by `f`.
        """
        global bd
        from ..util.backend_functions import backend as bd

        self.f = f
        self.aberration = aberration
        self.radius = radius

    def get_transmittance(self, xx, yy, λ):

        t = 1
        if self.aberration != None:
            t = t*bd.exp(2j * bd.pi / λ * self.aberration(xx, yy))

        if self.radius != None:
            t = bd.where((xx**2 + yy**2) < self.radius**2, t, bd.zeros_like(xx))

        self.t = t

        t = self.t * bd.exp(-1j*bd.pi/(λ*self.f) * (xx**2 + yy**2))
        return t




    def get_coherent_PSF(self,  xx, yy, z, λ):
        """ 
        Get the coherent point spread function (PSF) of the lens pupil.
        Exactly, this method returns the result of the following integral:

        PSF(x,y) = 1 / (z*λ)**2 * ∫∫  t(u, v) * exp(-1j*pi/ (z*λ) *(u*x + v*y)) * du*dv
        """

        if (self.aberration != None) and (self.radius != None):

            if bd == np:
                from scipy import special
            else: 
                from cupyx.scipy import special

            # we use an analytical solution:

            rr = bd.sqrt(xx**2 + yy**2)
            tmp = 2*bd.pi*self.radius*rr/(λ*z)
            tmp = bd.where(tmp < 1e-9, 1e-9, tmp) #avoid division by 0

            PSF = 2 * bd.pi * self.radius**2 * (special.j1(tmp))/ tmp
            PSF = 1 / (z*λ)**2 * PSF
            
            return PSF

        else:
            t = bd.ones_like(xx)

            if self.aberration != None:
                t = t*bd.exp(2*bd.pi * 1j * self.aberration(xx, yy))

            if self.radius != None:
                t = t*bd.where((xx**2 + yy**2) < self.radius**2, t, bd.zeros_like(xx))



        xx, yy, PSF = scaled_fourier_transform(xx, yy, t, λ = λ,z =z, scale_factor = 1, mesh = True)
        PSF = 1 / (z*λ)**2 * PSF
        
        return PSF