from abc import ABC, abstractmethod
from ..util.scaled_FT import scaled_fourier_transform

"""

MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.

"""

class DOE(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_transmittance(self, xx, yy, λ):
        pass

    def __add__(self, DOE2):
        return DOE_mix(self, DOE2)

    def get_E(self, E, xx, yy, λ):
        # by default the behavior of all DOE is linear in amplitude
        return E*self.get_transmittance(xx, yy, λ)

    def get_coherent_PSF(self,  xx, yy, z, λ):
        """ 
        Get the coherent point spread function (PSF) of the DOE when it acts as the pupil of an imaging system
        Exactly, this method returns the result of the following integral:

        PSF(x,y) = 1 / (z*λ)**2 * ∫∫  t(u, v) * exp(-1j*pi/ (z*λ) *(u*x + v*y)) * du*dv
        """

        xx, yy, PSF = scaled_fourier_transform(xx, yy, self.get_transmittance(xx, yy, λ), λ = λ,z =z, scale_factor = 1, mesh = True)
        PSF = 1 / (z*λ)**2 * PSF
        
        return PSF


    def get_incoherent_PSF(self,  xx, yy, z, λ):
        """ 
        Get the incoherent point spread function of the DOE when it acts as the pupil of an imaging system
        """
        return bd.abs(self.get_coherent_PSF(xx, yy, z, λ))**2



    def get_amplitude_transfer_function(self,  fxx, fyy, z, λ):
        """ 
        Get the (coherent) amplitude transfer function (ATF) of the DOE when it acts as the pupil of an imaging system
        """
        return self.get_transmittance(-fxx*λ*z, -fyy*λ*z, λ)


    def get_optical_transfer_function(self,  fxx, fyy, z, λ):
        """ 
        Get the (incoherent) optical transfer function (OTF) of the DOE when it acts as the pupil of an imaging system
        """
        global bd
        from ..util.backend_functions import backend as bd

        h = bd.fft.ifft2(bd.fft.ifftshift(self.get_amplitude_transfer_function(fxx, fyy, z, λ)))
        H = bd.fft.fftshift(bd.fft.fft2(h*bd.conjugate(h))) 

        dfx = fxx[0,1]-fxx[0,0]
        dfy = fyy[1,0]-fyy[0,0]

        #normalize OTF
        H = H/bd.amax(bd.abs(H))


        return H



class DOE_mix(DOE):
    def __init__(self, DOE1, DOE2):
        self.DOE1 = DOE1
        self.DOE2 = DOE2

    def get_transmittance(self, xx, yy, λ):
        return self.DOE1.get_transmittance(xx, yy, λ) + self.DOE2.get_transmittance(xx, yy, λ)

    def get_coherent_PSF(self, xx, yy, λ):
        return self.DOE1.get_coherent_PSF(xx, yy, z, λ) + self.DOE2.get_coherent_PSF(xx, yy, z, λ)
