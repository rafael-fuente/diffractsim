from . import colour_functions as cf
import matplotlib.pyplot as plt
import time
import progressbar
from .util.constants import *
from .propagation_methods.angular_spectrum_method import angular_spectrum_method

import numpy as np
from .util.backend_functions import backend as bd

"""
Vectorial Electric Field Model
"""

class VectorialField:
    def __init__(self, wavelength, extent_x, extent_y, Nx, Ny, intensity=0.1 * W / (m**2), pol_state=[1, 0]):
        """
        Initializes the vectorial field.

        Parameters
        ----------
        wavelength: wavelength of the plane wave
        extent_x: length of the rectangular grid 
        extent_y: height of the rectangular grid 
        Nx: horizontal dimension of the grid 
        Ny: vertical dimension of the grid 
        intensity: total intensity of the field
        pol_state: list [Ex_rel, Ey_rel] representing the relative polarization state
        """
        global bd
        from .util.backend_functions import backend as bd
        
        self.extent_x = extent_x
        self.extent_y = extent_y

        self.dx = extent_x/Nx
        self.dy = extent_y/Ny

        self.x = self.dx*(bd.arange(Nx)-Nx//2)
        self.y = self.dy*(bd.arange(Ny)-Ny//2)
        self.xx, self.yy = bd.meshgrid(self.x, self.y)

        self.Nx = Nx
        self.Ny = Ny
        self.λ = wavelength
        self.z = 0
        self.cs = cf.ColourSystem(clip_method = 0)
        
        # Normalize polarization state
        pol_norm = np.sqrt(np.abs(pol_state[0])**2 + np.abs(pol_state[1])**2)
        ex = pol_state[0] / pol_norm
        ey = pol_state[1] / pol_norm
        
        amplitude = bd.sqrt(intensity)
        self.Ex = bd.ones((self.Ny, self.Nx), dtype=complex) * amplitude * ex
        self.Ey = bd.ones((self.Ny, self.Nx), dtype=complex) * amplitude * ey
        self.Ez = bd.zeros((self.Ny, self.Nx), dtype=complex)

    def add(self, optical_element):
        """
        Interaction with an optical element.
        """
        if hasattr(optical_element, 'apply_to_vectorial_field'):
            optical_element.apply_to_vectorial_field(self)
        else:
            # Assume scalar element, apply to each component independently
            self.Ex = optical_element.get_E(self.Ex, self.xx, self.yy, self.λ)
            self.Ey = optical_element.get_E(self.Ey, self.xx, self.yy, self.λ)
            self.Ez = optical_element.get_E(self.Ez, self.xx, self.yy, self.λ)

    def propagate(self, z):
        """
        Propagate the vectorial field a distance z using the Angular Spectrum Method.
        """
        self.z += z
        
        # Fourier transform of Ex and Ey
        tilde_Ex0 = bd.fft.fftshift(bd.fft.fft2(self.Ex))
        tilde_Ey0 = bd.fft.fftshift(bd.fft.fft2(self.Ey))

        fx = bd.fft.fftshift(bd.fft.fftfreq(self.Nx, d=self.dx))
        fy = bd.fft.fftshift(bd.fft.fftfreq(self.Ny, d=self.dy))
        fxx, fyy = bd.meshgrid(fx, fy)

        # k_z calculation
        k = 2 * bd.pi / self.λ
        kx = 2 * bd.pi * fxx
        ky = 2 * bd.pi * fyy
        
        # Calculate kz, handling evanescent waves
        k_sq = k**2 - kx**2 - ky**2
        kz = bd.sqrt(bd.abs(k_sq))
        kz = bd.where(k_sq >= 0, kz, 1j * kz)

        # Propagation transfer function
        H = bd.exp(1j * kz * z)
        
        # Propagated angular spectrum for Ex and Ey
        tilde_Ex = tilde_Ex0 * H
        tilde_Ey = tilde_Ey0 * H
        
        # Calculate tilde_Ez from Gauss's Law: kx*Ex + ky*Ey + kz*Ez = 0
        # tilde_Ez = -(kx*tilde_Ex + ky*tilde_Ey) / kz
        # Note: avoid division by zero for kz
        kz_safe = bd.where(kz == 0, 1e-12, kz)
        tilde_Ez = -(kx * tilde_Ex + ky * tilde_Ey) / kz_safe

        # Inverse Fourier transform to get the spatial fields
        self.Ex = bd.fft.ifft2(bd.fft.ifftshift(tilde_Ex))
        self.Ey = bd.fft.ifft2(bd.fft.ifftshift(tilde_Ey))
        self.Ez = bd.fft.ifft2(bd.fft.ifftshift(tilde_Ez))

    def get_intensity(self):
        """
        Compute total intensity I = |Ex|^2 + |Ey|^2 + |Ez|^2
        """
        return bd.real(self.Ex * bd.conj(self.Ex) + self.Ey * bd.conj(self.Ey) + self.Ez * bd.conj(self.Ez))

    def get_stokes_parameters(self):
        """
        Compute the Stokes parameters (S0, S1, S2, S3)
        """
        S0 = bd.real(self.Ex * bd.conj(self.Ex) + self.Ey * bd.conj(self.Ey))
        S1 = bd.real(self.Ex * bd.conj(self.Ex) - self.Ey * bd.conj(self.Ey))
        S2 = bd.real(self.Ex * bd.conj(self.Ey) + self.Ey * bd.conj(self.Ex))
        S3 = bd.imag(self.Ex * bd.conj(self.Ey) - self.Ey * bd.conj(self.Ex))
        return S0, S1, S2, S3

    def get_colors(self):
        """compute RGB colors of the cross-section profile at the current distance"""
        I = self.get_intensity()
        rgb = self.cs.wavelength_to_sRGB(self.λ / nm, 10 * I.ravel()).T.reshape(
            (self.Ny, self.Nx, 3)
        )
        return rgb

    def plot_colors(self, **kwargs):
        from .visualization import plot_colors
        plot_colors(self, **kwargs)

    def plot_intensity(self, **kwargs):
        from .visualization import plot_intensity
        plot_intensity(self, **kwargs)
