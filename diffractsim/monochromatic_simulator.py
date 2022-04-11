from . import colour_functions as cf
import matplotlib.pyplot as plt
import time
import progressbar
from .util.constants import *
from .propagation_methods import angular_spectrum_method, two_steps_fresnel_method, bluestein_method, apply_transfer_function

import numpy as np
from .util.backend_functions import backend as bd


"""
BSD 3-Clause License

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""


class MonochromaticField:
    def __init__(self,  wavelength, extent_x, extent_y, Nx, Ny, intensity = 0.1 * W / (m**2)):
        """
        Initializes the field, representing the cross-section profile of a plane wave

        Parameters
        ----------
        wavelength: wavelength of the plane wave
        extent_x: length of the rectangular grid 
        extent_y: height of the rectangular grid 
        Nx: horizontal dimension of the grid 
        Ny: vertical dimension of the grid 
        intensity: intensity of the field
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
        self.E = bd.ones((self.Ny, self.Nx)) * bd.sqrt(intensity)
        self.λ = wavelength
        self.z = 0
        self.cs = cf.ColourSystem(clip_method = 0)
        
    def add(self, optical_element):

        self.E = optical_element.get_E(self.E, self.xx, self.yy, self.λ)


    def propagate(self, z):
        """
        Compute the field in distance equal to z with the angular spectrum method
        The ouplut plane coordinates is the same than the input.
        """

        self.z += z
        self.E = angular_spectrum_method(self, self.E, z, self.λ)


    def scale_propagate(self, z, scale_factor):
        """
        Compute the field in distance equal to z with the two step Fresnel propagator, rescaling the field in the new coordinates
        with extent equal to:
        new_extent_x = scale_factor * self.extent_x
        new_extent_y = scale_factor * self.extent_y

        Reference: VOELZ, D. G. (2011). Computational Fourier optics. Bellingham, Wash, SPIE.
        
        Note that unlike within in the propagate method, Fresnel approximation is used here.
        To arbitrarily choose and zoom in a region of interest, use zoom_propagate method instead.
        """
        
        self.z += z
        self.E = two_steps_fresnel_method(self, self.E, z, self.λ, scale_factor)


    def zoom_propagate(self, z, x_interval, y_interval):
        """
        Compute the field in distance equal to z with the Bluestein method.
        Bluestein propagation is the more versatile method as the dimensions of the output plane can be arbitrarily chosen by using 
        the arguments x_interval and y_interval

        Parameters
        ----------

        x_interval: A length-2 sequence [x1, x2] giving the x outplut plane range
        y_interval: A length-2 sequence [y1, y2] giving the y outplut plane range

        Example of use:
        F.zoom_propagate(400*cm, x_interval = [-10*mm, 50*mm], y_interval = [-20*mm, 40*mm])

        Reference: 
        Hu, Y., Wang, Z., Wang, X. et al. Efficient full-path optical calculation of scalar and vector diffraction using the Bluestein method. 
        Light Sci Appl 9, 119 (2020).
        """
        
        self.z += z
        self.E = bluestein_method(self, self.E, z, self.λ, x_interval, y_interval)




    def propagate_to_image_plane(self, pupil, M, zi, z0, scale_factor = 1):
        """
        Assuming an optical system with linear response and assuming the system is only diffraction-limited by
        the exit pupil of the system, compute the field at its image plane

        
        Parameters
        ----------

        pupil: diffractive optical element used as exit pupil. Can be circular aperture, a diaphragm etc

        zi: distance from the image plane to the exit pupil
        z0: distance from the exit pupil to the current simulation plane

        M: magnification factor of the optical system
        (If the optical system is a single lens, magnification = - zi/z0)

        Reference:
        Introduction to Fourier Optics J. Goodman: Frequency Analysis of Optical Imaging Systems
        
        """
        self.z += zi + z0
        
        # if the magnification is negative, the image is inverted
        if M < 0:
            self.E = bd.flip(self.E)
        M_abs = bd.abs(M)

        self.E = self.E/M_abs

        fft_c = bd.fft.fft2(self.E)
        c = bd.fft.fftshift(fft_c)

        fx = bd.fft.fftshift(bd.fft.fftfreq(self.Nx, d = self.x[1]-self.x[0]))/M_abs
        fy = bd.fft.fftshift(bd.fft.fftfreq(self.Ny, d = self.y[1]-self.y[0]))/M_abs
        fxx, fyy = bd.meshgrid(fx, fy)

        H = pupil.get_amplitude_transfer_function(fxx, fyy, zi, self.λ)

        self.E = apply_transfer_function(self, self.E, self.λ, H, scale_factor)

        self.xx = M_abs * self.xx
        self.yy = M_abs * self.yy
        self.x = M_abs * self.x
        self.y = M_abs * self.y
        self.dx = M_abs * self.dx
        self.dy = M_abs * self.dy

    def get_colors(self):
        """compute RGB colors of the cross-section profile at the current distance"""

        # compute Field Intensity
        I = bd.real(self.E * bd.conjugate(self.E))  

        rgb = self.cs.wavelength_to_sRGB(self.λ / nm, 10 * I.ravel()).T.reshape(
            (self.Ny, self.Nx, 3)
        )
        return rgb


    def get_field(self):
        """get field of the cross-section profile at the current distance"""

        return self.E


    def get_intensity(self):
        """compute field intensity of the cross-section profile at the current distance"""

        return bd.real(self.E * bd.conjugate(self.E))  



    def compute_colors_at(self, z):
        """propagate the field to a distance equal to z and compute the RGB colors of the beam profile"""

        self.propagate(z)
        rgb = self.get_colors()
        return rgb

    def interpolate(self, Nx, Ny):
        """Interpolate the field to the new shape (Nx,Ny)"""
        from scipy.interpolate import interp2d


        if bd != np:
            self.E = self.E.get()

        fun_real = interp2d(
                    self.dx*(np.arange(self.Nx)-self.Nx//2),
                    self.dy*(np.arange(self.Ny)-self.Ny//2),
                    np.real(self.E),
                    kind="cubic",)

        fun_imag = interp2d(
                    self.dx*(np.arange(self.Nx)-self.Nx//2),
                    self.dy*(np.arange(self.Ny)-self.Ny//2),
                    np.imag(self.E),
                    kind="cubic",)


        self.Nx = Nx
        self.Ny = Ny

        self.dx = self.extent_x/Nx
        self.dy = self.extent_y/Ny

        self.E = bd.array(fun_real(self.dx*(np.arange(Nx)-Nx//2), self.dy*(np.arange(Ny)-Ny//2))  +  fun_imag(self.dx*(np.arange(Nx)-Nx//2), self.dy*(np.arange(Ny)-Ny//2))*1j)


        self.x = self.dx*(bd.arange(Nx)-Nx//2)
        self.y = self.dy*(bd.arange(Ny)-Ny//2)
        self.xx, self.yy = bd.meshgrid(self.x, self.y)


    def get_longitudinal_profile(self, start_distance, end_distance, steps):
        """
        Propagates the field at n steps equally spaced between start_distance and end_distance, and returns
        the colors and the field over the xz plane
        """

        z = bd.linspace(start_distance, end_distance, steps)

        self.E0 = self.E.copy()

        longitudinal_profile_rgb = bd.zeros((steps,self.Nx, 3))
        longitudinal_profile_E = bd.zeros((steps,self.Nx), dtype = complex)
        z0 = self.z 
        t0 = time.time()

        bar = progressbar.ProgressBar()
        for i in bar(range(steps)):
                 
            self.propagate(z[i])
            rgb = self.get_colors()
            longitudinal_profile_rgb[i,:,:]  = rgb[self.Ny//2,:,:]
            longitudinal_profile_E[i,:] = self.E[self.Ny//2,:]
            self.E = np.copy(self.E0)

        # restore intial values
        self.z = z0
        self.I = bd.real(self.E * bd.conjugate(self.E))  

        print ("Took", time.time() - t0)

        return longitudinal_profile_rgb, longitudinal_profile_E

    def __add__(self, Field):
        """
        Interfere MonochromaticField with another MonochromaticField instance. 
        The wavelength, dimensions and sampling of the interfering fields must be identical
        """

        if ((self.extent_x == Field.extent_x) and (self.extent_y == Field.extent_y) and (self.Nx == Field.Nx) and (self.Ny == Field.Ny) and (self.λ == Field.λ )):
            mixed_field = MonochromaticField(self.λ, self.extent_x, self.extent_y, self.Nx, self.Ny)
            mixed_field.E = self.E + Field.E
            return mixed_field

        else:
            raise ValueError(
            "The wavelength, dimensions and sampling of the interfering fields must be identical")


    from .visualization import plot_colors, plot_phase, plot_intensity, plot_longitudinal_profile_colors, plot_longitudinal_profile_intensity
