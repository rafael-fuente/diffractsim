from . import colour_functions as cf
import matplotlib.pyplot as plt
import time
import progressbar
from .util.constants import *
from .propagation_methods import angular_spectrum_method, two_steps_fresnel_method, bluestein_method, apply_transfer_function

import numpy as np
from .util.backend_functions import backend as bd





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

        self.dx = self.extent_x/Nx
        self.dy = self.extent_y/Ny

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

        # compute Field Intensity
        self.I = bd.real(self.E * bd.conjugate(self.E))  




    def propagate(self, z):
        """
        Compute the field in distance equal to z with the angular spectrum method
        The ouplut plane coordinates is the same than the input.
        """

        self.z += z
        self.E = angular_spectrum_method(self, self.E, z, self.λ)

        # compute Field Intensity
        self.I = bd.real(self.E * bd.conjugate(self.E))  



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

        # compute Field Intensity
        self.I = bd.real(self.E * bd.conjugate(self.E))  


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

        # compute Field Intensity
        self.I = bd.real(self.E * bd.conjugate(self.E))  




    def propagate_to_image_plane(self, pupil, zi, z0, scale_factor = 1):
        from scipy.interpolate import interp2d
        """
        Parameters
        ----------
        zi: distance from the image plane to the lens
        z0: distance from the lens the current position
        zi and z0 should satisfy the equation 1/zi + 1/z0 = 1/f 
        where f is the focal distance of the lens
        pupil: diffractive optical element used as pupil
        """
        self.z += zi + z0
        
        #magnification factor
        M = zi/z0

        if bd == np:
            fun = interp2d(self.x,self.y,self.E,kind="cubic",)
            self.E = fun(self.x/M, self.y/M )/M
            self.E = np.flip(self.E)
        else: 
            fun = interp2d(
                        self.extent_x*(np.arange(self.Nx)-self.Nx//2)/self.Nx,
                        self.extent_y*(np.arange(self.Ny)-self.Ny//2)/self.Ny,
                        self.E,
                        kind="cubic",)
            
            self.E = fun(self.extent_x*(np.arange(self.Nx)-self.Nx//2)/self.Nx/M, 
                       self.extent_y*(np.arange(self.Ny)-self.Ny//2)/self.Ny/M )/M
            self.E = bd.array(np.flip(self.E))

        fft_c = bd.fft.fft2(self.E)
        c = bd.fft.fftshift(fft_c)

        fx = bd.fft.fftshift(bd.fft.fftfreq(self.Nx, d = self.x[1]-self.x[0]))
        fy = bd.fft.fftshift(bd.fft.fftfreq(self.Ny, d = self.y[1]-self.y[0]))
        fxx, fyy = bd.meshgrid(fx, fy)

        H = pupil.get_amplitude_transfer_function(fxx, fyy, zi, self.λ)

        self.E = apply_transfer_function(self, self.E, self.λ, H, scale_factor)

        # compute Field Intensity
        self.I = bd.real(self.E * bd.conjugate(self.E))  


    def get_colors(self):
        """ compute RGB colors"""

        rgb = self.cs.wavelength_to_sRGB(self.λ / nm, 10 * self.I.ravel()).T.reshape(
            (self.Ny, self.Nx, 3)
        )
        return rgb


    def compute_colors_at(self, z):
        """propagate the field to a distance equal to z and compute the RGB colors of the beam profile"""

        self.propagate(z)
        rgb = self.get_colors()
        return rgb


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



    from .visualization import plot_colors, plot_intensity, plot_longitudinal_profile_colors, plot_longitudinal_profile_intensity
