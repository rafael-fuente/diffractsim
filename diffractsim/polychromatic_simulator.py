from . import colour_functions as cf
import matplotlib.pyplot as plt
import progressbar
from scipy.interpolate import interp2d
from pathlib import Path
from PIL import Image
import time
from .propagation_methods import angular_spectrum_method, two_steps_fresnel_method, apply_transfer_function

import numpy as np
from .util.backend_functions import backend as bd
from .util.constants import *


"""
MPL 2.0 Clause License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""


class PolychromaticField:
    def __init__(self, spectrum, extent_x, extent_y, Nx, Ny, spectrum_size = 180, spectrum_divisions = 30):
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
        self.E = bd.ones((self.Ny, self.Nx))

        if not(spectrum_size/spectrum_divisions).is_integer():
            raise ValueError("spectrum_size/spectrum_divisions must be an integer")

        if spectrum_size == 400: 
            self.spectrum = bd.array(spectrum)
        else: #by default spectrum has a size of 400. If new size, we interpolate
            self.spectrum = bd.array(np.interp(np.linspace(380,779, spectrum_size), np.linspace(380,779, 400), spectrum))

        self.spectrum_divisions = spectrum_divisions
        self.dλ_partition = (780 - 380) / self.spectrum_divisions
        self.λ_list_samples = bd.arange(380, 780, self.dλ_partition)
        self.spec_partitions = bd.split(self.spectrum, self.spectrum_divisions)

        self.cs = cf.ColourSystem(spectrum_size = spectrum_size, spec_divisions = spectrum_divisions, clip_method = 1)

        self.z = 0

        self.steps = []
        self.steps_type = []
        self.steps_args = []
        self.optical_elements = []
        self.number_of_propagations = 0

    def add(self, optical_element):

        self.optical_elements += [optical_element]
        self.steps += [optical_element]
        self.steps_type += ['optical_element']
        self.steps_args += [None]

    def propagate(self, z, spectrum_divisions=40, grid_divisions=10):
        """compute the field in distance equal to z with the angular spectrum method"""
        self.z += z

        self.steps += [angular_spectrum_method]
        self.number_of_propagations += 1
        self.steps_type += ['propagation']

        scale_factor = 1
        self.steps_args += [[z, scale_factor]]


    def get_colors(self):

        t0 = time.time()

        propagation_index = np.zeros(self.spectrum_divisions)

        kx = 2*bd.pi*bd.fft.fftshift(bd.fft.fftfreq(self.Nx, d = self.dx))
        ky = 2*bd.pi*bd.fft.fftshift(bd.fft.fftfreq(self.Ny, d = self.dy))
        kx, ky = bd.meshgrid(kx, ky)

        sRGB_linear = bd.zeros((3, self.Nx * self.Ny))

        bar = progressbar.ProgressBar()

        # We compute the pattern of each wavelength separately, and associate it to small spectrum interval dλ = (780- 380)/spectrum_divisions . We approximately the final colour
        # by summing the contribution of each small spectrum interval converting its intensity distribution to a RGB space.
        

        t0 = time.time()

        for i in bar(range(self.spectrum_divisions)):

            E_λ = self.E.copy()
            for j in range(len(self.steps)):

                if self.steps_type[j] == 'optical_element':

                    E_λ = self.steps[j].get_E(E_λ, self.xx, self.yy, self.λ_list_samples[i]* nm)

                else: #type == 'propagation'

                    propagation_index[i] += 1

                    z, scale_factor = self.steps_args[j]

                    E_λ = self.steps[j](self, E_λ, z, self.λ_list_samples[i]* nm, scale_factor)

                    if propagation_index[i] == self.number_of_propagations:
                        Iλ = bd.real(E_λ * bd.conjugate(E_λ))
                        XYZ = self.cs.spec_partition_to_XYZ(bd.outer(Iλ, self.spec_partitions[i]),i)
                        sRGB_linear += self.cs.XYZ_to_sRGB_linear(XYZ)



        if bd != np:
            bd.cuda.Stream.null.synchronize()
        rgb = self.cs.sRGB_linear_to_sRGB(sRGB_linear)
        rgb = (rgb.T).reshape((self.Ny, self.Nx, 3))
        print ("Computation Took", time.time() - t0)
        return rgb


    def scale_propagate(self, z, scale_factor):
        """
        #raise NotImplementedError(self.__class__.__name__ + '.scale_propagate')

        two_steps_fresnel_method is implemented but it cannot be used with ApertureFromImage from now if the aperture is at z != 0. For this case,
        use angular_spectrum_method instead.
        """

        self.z += z

        self.steps += [two_steps_fresnel_method]
        self.number_of_propagations += 1
        self.steps_type += ['propagation']

        scale_factor = 1
        self.steps_args += [[z, scale_factor]]


    def get_colors_at_image_plane(self, pupil, M, zi, z0, scale_factor = 1):
        """
        Assuming an optical system with linear response and assuming the system is only diffraction-limited by
        the exit pupil of the system, compute the field at its image plane

        
        Parameters
        ----------

        pupil: diffractive optical element used as exit pupil. Can be circular aperture, a diaphragm etc

        zi: distance from the image plane to the objective lens
        z0: distance from the objective lens to the current simulation plane

        M: magnification factor of the optical system
        (If the optical system is a single lens, magnification = - zi/z0)

        Reference:
        Introduction to Fourier Optics J. Goodman, Frequency Analysis of Optical Imaging Systems
        
        """



        for j in range(len(self.optical_elements)):
            self.E = self.E * self.optical_elements[j].get_transmittance(self.xx, self.yy, 0)


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

        bar = progressbar.ProgressBar()

        # We compute the pattern of each wavelength separately, and associate it to small spectrum interval dλ = (780- 380)/spectrum_divisions . We approximately the final colour
        # by summing the contribution of each small spectrum interval converting its intensity distribution to a RGB space.
        
        sRGB_linear = bd.zeros((3, self.Nx * self.Ny))

        t0 = time.time()

        for i in bar(range(self.spectrum_divisions)):
            #Definte the ATF function, representing the Fourier transform of the circular pupil function.
            H = pupil.get_amplitude_transfer_function(fxx, fyy, zi, self.λ_list_samples[i]* nm)

            E_λ = bd.fft.ifft2(bd.fft.ifftshift(c*H))

            Iλ = bd.real(E_λ * bd.conjugate(E_λ))

            XYZ = self.cs.spec_partition_to_XYZ(bd.outer(Iλ, self.spec_partitions[i]),i)
            sRGB_linear += self.cs.XYZ_to_sRGB_linear(XYZ)

        if bd != np:
            bd.cuda.Stream.null.synchronize()

        self.xx = M_abs * self.xx
        self.yy = M_abs * self.yy
        self.x = M_abs * self.x
        self.y = M_abs * self.y
        self.dx = M_abs * self.dx
        self.dy = M_abs * self.dy


        rgb = self.cs.sRGB_linear_to_sRGB(sRGB_linear)
        rgb = (rgb.T).reshape((self.Ny, self.Nx, 3))
        print ("Computation Took", time.time() - t0)
        return rgb



    from .visualization import plot_colors
