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
        global backend_name
        from .util.backend_functions import backend as bd
        from .util.backend_functions import backend_name

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
        KX, KY = bd.meshgrid(kx, ky)

        E_x = bd.zeros((self.Ny, self.Nx), dtype=complex)
        E_y = bd.zeros((self.Ny, self.Nx), dtype=complex)

        for λ in self.λ_list_samples:
            H = 1j * (KX**2 + KY**2)**0.5
            E_x += self.spectrum[λ] * np.exp(-1j * KX * self.x[:, None]) * np.exp(-1j * KY * self.y)
            E_y += self.spectrum[λ] * np.exp(-1j * KX * self.x[:, None]) * np.exp(-1j * KY * self.y)

        B_x = bd.gradient(E_y, axis=0)
        B_y = -bd.gradient(E_x, axis=1)

        return E_x, E_y, B_x, B_y


    def get_energy_density(self):

        E_x, E_y, _, _ = self.get_colors()
        energy_density = np.abs(E_x)**2 + np.abs(E_y)**2
        return energy_density