from . import colour_functions as cf
import matplotlib.pyplot as plt
import progressbar
from scipy.interpolate import interp2d
from pathlib import Path
from PIL import Image
import time
from .util.image_handling import convert_graymap_image_to_hsvmap_image, rescale_img_to_simulation_coordinates

import numpy as np
from .util.backend_functions import backend as bd
from .util.constants import *




class PolychromaticField:
    def __init__(self, spectrum, extent_x, extent_y, Nx, Ny, spectrum_size = 180, spectrum_divisions = 30):
        global bd
        from .util.backend_functions import backend as bd

        self.extent_x = extent_x
        self.extent_y = extent_y

        self.dx = self.extent_x/Nx
        self.dy = self.extent_y/Ny

        self.x = self.dx*(bd.arange(Nx)-Nx//2)
        self.y = self.dy*(bd.arange(Ny)-Ny//2)
        self.xx, self.yy = bd.meshgrid(self.x, self.y)

        self.Nx = bd.int(Nx)
        self.Ny = bd.int(Ny)
        self.E = bd.ones((int(self.Ny), int(self.Nx)))

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


        self.lens = False
        self.lens_f = 0.
        self.z = 0


    def add_rectangular_slit(self, x0, y0, width, height):
        """
        Creates a slit centered at the point (x0, y0) with width width and height height
        """
        t = bd.where((((self.xx > (x0 - width / 2)) & (self.xx < (x0 + width / 2)))
                        & ((self.yy > (y0 - height / 2)) & (self.yy < (y0 + height / 2)))),
                        bd.ones_like(self.E), bd.zeros_like(self.E))
        
        self.E = self.E*t

    def add_circular_slit(self, x0, y0, R):
        """
        Creates a circular slit centered at the point (x0,y0) with radius R
        """

        t = bd.select(
            [(self.xx - x0) ** 2 + (self.yy - y0) ** 2 < R ** 2, bd.ones_like(self.E, dtype=bool)], [bd.ones_like(self.E), bd.zeros_like(self.E)]
        )

        self.E = self.E*t



    def add_gaussian_beam(self, w0):
        """
        Creates a Gaussian beam with radius equal to w0
        """

        r2 = self.xx**2 + self.yy**2 
        self.E = self.E*bd.exp(-r2/(w0**2))



    def add_diffraction_grid(self, D, a, Nx, Ny):
        """
        Creates a diffraction_grid with Nx *  Ny slits with separation distance D and width a
        """

        E0 = bd.copy(self.E)
        t = 0

        b = D - a
        width, height = Nx * a + (Nx - 1) * b, Ny * a + (Ny - 1) * b
        x0, y0 = -width / 2, height / 2

        x0 = -width / 2 + a / 2
        for _ in range(Nx):
            y0 = height / 2 - a / 2
            for _ in range(Ny):

                t += bd.select(
                    [
                        ((self.xx > (x0 - a / 2)) & (self.xx < (x0 + a / 2)))
                        & ((self.yy > (y0 - a / 2)) & (self.yy < (y0 + a / 2))),
                        bd.ones_like(self.E, dtype=bool),
                    ],
                    [bd.ones_like(self.E), bd.zeros_like(self.E)])
                y0 -= D
            x0 += D
        self.E = self.E*t

    def add_aperture_from_function(self,function):
        """
        Evaluate a function with arguments 'x : 2D  array' and 'y : 2D array' as the amplitude transmittance of the aperture. 
        """

        t = function(self.xx, self.yy)
        self.E = self.E*bd.array(t)
        return t


    def add_aperture_from_image(self, amplitude_mask_path= None, phase_mask_path= None, image_size = None, phase_mask_format = 'hsv'):
        """
        Load the image specified at "amplitude_mask_path" as a numpy graymap array represeting the amplitude transmittance of the aperture. 
        The image is centered on the plane and its physical size is specified in image_size parameter as image_size = (float, float)

        - If image_size isn't specified, the image fills the entire aperture plane
        """

        if amplitude_mask_path != None:
            
            #load the amplitude_mask image
            img = Image.open(Path(amplitude_mask_path))
            img = img.convert("RGB")

            rescaled_img = rescale_img_to_simulation_coordinates(self, img, image_size)
            imgRGB = np.asarray(rescaled_img) / 255.0

            t = 0.2990 * imgRGB[:, :, 0] + 0.5870 * imgRGB[:, :, 1] + 0.1140 * imgRGB[:, :, 2]
            t = bd.array(np.flip(t, axis = 0))

            self.E = self.E*t


        if phase_mask_path != None:
            from matplotlib.colors import rgb_to_hsv

            #load the phase_mask image
            img = Image.open(Path(phase_mask_path))
            img = img.convert("RGB")

            if phase_mask_format == 'graymap':
                img = convert_graymap_image_to_hsvmap_image(img)
                
            rescaled_img = rescale_img_to_simulation_coordinates(self, img, image_size)
            imgRGB = np.asarray(rescaled_img) / 255.0


            h = rgb_to_hsv(   np.moveaxis(np.array([imgRGB[:, :, 0],imgRGB[:, :, 1],imgRGB[:, :, 2]]) , 0, -1))[:,:,0]
            phase_mask = bd.flip(bd.array(h) * 2 * bd.pi - bd.pi, axis = 0)
            self.E = self.E*bd.exp(1j *  phase_mask)




    def add_lens(self, f,radius = None, aberration = None):
        """add a thin lens with a focal length equal to f """

        self.lens = True
        self.lens_f = f

        #lens amplitude transmittance
        self.lens_t = 1
        if aberration != None:
            self.lens_t = self.lens_t*bd.exp(2*bd.pi * 1j *aberration(self.xx, self.yy))

        if radius != None:
            self.lens_t = bd.where((self.xx**2 + self.yy**2) < radius**2, self.lens_t, bd.zeros_like(self.E))






    def compute_colors_at(self, z):
        """propagate the field to a distance equal to z and compute the RGB colors of the beam profile profile"""
        t0 = time.time()
        self.z = z


        kx = 2*bd.pi*bd.fft.fftshift(bd.fft.fftfreq(self.Nx, d = self.dx))
        ky = 2*bd.pi*bd.fft.fftshift(bd.fft.fftfreq(self.Ny, d = self.dy))
        kx, ky = bd.meshgrid(kx, ky)

        sRGB_linear = bd.zeros((3, self.Nx * self.Ny))

        

        if self.lens == False:
            fft_c = bd.fft.fft2(self.E)
            c = bd.fft.fftshift(fft_c)
            # if not is computed in the loop



        bar = progressbar.ProgressBar()

        # We compute the pattern of each wavelength separately, and associate it to small spectrum interval dλ = (780- 380)/spectrum_divisions . We approximately the final colour
        # by summing the contribution of each small spectrum interval converting its intensity distribution to a RGB space.
        

        t0 = time.time()

        for i in bar(range(self.spectrum_divisions)):
            if self.lens == True:
                fft_c = bd.fft.fft2(self.E * bd.exp(-1j*bd.pi/(self.λ_list_samples[i]* nm * self.lens_f) * (self.xx**2 + self.yy**2))  * self.lens_t )
                c = bd.fft.fftshift(fft_c)
                # if not is computed in the loop


            argument = (2 * bd.pi / (self.λ_list_samples[i]* nm)) ** 2 - kx ** 2 - ky ** 2

            #Calculate the propagating and the evanescent (complex) modes
            tmp = bd.sqrt(bd.abs(argument))
            kz = bd.where(argument >= 0, tmp, 1j*tmp)

            E_λ = bd.fft.ifft2(bd.fft.ifftshift(c * bd.exp(1j * kz * z)))
            Iλ = bd.real(E_λ * bd.conjugate(E_λ))

            XYZ = self.cs.spec_partition_to_XYZ(bd.outer(Iλ, self.spec_partitions[i]),i)
            sRGB_linear += self.cs.XYZ_to_sRGB_linear(XYZ)

        if bd != np:
            bd.cuda.Stream.null.synchronize()
        rgb = self.cs.sRGB_linear_to_sRGB(sRGB_linear)
        rgb = (rgb.T).reshape((self.Ny, self.Nx, 3))
        print ("Computation Took", time.time() - t0)
        return rgb



    def propagate(self, z, spectrum_divisions=40, grid_divisions=10):

        raise NotImplementedError(self.__class__.__name__ + '.propagate')

    def get_colors(self):

        raise NotImplementedError(self.__class__.__name__ + '.get_colors')

    def add_spatial_noise(self, noise_radius, f_mean, f_size, N = 30, A = 1):

        raise NotImplementedError(self.__class__.__name__ + '.add_spatial_noise')


    from .visualization import plot_colors
