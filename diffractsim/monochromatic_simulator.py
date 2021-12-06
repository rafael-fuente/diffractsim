from . import colour_functions as cf
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from pathlib import Path
from PIL import Image
import time
import progressbar
from .util.image_handling import convert_graymap_image_to_hsvmap_image, rescale_img_to_simulation_coordinates
from .util.constants import *


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

        self.Nx = bd.int(Nx)
        self.Ny = bd.int(Ny)
        self.E = bd.ones((int(self.Ny), int(self.Nx))) * bd.sqrt(intensity)
        self.λ = wavelength
        self.z = 0
        self.cs = cf.ColourSystem(clip_method = 0)
        
    def add_rectangular_slit(self, x0, y0, width, height):
        """
        Creates a slit centered at the point (x0, y0) with width width and height height
        """
        t = bd.where((((self.xx > (x0 - width / 2)) & (self.xx < (x0 + width / 2)))
                        & ((self.yy > (y0 - height / 2)) & (self.yy < (y0 + height / 2)))),
                        bd.ones_like(self.E), bd.zeros_like(self.E))

        self.E = self.E*t

        self.I = bd.real(self.E * bd.conjugate(self.E))  

    def add_circular_slit(self, x0, y0, R):
        """
        Creates a circular slit centered at the point (x0,y0) with radius R
        """

        t = bd.select(
            [(self.xx - x0) ** 2 + (self.yy - y0) ** 2 < R ** 2, bd.ones_like(self.E, dtype=bool)], [bd.ones_like(self.E), bd.zeros_like(self.E)]
        )

        self.E = self.E*t
        self.I = bd.real(self.E * bd.conjugate(self.E))  



    def add_gaussian_beam(self, w0):
        """
        Creates a Gaussian beam with radius equal to w0
        """

        r2 = self.xx**2 + self.yy**2 
        self.E = self.E*bd.exp(-r2/(w0**2))
        self.I = bd.real(self.E * bd.conjugate(self.E))  




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
                    [bd.ones_like(self.E), bd.zeros_like(self.E)],)
                y0 -= D
            x0 += D
        self.E = self.E*t
        self.I = bd.real(self.E * bd.conjugate(self.E))  

    def add_aperture_from_function(self,function):
        """
        Evaluate a function with arguments 'x : 2D  array' and 'y : 2D array' as the amplitude transmittance of the aperture. 
        """


        t = function(self.xx, self.yy)
        self.E = self.E*bd.array(t)
        self.I = bd.real(self.E * bd.conjugate(self.E))  
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


        # compute Field Intensity
        self.I = bd.real(self.E * bd.conjugate(self.E))  

    def compute_fft(self):
        """compute the field in distance equal to z with the angular spectrum method"""


        # compute angular spectrum
        fft_c = bd.fft.fft2(self.E)
        self.E = bd.fft.fftshift(fft_c)

        # compute Field Intensity
        self.I = bd.real(self.E * bd.conjugate(self.E))  

    def add_lens(self, f, radius = None, aberration = None):
        """add a thin lens with a focal length equal to f """

        self.E = self.E * bd.exp(-1j*bd.pi/(self.λ*f) * (self.xx**2 + self.yy**2))

        if aberration != None:
            self.E = self.E*bd.exp(2*bd.pi * 1j *aberration(self.xx, self.yy))

        if radius != None:
            self.E = bd.where((self.xx**2 + self.yy**2) < radius**2, self.E, bd.zeros_like(self.E))



    def propagate(self, z):
        """compute the field in distance equal to z with the angular spectrum method"""

        self.z += z

        # compute angular spectrum
        fft_c = bd.fft.fft2(self.E)
        c = bd.fft.fftshift(fft_c)

        kx = 2*bd.pi*bd.fft.fftshift(bd.fft.fftfreq(self.Nx, d = self.dx))
        ky = 2*bd.pi*bd.fft.fftshift(bd.fft.fftfreq(self.Ny, d = self.dy))
        kx, ky = bd.meshgrid(kx, ky)

        argument = (2 * bd.pi / self.λ) ** 2 - kx ** 2 - ky ** 2

        #Calculate the propagating and the evanescent (complex) modes
        tmp = bd.sqrt(bd.abs(argument))
        kz = bd.where(argument >= 0, tmp, 1j*tmp)

        # propagate the angular spectrum a distance z
        E = bd.fft.ifft2(bd.fft.ifftshift(c * bd.exp(1j * kz * z)))
        self.E = E

        # compute Field Intensity
        self.I = bd.real(E * bd.conjugate(E))  

    def get_colors(self):
        """ compute RGB colors"""

        rgb = self.cs.wavelength_to_sRGB(self.λ / nm, 10 * self.I.flatten()).T.reshape(
            (self.Ny, self.Nx, 3)
        )
        return rgb


    def compute_colors_at(self, z):
        """propagate the field to a distance equal to z and compute the RGB colors of the beam profile profile"""

        self.propagate(z)
        rgb = self.get_colors()
        return rgb



    def add_spatial_noise(self, noise_radius, f_mean, f_size, N = 30, A = 1):
        """
        add spatial noise following a radial normal distribution

        Parameters
        ----------
        noise_radius: maximum radius affected by the spatial noise
        f_mean: mean spatial frequency of the spatial noise 
        f_size: spread spatial frequency of the noise 
        N: number of samples
        A: amplitude of the noise
        """

        def random_noise(xx,yy, f_mean,A):
            A = bd.random.rand(1)*A
            phase = bd.random.rand(1)*2*bd.pi
            fangle = bd.random.rand(1)*2*bd.pi
            f = bd.random.normal(f_mean, f_size/2)

            fx = f*bd.cos(fangle) 
            fy = f*bd.sin(fangle) 
            return A*bd.exp((xx**2 + yy**2)/ (noise_radius*2)**2)*bd.sin(2*bd.pi*fx*xx + 2*bd.pi*fy*yy + phase)

        E_noise = 0
        for i in range(0,N):
            E_noise += random_noise(self.xx,self.yy,f_mean,A)/bd.sqrt(N)

        self.E += E_noise *bd.exp(-(self.xx**2 + self.yy**2)/ (noise_radius)**2)
        self.I = bd.real(self.E * bd.conjugate(self.E)) 


    def get_longitudinal_profile(self, start_distance, end_distance, steps):
        """
        Propagates the field at n steps equally spaced between start_distance and end_distance, and returns
        the colors and the field over the xz plane
        """

        z = bd.linspace(start_distance, end_distance, steps)

        self.E0 = bd.copy(self.E)

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
