from . import colour_functions as cf
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from pathlib import Path
from PIL import Image

import numpy as np
from .backend_functions import backend as bd


m = 1.
cm = 1e-2
mm = 1e-3
um = 1e-6
nm = 1e-9


class MonochromaticField:
    def __init__(self,  wavelength, extent_x, extent_y, Nx, Ny, intensity = 0.1):
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
        from .backend_functions import backend as bd

        self.extent_x = extent_x
        self.extent_y = extent_y

        self.x = bd.linspace(-extent_x / 2, extent_x / 2, Nx)
        self.y = bd.linspace(-extent_y / 2, extent_y / 2, Ny)
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
        t = bd.select(
            [
                ((self.xx > (x0 - width / 2)) & (self.xx < (x0 + width / 2)))
                & ((self.yy > (y0 - height / 2)) & (self.yy < (y0 + height / 2))),
                True,
            ],
            [bd.ones(self.E.shape), bd.zeros(self.E.shape)],
        )
        self.E = self.E*t

        self.I = bd.real(self.E * bd.conjugate(self.E))  

    def add_circular_slit(self, x0, y0, R):
        """
        Creates a circular slit centered at the point (x0,y0) with radius R
        """

        t = bd.select(
            [(self.xx - x0) ** 2 + (self.yy - y0) ** 2 < R ** 2, bd.full(self.E.shape, True, dtype=bool)], [bd.ones(self.E.shape), bd.zeros(self.E.shape)]
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
                        True,
                    ],
                    [bd.ones(self.E.shape), bd.zeros(self.E.shape)],
                )
                y0 -= D
            x0 += D
        self.E = self.E*t
        self.I = bd.real(self.E * bd.conjugate(self.E))  



    def add_aperture_from_image(self, path, pad=None, Nx=None, Ny=None):
        """
        Load the image specified at "path" as a numpy graymap array.
        - If Nx and Ny is specified, we interpolate the pattern with interp2d method to the new specified resolution.
        - If pad is specified, we add zeros (black color) padded to the edges of each axis.
        """


        img = Image.open(Path(path))
        img = img.convert("RGB")
        imgRGB = np.asarray(img) / 256.0
        imgR = imgRGB[:, :, 0]
        imgG = imgRGB[:, :, 1]
        imgB = imgRGB[:, :, 2]
        t = 0.2989 * imgR + 0.5870 * imgG + 0.1140 * imgB

        fun = interp2d(
            np.linspace(0, 1, t.shape[1]),
            np.linspace(0, 1, t.shape[0]),
            t,
            kind="cubic",
        )
        t = fun(np.linspace(0, 1, self.Nx), np.linspace(0, 1, self.Ny))

        # optional: add zeros and interpolate to the new specified resolution
        if pad != None:

            if bd != np:
                self.E = self.E.get()

            Nxpad = int(np.round(self.Nx / self.extent_x * pad[0]))
            Nypad = int(np.round(self.Ny / self.extent_y * pad[1]))
            self.E = np.pad(self.E, ((Nypad, Nypad), (Nxpad, Nxpad)), "constant")
            t = np.pad(t, ((Nypad, Nypad), (Nxpad, Nxpad)), "constant")
            self.E = np.array(self.E*t)

            scale_ratio = self.E.shape[1] / self.E.shape[0]
            self.Nx = int(np.round(self.E.shape[0] * scale_ratio)) if Nx is None else Nx
            self.Ny = self.E.shape[0] if Ny is None else Ny
            self.extent_x += 2 * pad[0]
            self.extent_y += 2 * pad[1]

            fun = interp2d(
                np.linspace(0, 1, self.E.shape[1]),
                np.linspace(0, 1, self.E.shape[0]),
                self.E,
                kind="cubic",
            )
            self.E = bd.array(fun(np.linspace(0, 1, self.Nx), np.linspace(0, 1, self.Ny)))

            # new grid units
            self.x = bd.linspace(-self.extent_x / 2, self.extent_x / 2, self.Nx)
            self.y = bd.linspace(-self.extent_y / 2, self.extent_y / 2, self.Ny)
            self.xx, self.yy = bd.meshgrid(self.x, self.y)  

        else:
            self.E = self.E*bd.array(t)


        # compute Field Intensity
        self.I = bd.real(self.E * bd.conjugate(self.E))  


    def add_lens(self, f):
        """add a thin lens with a focal length equal to f """
        self.E = self.E * bd.exp(-1j*bd.pi/(self.λ*f) * (self.xx**2 + self.yy**2))



    def propagate(self, z):
        """compute the field in distance equal to z with the angular spectrum method"""

        self.z += z

        # compute angular spectrum
        fft_c = bd.fft.fft2(self.E)
        c = bd.fft.fftshift(fft_c)

        kx = bd.linspace(
            -bd.pi * self.Nx // 2 / (self.extent_x / 2),
            bd.pi * self.Nx // 2 / (self.extent_x / 2),
            self.Nx,
        )
        ky = bd.linspace(
            -bd.pi * self.Ny // 2 / (self.extent_y / 2),
            bd.pi * self.Ny // 2 / (self.extent_y / 2),
            self.Ny,
        )
        kx, ky = bd.meshgrid(kx, ky)
        kz = bd.sqrt((2 * bd.pi / self.λ) ** 2 - kx ** 2 - ky ** 2)

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

    def plot(self, rgb, figsize=(6, 6), xlim=None, ylim=None):
        """visualize the diffraction pattern with matplotlib"""

        plt.style.use("dark_background")
        if bd != np:
            rgb = rgb.get()

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1)

        if xlim != None:
            ax.set_xlim(xlim)

        if ylim != None:
            ax.set_ylim(ylim)

        # we use mm by default
        ax.set_xlabel("[mm]")
        ax.set_ylabel("[mm]")

        ax.set_title("Screen distance = " + str(self.z * 100) + " cm")

        im = ax.imshow(
            (rgb),
            extent=[
                -self.extent_x / 2 / mm,
                self.extent_x / 2 / mm,
                -self.extent_y / 2 / mm,
                self.extent_y / 2 / mm,
            ],
            interpolation="spline36",
        )
        plt.show()


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
