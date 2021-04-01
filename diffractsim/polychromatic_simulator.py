from . import colour_functions as cf
import matplotlib.pyplot as plt
import progressbar
from scipy.interpolate import interp2d
from pathlib import Path
from PIL import Image
import time

import numpy as np
from .backend_functions import backend as bd

m = 1.
cm = 1e-2
mm = 1e-3
um = 1e-6
nm = 1e-9




class PolychromaticField:
    def __init__(self, spectrum, extent_x, extent_y, Nx, Ny, spectrum_size = 180, spectrum_divisions = 30):
        global bd
        from .backend_functions import backend as bd

        self.extent_x = extent_x
        self.extent_y = extent_y

        self.x = bd.linspace(-extent_x / 2, extent_x / 2, Nx)
        self.y = bd.linspace(-extent_y / 2, extent_y / 2, Ny)
        self.xx, self.yy = bd.meshgrid(self.x, self.y)

        self.Nx = bd.int(Nx)
        self.Ny = bd.int(Ny)
        self.E = bd.ones((int(self.Ny), int(self.Nx)))

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
        t = bd.select(
            [
                ((self.xx > (x0 - width / 2)) & (self.xx < (x0 + width / 2)))
                & ((self.yy > (y0 - height / 2)) & (self.yy < (y0 + height / 2))),
                True,
            ],
            [bd.ones(self.E.shape), bd.zeros(self.E.shape)],
        )
        self.E = self.E*t

    def add_circular_slit(self, x0, y0, R):
        """
        Creates a circular slit centered at the point (x0,y0) with radius R
        """

        t = bd.select(
            [(self.xx - x0) ** 2 + (self.yy - y0) ** 2 < R ** 2, bd.full(self.E.shape, True, dtype=bool)], [bd.ones(self.E.shape), bd.zeros(self.E.shape)]
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
                        True,
                    ],
                    [bd.ones(self.E.shape), bd.zeros(self.E.shape)],
                )
                y0 -= D
            x0 += D
        self.E = self.E*t



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

    def add_lens(self, f):
        """add a thin lens with a focal length equal to f """
        self.lens = True
        self.lens_f = f


    def compute_colors_at(self, z):
        """propagate the field to a distance equal to z and compute the RGB colors of the beam profile profile"""
        t0 = time.time()
        self.z = z

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
                fft_c = bd.fft.fft2(self.E * bd.exp(-1j*bd.pi/(self.λ_list_samples[i]* nm * self.lens_f) * (self.xx**2 + self.yy**2)))
                c = bd.fft.fftshift(fft_c)
                # if not is computed in the loop


            kz = bd.sqrt(
                (2 * bd.pi / (self.λ_list_samples[i] * nm)) ** 2 - kx ** 2 - ky ** 2
            )

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
        ax.set_xlabel("[mm]")
        ax.set_ylabel("[mm]")

        ax.set_title("Screen distance = " + str(self.z * 100) + " cm")
        ax.set_aspect("equal")

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


    def propagate(self, z, spectrum_divisions=40, grid_divisions=10):

        raise NotImplementedError(self.__class__.__name__ + '.propagate')

    def get_colors(self):

        raise NotImplementedError(self.__class__.__name__ + '.get_colors')

    def add_spatial_noise(self, noise_radius, f_mean, f_size, N = 30, A = 1):

        raise NotImplementedError(self.__class__.__name__ + '.add_spatial_noise')