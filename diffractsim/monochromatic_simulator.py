from . import colour_functions as cf
import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
from scipy.interpolate import interp2d
from pathlib import Path
from PIL import Image


m = 1.
cm = 1e-2
mm = 1e-3
nm = 1e-9


class MonochromaticField:
    def __init__(self, wavelength, extent_x, extent_y, Nx, Ny):

        self.extent_x = extent_x
        self.extent_y = extent_y

        self.x = np.linspace(-extent_x / 2, extent_x / 2, Nx)
        self.y = np.linspace(-extent_y / 2, extent_y / 2, Ny)
        self.xx, self.yy = np.meshgrid(self.x, self.y)

        self.Nx = np.int(Nx)
        self.Ny = np.int(Ny)
        self.E = np.ones((int(self.Ny), int(self.Nx)))
        self.λ = wavelength

    def add_rectangular_slit(self, x0, y0, width, height):
        """
        Creates a slit centered at the point (x0, y0) with width width and height height
        """
        self.E = np.select(
            [
                ((self.xx > (x0 - width / 2)) & (self.xx < (x0 + width / 2)))
                & ((self.yy > (y0 - height / 2)) & (self.yy < (y0 + height / 2))),
                True,
            ],
            [self.E, 0],
        )

    def add_circular_slit(self, x0, y0, R):
        """
        Creates a circular slit centered at the point (x0,y0) with radius R
        """

        self.E = np.select(
            [(self.xx - x0) ** 2 + (self.yy - y0) ** 2 < R ** 2, True], [self.E, 0]
        )

    def add_diffraction_grid(self, D, a, Nx, Ny):
        """
        Creates a diffraction_grid with Nx *  Ny slits with separation distance D and width a
        """

        E0 = np.copy(self.E)
        Ef = 0

        b = D - a
        width, height = Nx * a + (Nx - 1) * b, Ny * a + (Ny - 1) * b
        x0, y0 = -width / 2, height / 2

        x0 = -width / 2 + a / 2
        for _ in range(Nx):
            y0 = height / 2 - a / 2
            for _ in range(Ny):

                Ef += np.select(
                    [
                        ((self.xx > (x0 - a / 2)) & (self.xx < (x0 + a / 2)))
                        & ((self.yy > (y0 - a / 2)) & (self.yy < (y0 + a / 2))),
                        True,
                    ],
                    [E0, 0],
                )
                y0 -= D
            x0 += D
        self.E = Ef

    def add_aperture_from_image(self, path, pad=None, Nx=None, Ny=None):
        # This function load the image specified at "path" as a numpy graymap array.
        # If Nx and Ny is specified, we interpolate the pattern with interp2d method to the new specified resolution.
        # If pad is specified, we add zeros (black color) padded to the edges of each axis.

        img = Image.open(Path(path))
        img = img.convert("RGB")
        imgRGB = np.asarray(img) / 256.0
        imgR = imgRGB[:, :, 0]
        imgG = imgRGB[:, :, 1]
        imgB = imgRGB[:, :, 2]
        self.E = 0.2989 * imgR + 0.5870 * imgG + 0.1140 * imgB

        fun = interp2d(
            np.linspace(0, 1, self.E.shape[1]),
            np.linspace(0, 1, self.E.shape[0]),
            self.E,
            kind="cubic",
        )
        self.E = fun(np.linspace(0, 1, self.Nx), np.linspace(0, 1, self.Ny))

        # optional: add zeros and interpolate to the new specified resolution
        if pad != None:

            Nxpad = int(np.round(self.Nx / self.extent_x * pad[0]))
            Nypad = int(np.round(self.Ny / self.extent_y * pad[1]))
            self.E = np.pad(self.E, ((Nypad, Nypad), (Nxpad, Nxpad)), "constant")

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
            self.E = fun(np.linspace(0, 1, self.Nx), np.linspace(0, 1, self.Ny))

        # grid units
        self.x = np.linspace(-self.extent_x / 2, self.extent_x / 2, self.Nx)
        self.y = np.linspace(-self.extent_y / 2, self.extent_y / 2, self.Ny)
        self.xx, self.yy = np.meshgrid(self.x, self.y)

    def compute_colors_at(self, z):
        self.z = z

        # compute angular spectrum
        fft_c = fft2(self.E)
        c = fftshift(fft_c)

        kx = np.linspace(
            -np.pi * self.Nx // 2 / (self.extent_x / 2),
            np.pi * self.Nx // 2 / (self.extent_x / 2),
            self.Nx,
        )
        ky = np.linspace(
            -np.pi * self.Ny // 2 / (self.extent_y / 2),
            np.pi * self.Ny // 2 / (self.extent_y / 2),
            self.Ny,
        )
        kx, ky = np.meshgrid(kx, ky)
        kz = np.sqrt((2 * np.pi / self.λ) ** 2 - kx ** 2 - ky ** 2)

        # propagate the angular spectrum a distance z
        E = ifft2(ifftshift(c * np.exp(1j * kz * z)))

        # compute Field Intensity
        self.I = np.real(E * np.conjugate(E)) * 0.1

        # compute RGB colors
        rgb = cf.wavelength_to_sRGB(self.λ / nm, 10 * self.I.flatten()).T.reshape(
            (self.Ny, self.Nx, 3)
        )
        return rgb

    def plot(self, rgb, figsize=(6, 6), xlim=None, ylim=None):
        plt.style.use("dark_background")

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
