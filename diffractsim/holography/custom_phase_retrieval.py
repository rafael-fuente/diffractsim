import numpy as np
from ..util.backend_functions import backend as bd

from ..util.file_handling import load_graymap_image_as_array, save_phase_mask_as_image
from ..util.image_handling import rescale_img_to_custom_coordinates
from ..monochromatic_simulator import MonochromaticField
from pathlib import Path
from PIL import Image
from ..util.constants import *
import progressbar

class CustomPhaseRetrieval():
    def __init__(self, wavelength, extent_x, extent_y, Nx, Ny):
        "class for retrieve the phase mask required to reconstruct an image (specified at target amplitude path) at an arbitrary diffraction plane"

        global bd
        from ..util.backend_functions import backend as bd


        self.extent_x = extent_x
        self.extent_y = extent_y

        self.dx = self.extent_x/Nx
        self.dy = self.extent_y/Ny

        self.x = self.dx*(bd.arange(Nx)-Nx//2)
        self.y = self.dy*(bd.arange(Ny)-Ny//2)
        self.xx, self.yy = bd.meshgrid(self.x, self.y)

        self.Nx = Nx
        self.Ny = Ny
        self.E = bd.ones((self.Ny, self.Nx))
        self.wavelength = wavelength



    def set_source_amplitude(self, amplitude_mask_path, image_size = None):
        
        #load the amplitude_mask image
        img = Image.open(Path(amplitude_mask_path))
        img = img.convert("RGB")

        rescaled_img = rescale_img_to_custom_coordinates(img, image_size, self.extent_x, self.extent_y, self.Nx, self.Ny)
        imgRGB = np.asarray(rescaled_img) / 255.0

        t = 0.2990 * imgRGB[:, :, 0] + 0.5870 * imgRGB[:, :, 1] + 0.1140 * imgRGB[:, :, 2]
        t = bd.array(np.flip(t, axis = 0))

        self.source_amplitude = t



    def set_target_amplitude(self, amplitude_mask_path, image_size = None):
        
        #load the amplitude_mask image
        img = Image.open(Path(amplitude_mask_path))
        img = img.convert("RGB")

        rescaled_img = rescale_img_to_custom_coordinates(img, image_size, self.extent_x, self.extent_y, self.Nx, self.Ny)
        imgRGB = np.asarray(rescaled_img) / 255.0

        t = 0.2990 * imgRGB[:, :, 0] + 0.5870 * imgRGB[:, :, 1] + 0.1140 * imgRGB[:, :, 2]
        t = bd.array(np.flip(t, axis = 0))

        self.target_amplitude = t

    def set_field_propagate_function(self, propagate_function, inverse_function):

        """
            for example, the following functions can be used for reconstructing an image at the Fourier plane

            def propagate_function(F):
                F.add_lens(f = 80*cm)
                F.propagate(80*cm)


            def inverse_function(F):
                F.propagate(-80*cm)
                F.add_lens(f = -80*cm)
        """

        self.propagate_function = propagate_function
        self.inverse_function = inverse_function

    def retrieve_phase_mask(self, max_iter = 200, method = 'Gerchberg-Saxton'):
        
        implemented_methods = ('Gerchberg-Saxton')

        if method == 'Gerchberg-Saxton':

            F = MonochromaticField(
                wavelength=self.wavelength, extent_x=self.extent_x, extent_y=self.extent_y, Nx=self.Nx, Ny=self.Ny
            )

            # Gerchberg Saxton iteration
            F.E = bd.fft.ifft2(bd.fft.ifftshift(self.target_amplitude))

            bar = progressbar.ProgressBar()
            for iter in bar(range(max_iter)):

                F.E = bd.abs(self.source_amplitude) * bd.exp(1j * bd.angle(F.E))
                self.propagate_function(F)
                F.E = bd.abs(self.target_amplitude) * bd.exp(1j * bd.angle(F.E))
                self.inverse_function(F)

            self.retrieved_phase = bd.angle(F.E)



        else:
            raise NotImplementedError(
                f"{method} has not been implemented. Use one of {implemented_method}")




        
    def save_retrieved_phase_as_image(self, name):

        if bd == np:
            save_phase_mask_as_image(name, self.retrieved_phase)
        else:
            save_phase_mask_as_image(name, self.retrieved_phase.get())
