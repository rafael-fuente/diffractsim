import numpy as np
from ..util.file_handling import load_graymap_image_as_array, save_phase_mask_as_image
from ..util.image_handling import resize_array
from ..util.bluestein_FFT import bluestein_fft2, bluestein_ifft2, bluestein_fftfreq

from ..util.backend_functions import backend as bd
import progressbar


"""

MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.


Reference for the phase retrieval algorithms: 
J. R. Fienup, "Phase retrieval algorithms: a comparison," Appl. Opt. 21, 2758-2769 (1982)
https://www.osapublishing.org/ao/fulltext.cfm?uri=ao-21-15-2758&id=26002

In this implementation, we use the same the notation of the Fineup article.
"""

class FourierPhaseRetrieval():
    def __init__(self, target_amplitude_path, source_amplitude_path = None, new_size = None, pad = None):
        "class for retrieve the phase mask required to reconstruct an image (specified at target amplitude path) at the Fourier plane"

        global bd
        global backend_name
        from ..util.backend_functions import backend as bd
        from ..util.backend_functions import backend_name

        self.target_amplitude = np.array(load_graymap_image_as_array(target_amplitude_path, new_size = new_size))
        
        if pad != None:
            self.target_amplitude = np.pad(self.target_amplitude, ((pad[1], pad[1]), (pad[0], pad[0])), "constant")

        self.Nx = self.target_amplitude.shape[1]
        self.Ny = self.target_amplitude.shape[0]
                
        if source_amplitude_path != None:
            self.source_amplitude = np.array(load_graymap_image_as_array(source_amplitude_path, new_size = (self.Nx, self.Ny)))
        else:
            self.source_amplitude = np.ones((self.Ny, self.Nx))

        
        self.retrieved_phase = None


    def retrieve_phase_mask(self, max_iter = 200, method = 'Conjugate-Gradient', CG_step = 1.):
        
        implemented_methods = ('Gerchberg-Saxton', 'Conjugate-Gradient')

        bar = progressbar.ProgressBar()
        if method == 'Gerchberg-Saxton':


            # a padding of the source_amplitude will improve image reconstruction quality, while mantaining the phase mask hologram with the same size
            target_amplitude = bd.array(resize_array(self.target_amplitude, (self.Ny + 2 * self.Ny//2 , self.Nx + 2 * self.Nx//2)))
            source_amplitude = bd.pad(bd.array(self.source_amplitude), ((self.Ny//2, self.Ny//2), (self.Nx//2, self.Nx//2)), "constant")

            # Gerchberg Saxton iteration
            target_amplitude  = bd.abs(bd.fft.ifftshift(target_amplitude))
            source_amplitude  = bd.abs(bd.fft.ifftshift(source_amplitude))
            g_p = bd.fft.ifft2(bd.fft.ifftshift(target_amplitude))

            for iter in bar(range(max_iter)):
                g = source_amplitude * bd.exp(1j * bd.angle(g_p))
                G = bd.fft.fft2(g)
                G_p = target_amplitude * bd.exp(1j * bd.angle(G))
                g_p = bd.fft.ifft2(G_p)

                # compute the squared error to test the performance:
                # diff = bd.abs(G)/bd.sum(bd.abs(G)) - target_amplitude/bd.sum(target_amplitude)
                # squared_err = (bd.sum(diff**2))
                # print(squared_err)

            self.retrieved_phase = bd.fft.fftshift(bd.angle(g_p))

            # undo padding
            self.retrieved_phase = self.retrieved_phase[self.Ny//2:-self.Ny//2, self.Nx//2:-self.Nx//2]


        elif method == 'Conjugate-Gradient':


            # a padding of the source_amplitude will improve image reconstruction quality, while mantaining the phase mask hologram with the same size
            target_amplitude = bd.array(resize_array(self.target_amplitude, (self.Ny + 2 * self.Ny//2 , self.Nx + 2 * self.Nx//2)))
            source_amplitude = bd.pad(bd.array(self.source_amplitude), ((self.Ny//2, self.Ny//2), (self.Nx//2, self.Nx//2)), "constant")

            target_amplitude  = bd.abs(bd.fft.ifftshift(target_amplitude))
            source_amplitude  = bd.abs(bd.fft.ifftshift(source_amplitude))
            g_pp = bd.fft.ifft2(bd.fft.ifftshift(target_amplitude))

            g = bd.abs(source_amplitude) * bd.exp(1j * bd.angle(g_pp))
            gp_last_iter = g


            bar = progressbar.ProgressBar()
            for iter in bar(range(max_iter)):
                
                G = bd.fft.fft2(g)
                G_p = target_amplitude * bd.exp(1j * bd.angle(G))
                g_p = bd.fft.ifft2(G_p)

                # compute the squared error to test the performance:
                # diff = bd.abs(G)/bd.sum(bd.abs(G)) - target_amplitude/bd.sum(target_amplitude)
                # squared_err = (bd.sum(diff**2))
                # print(squared_err)

            self.retrieved_phase = bd.fft.fftshift(bd.angle(g_p))

            # undo padding
            self.retrieved_phase = self.retrieved_phase[self.Ny//2:-self.Ny//2, self.Nx//2:-self.Nx//2]


        else:
            raise ValueError(f"Unsupported method: {method}")

    def save_phase_mask(self, output_path):
        np.save(output_path, self.retrieved_phase)

    def get_phase_mask(self):
        return self.retrieved_phase