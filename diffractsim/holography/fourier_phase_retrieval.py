import numpy as np
from ..util.file_handling import load_graymap_image_as_array, save_phase_mask_as_image
from ..util.image_handling import resize_array

from ..util.backend_functions import backend as bd

class FourierPhaseRetrieval():
    def __init__(self, target_amplitude_path, source_amplitude_path = None, new_size = None, pad = None):
        "class for retrieve the phase mask required to reconstruct an image (specified at target amplitude path) at the Fourier plane"

        global bd
        from ..util.backend_functions import backend as bd

        self.target_amplitude = np.array(load_graymap_image_as_array(target_amplitude_path, new_size = new_size))
        
        if pad != None:
            self.target_amplitude = np.pad(self.target_amplitude, ((pad, pad), (pad, pad)), "constant")

        self.Nx = self.target_amplitude.shape[1]
        self.Ny = self.target_amplitude.shape[0]
                
        if source_amplitude_path != None:
            self.source_amplitude = np.array(load_graymap_image_as_array(source_amplitude_path, new_size = (self.Nx, self.Ny)))
        else:
            self.source_amplitude = np.ones((self.Ny, self.Nx))

        
        self.retrieved_phase = None


    def retrieve_phase_mask(self, max_iter = 200, method = 'Gerchberg-Saxton'):
        
        implemented_methods = ('Gerchberg-Saxton')

        if method == 'Gerchberg-Saxton':


            # a padding of the source_amplitude will improve image reconstruction quality, while mantaining the phase mask hologram with the same size
            target_amplitude = bd.array(resize_array(self.target_amplitude, (self.Ny + 2 * self.Ny//2 , self.Nx + 2 * self.Nx//2)))
            source_amplitude = bd.pad(bd.array(self.source_amplitude), ((self.Ny//2, self.Ny//2), (self.Nx//2, self.Nx//2)), "constant")

            # Gerchberg Saxton iteration
            target_amplitude  = bd.fft.fftshift(target_amplitude)
            A = bd.fft.ifft2(target_amplitude)

            for iter in range(max_iter):
                B = bd.abs(source_amplitude) * bd.exp(1j * bd.angle(A))
                C = bd.fft.fft2(B)
                D = bd.abs(target_amplitude) * bd.exp(1j * bd.angle(C))
                A = bd.fft.ifft2(D)
                
            self.retrieved_phase = bd.angle(A)

            # undo padding
            self.retrieved_phase = self.retrieved_phase[self.Ny//2:-self.Ny//2, self.Nx//2:-self.Nx//2]



        else:
            raise NotImplementedError(
                f"{method} has not been implemented. Use one of {implemented_method}")




        
    def save_retrieved_phase_as_image(self, name):


        if bd == np:
            save_phase_mask_as_image(name, self.retrieved_phase)
        else:
            save_phase_mask_as_image(name, self.retrieved_phase.get())

            

            
    def set_source_amplitude_from_function(self, function, extent_x, extent_y):
        
        x = extent_x*(bd.arange(self.Nx)-self.Nx//2)/self.Nx
        y = extent_y*(bd.arange(self.Ny)-self.Ny//2)/self.Ny
        xx, yy = bd.meshgrid(x, y)
        self.source_amplitude = function(xx, yy)


