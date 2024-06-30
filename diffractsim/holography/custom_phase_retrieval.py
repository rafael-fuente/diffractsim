import numpy as np
from ..util.backend_functions import backend as jnp

from ..util.file_handling import load_graymap_image_as_array, save_phase_mask_as_image
from ..util.image_handling import rescale_img_to_custom_coordinates
from ..util.file_handling import load_graymap_image_as_array, save_phase_mask_as_image
from ..util.image_handling import rescale_img_to_custom_coordinates
from ..monochromatic_simulator import MonochromaticField

from pathlib import Path
from PIL import Image
from diffractsim.util.constants import *
import progressbar




"""

MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.

"""


# implemented custom phase retrieval methods: ('Stochastic-Gradient-Descent', 'Adam-Optimizer')
# CustomPhaseRetrieval requires JAX

class CustomPhaseRetrieval():
    def __init__(self, wavelength, z, extent_x, extent_y, Nx, Ny):
        "class for retrieve the phase mask required to reconstruct an image (specified at target amplitude path) at a distance z"

        self.z = z
        self.Nx = Nx
        self.Ny = Ny
        self.λ = wavelength
        
        self.F = MonochromaticField(
            wavelength=wavelength, extent_x=extent_x, extent_y=extent_y, Nx=Nx, Ny=Ny, intensity = 0.001
        )


    def set_source_amplitude(self, amplitude_mask_path, image_size = None):
        
        #load the amplitude_mask image
        img = Image.open(Path(amplitude_mask_path))
        img = img.convert("RGB")

        rescaled_img = rescale_img_to_custom_coordinates(img, image_size, self.F.extent_x, self.F.extent_y, self.F.Nx, self.F.Ny)
        imgRGB = np.asarray(rescaled_img) / 255.0

        t = 0.2990 * imgRGB[:, :, 0] + 0.5870 * imgRGB[:, :, 1] + 0.1140 * imgRGB[:, :, 2]
        t = jnp.array(np.flip(t, axis = 0))

        self.source_amplitude = t
        self.source_size = image_size



    def set_target_amplitude(self, amplitude_mask_path, image_size = None):
        
        #load the amplitude_mask image
        img = Image.open(Path(amplitude_mask_path))
        img = img.convert("RGB")

        rescaled_img = rescale_img_to_custom_coordinates(img, image_size, self.F.extent_x, self.F.extent_y, self.Nx, self.Ny)
        imgRGB = np.asarray(rescaled_img) / 255.0

        t = 0.2990 * imgRGB[:, :, 0] + 0.5870 * imgRGB[:, :, 1] + 0.1140 * imgRGB[:, :, 2]
        t = jnp.array(np.flip(t, axis = 0))

        self.target_amplitude = t


    def retrieve_phase_mask(self,  max_iter = 20, method = 'Adam-Optimizer', propagation_method = 'Angular-Spectrum', learning_rate = 1.0):

        implemented_phase_retrieval_methods = ('Stochastic-Gradient-Descent', 'Adam-Optimizer')
        implemented_propagation_methods = ('Angular-Spectrum', 'Fresnel')

        import jax.numpy as jnp 
        from jax import value_and_grad, grad



        if propagation_method == 'Angular-Spectrum':

            def objective_function(phase):

                phase = phase.reshape(self.Ny, self.Nx) 
                self.F.E = self.source_amplitude*jnp.exp(1j*phase)
                self.F.z = 0
                self.F.propagate(z = self.z)
                
                return jnp.sum((jnp.abs(self.target_amplitude - jnp.abs(self.F.E))**2))
            
            self.objective_function = grad(objective_function)
            self.grad_F = grad(objective_function)


            def masked_objective_function(phase, mask):

                phase = phase.reshape(self.Ny, self.Nx) 
                self.F.E = self.source_amplitude*jnp.exp(1j*phase)
                self.F.z = 0
                self.F.propagate(z = self.z)
                
                f = (jnp.abs(self.target_amplitude - jnp.abs(self.F.E)))[mask]
                return jnp.sum(f**2)

            self.grad_F_masked = grad(masked_objective_function)

        elif propagation_method == 'Fresnel':
            
            def objective_function(phase):
                phase = phase.reshape(self.Ny, self.Nx) 
                self.F.E = self.source_amplitude*jnp.exp(1j*phase)
                self.F.z = 0
                self.F.scale_propagate(z = self.z, scale_factor=1)
                
                f = (jnp.abs(self.target_amplitude - jnp.abs(self.F.E)))
                return jnp.sum(f**2)


            self.grad_F = grad(objective_function)

            def masked_objective_function(phase, mask):

                phase = phase.reshape(self.Ny, self.Nx) 
                self.F.E = self.source_amplitude*jnp.exp(1j*phase)
                self.F.z = 0
                self.F.scale_propagate(z = self.z, scale_factor=1)
                f = (jnp.abs(self.target_amplitude - jnp.abs(self.F.E)))
                return jnp.sum(f**2)

            self.grad_F_masked = grad(masked_objective_function)




        else:
            raise NotImplementedError(
                f"{method} has not been implemented. Use one of {implemented_propagation_methods}")


        if method == 'Stochastic-Gradient-Descent':

            """
            Two batch stochastic gradient descent with momentum.
            """

            mass=0.9

            

            intial_phase = jnp.array(jnp.angle(jnp.fft.ifft2(jnp.fft.ifftshift(self.target_amplitude))))

            x = intial_phase
            velocity = np.zeros_like(x)

            bar = progressbar.ProgressBar()
            for i in bar(range(max_iter)):

                mask = jnp.array(np.random.randint(0,2,x.shape, dtype = bool))

                g = self.grad_F_masked(x,mask)

                velocity = mass * velocity - (1.0 - mass) * g
                x = x + learning_rate * velocity


                g = self.grad_F_masked(x,~mask)

                velocity = mass * velocity - (1.0 - mass) * g
                x = x + learning_rate * velocity


                #print(objective_function(x))

            i += 1
            retrieved_phase = x.reshape(self.Ny, self.Nx)
            self.retrieved_phase = np.where(retrieved_phase < 0, retrieved_phase + np.floor(np.min(retrieved_phase) / (2*np.pi)) * 2*np.pi, retrieved_phase )
            self.retrieved_phase = self.retrieved_phase % (2*np.pi)   -  np.pi

        elif method == 'Adam-Optimizer':

            """
            Reference: 
            Adam: A Method for Stochastic Optimization
            http://arxiv.org/pdf/1412.6980.pdf.
            """

            beta1=0.9
            beta2=0.999
            eps=1e-8

            

            intial_phase = jnp.array(jnp.angle(jnp.fft.ifft2(jnp.fft.ifftshift(self.target_amplitude))))

            x = intial_phase
            m = jnp.zeros_like(x)
            v = jnp.zeros_like(x)
            
            g = self.grad_F(x)

            bar = progressbar.ProgressBar()
            for i in bar(range(max_iter)):

                g = self.grad_F(x)
                m = (1 - beta1) * g + beta1 * m  # first  moment estimate.
                v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
                mhat = m / (1 - beta1**(i + 1))  # bias correction.
                vhat = v / (1 - beta2**(i + 1))
                x = x - learning_rate * mhat / (jnp.sqrt(vhat) + eps)


                #print(objective_function(x))

            i += 1

            retrieved_phase = x.reshape(self.Ny, self.Nx)
            self.retrieved_phase = np.where(retrieved_phase < 0, retrieved_phase + np.floor(np.min(retrieved_phase) / (2*np.pi)) * 2*np.pi, retrieved_phase )
            self.retrieved_phase = self.retrieved_phase % (2*np.pi)   -  np.pi

        else:
            raise NotImplementedError(
                f"{method} has not been implemented. Use one of {implemented_phase_retrieval_methods}")




        
    def save_retrieved_phase_as_image(self, name, phase_mask_format = 'hsv'):


        save_phase_mask_as_image(name, self.retrieved_phase, phase_mask_format = phase_mask_format)
