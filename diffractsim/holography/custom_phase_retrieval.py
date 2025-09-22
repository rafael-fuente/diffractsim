import numpy as np
from ..util.backend_functions import backend as jnp

from ..util.file_handling import load_graymap_image_as_array, save_phase_mask_as_image
from ..util.image_handling import rescale_img_to_custom_coordinates
from ..monochromatic_simulator import MonochromaticField
from ..propagation_methods import two_steps_fresnel_method

from pathlib import Path
from PIL import Image
from ..util.constants import *
import progressbar

from ..util.image_handling import load_image_as_function



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


    def set_source_amplitude(self, source_function):
        
        self.source_function = source_function
        self.source_amplitude = source_function(self.F.xx, self.F.yy)



    def set_target_amplitude(self, target_function):
        
        #load the amplitude_mask image
        self.target_function = target_function
        self.target_amplitude = target_function(self.F.xx, self.F.yy)


    def retrieve_phase_mask(self,  max_iter = 20, method = 'Adam-Optimizer', propagation_method = 'Angular-Spectrum', learning_rate = 1.0, custom_objective_function = None):

        implemented_phase_retrieval_methods = ('Stochastic-Gradient-Descent', 'Adam-Optimizer', 'LBFGS')
        implemented_propagation_methods = ('Custom', 'Angular-Spectrum', 'Fresnel')

        import jax.numpy as jnp 
        from jax import value_and_grad, grad

        if propagation_method == 'Custom':
            self.objective_function = custom_objective_function
            self.grad_F = grad(objective_function)

        elif propagation_method == 'Angular-Spectrum':

            def objective_function(phase):

                phase = phase.reshape(self.Ny, self.Nx) 
                self.F.E = self.source_amplitude*jnp.exp(1j*phase)
                newF = self.F.propagate(z = self.z)
                return jnp.sum((jnp.abs(self.target_amplitude - jnp.abs(newF.E))**2))

            
            self.objective_function = objective_function
            self.grad_F = grad(objective_function)


        elif propagation_method == 'Fresnel':
            
            def objective_function(phase):
                phase = phase.reshape(self.Ny, self.Nx) 
                self.F.E = self.source_amplitude*jnp.exp(1j*phase)
                newF = self.F.scale_propagate(z = self.z, scale_factor = 1)
                
                return jnp.sum((jnp.abs(self.target_amplitude - jnp.abs(newF.E))**2))

            self.objective_function = objective_function
            self.grad_F = grad(objective_function)



        else:
            raise NotImplementedError(
                f"{method} has not been implemented. Use one of {implemented_propagation_methods}")



        if method == 'Adam-Optimizer':

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
            print("Final loss:", self.objective_function(x))

            retrieved_phase = x.reshape(self.Ny, self.Nx)
            self.retrieved_phase = np.where(retrieved_phase < 0, retrieved_phase + np.floor(np.min(retrieved_phase) / (2*np.pi)) * 2*np.pi, retrieved_phase )
            self.retrieved_phase = self.retrieved_phase % (2*np.pi)   -  np.pi
            
        elif method == 'LBFGS':
            import jaxopt

            intial_phase = jnp.array(jnp.angle(jnp.fft.ifft2(jnp.fft.ifftshift(self.target_amplitude)))).ravel()

            x = intial_phase
            solver = jaxopt.LBFGS(fun=objective_function, maxiter=max_iter)
            res = solver.run(intial_phase)
            x, state = res


            print("Final loss:", self.objective_function(x))

            retrieved_phase = x.reshape(self.Ny, self.Nx)
            self.retrieved_phase = np.where(retrieved_phase < 0, retrieved_phase + np.floor(np.min(retrieved_phase) / (2*np.pi)) * 2*np.pi, retrieved_phase )
            self.retrieved_phase = self.retrieved_phase % (2*np.pi)   -  np.pi



        elif method == 'Adam-JAX':
            import jax
            from jax import example_libraries
            from jax.example_libraries import optimizers

            phi_init = jnp.array(jnp.angle(jnp.fft.ifft2(jnp.fft.ifftshift(self.target_amplitude))))
            num_epochs = max_iter
            
            
            @jax.jit
            def step(i, opt_state):
                phi = get_params(opt_state)
                g = self.grad_F(phi)
                return opt_update(i, g, opt_state)
            
            opt_init, opt_update, get_params = optimizers.adam(learning_rate)
            opt_state = opt_init(phi_init)  # initialize φ

            for i in range(num_epochs):
                opt_state = step(i, opt_state)
                if i % 50 == 0:
                    current_loss = self.objective_function(get_params(opt_state))
                    print(f"Epoch {i}, Loss: {current_loss:.6f}")

            x = get_params(opt_state)
            print("Final loss:", self.objective_function(x))

            retrieved_phase = x.reshape(self.Ny, self.Nx)
            self.retrieved_phase = np.where(retrieved_phase < 0, retrieved_phase + np.floor(np.min(retrieved_phase) / (2*np.pi)) * 2*np.pi, retrieved_phase )
            self.retrieved_phase = self.retrieved_phase % (2*np.pi)   -  np.pi

        else:
            raise NotImplementedError(
                f"{method} has not been implemented. Use one of {implemented_phase_retrieval_methods}")


        
    def save_retrieved_phase_as_image(self, name, phase_mask_format = 'hsv'):
        save_phase_mask_as_image(name, self.retrieved_phase, phase_mask_format = phase_mask_format)
        
    def save_retrieved_phase_as_file(self, name):
        np.save(name, self.retrieved_phase)
