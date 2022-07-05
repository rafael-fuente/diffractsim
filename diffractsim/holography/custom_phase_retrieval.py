import numpy as np
from ..util.backend_functions import backend as bd

from ..util.file_handling import load_graymap_image_as_array, save_phase_mask_as_image
from ..util.image_handling import rescale_img_to_custom_coordinates
from pathlib import Path
from PIL import Image
from ..util.constants import *
import progressbar

"""

MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.

"""


# implemented custom phase retrieval methods: ('Conjugate-Gradient', 'Stochastic-Gradient-Descent', 'Adam-Optimizer')

class CustomPhaseRetrieval():
    def __init__(self, wavelength, z, extent_x, extent_y, Nx, Ny):
        "class for retrieve the phase mask required to reconstruct an image (specified at target amplitude path) at a distance z"

        global bd
        from ..util.backend_functions import backend as bd


        self.extent_x = extent_x
        self.extent_y = extent_y
        self.z = z
        self.dx = self.extent_x/Nx
        self.dy = self.extent_y/Ny

        self.x = self.dx*(bd.arange(Nx)-Nx//2)
        self.y = self.dy*(bd.arange(Ny)-Ny//2)
        self.xx, self.yy = bd.meshgrid(self.x, self.y)

        self.Nx = Nx
        self.Ny = Ny
        self.λ = wavelength



    def set_source_amplitude(self, amplitude_mask_path, image_size = None):
        
        #load the amplitude_mask image
        img = Image.open(Path(amplitude_mask_path))
        img = img.convert("RGB")

        rescaled_img = rescale_img_to_custom_coordinates(img, image_size, self.extent_x, self.extent_y, self.Nx, self.Ny)
        imgRGB = np.asarray(rescaled_img) / 255.0

        t = 0.2990 * imgRGB[:, :, 0] + 0.5870 * imgRGB[:, :, 1] + 0.1140 * imgRGB[:, :, 2]
        t = bd.array(np.flip(t, axis = 0))

        self.source_amplitude = t
        self.source_size = image_size



    def set_target_amplitude(self, amplitude_mask_path, image_size = None):
        
        #load the amplitude_mask image
        img = Image.open(Path(amplitude_mask_path))
        img = img.convert("RGB")

        rescaled_img = rescale_img_to_custom_coordinates(img, image_size, self.extent_x, self.extent_y, self.Nx, self.Ny)
        imgRGB = np.asarray(rescaled_img) / 255.0

        t = 0.2990 * imgRGB[:, :, 0] + 0.5870 * imgRGB[:, :, 1] + 0.1140 * imgRGB[:, :, 2]
        t = bd.array(np.flip(t, axis = 0))

        self.target_amplitude = t


    def retrieve_phase_mask(self,  max_iter = 20, method = 'Adam-Optimizer', propagation_method = 'Angular-Spectrum', learning_rate = 1.0):

        implemented_phase_retrieval_methods = ('Conjugate-Gradient', 'Stochastic-Gradient-Descent', 'Adam-Optimizer')
        implemented_propagation_methods = ('Angular-Spectrum', 'Fresnel')

        import autograd.numpy as npa 
        from autograd import elementwise_grad as egrad  


        if propagation_method == 'Angular-Spectrum':

            fx = npa.fft.fftshift(npa.fft.fftfreq(self.Nx, d = self.dx))
            fy = npa.fft.fftshift(npa.fft.fftfreq(self.Ny, d = self.dy))
            fxx, fyy = np.meshgrid(fx, fy)


            def objective_function(phase):

                phase = phase.reshape(self.Ny, self.Nx)                
                c = npa.fft.fftshift(npa.fft.fft2(self.source_amplitude*npa.exp(1j*phase)))
                argument = (2 * npa.pi)**2 * ((1. / self.λ) ** 2 - fxx ** 2 - fyy ** 2)

                #Calculate the propagating and the evanescent (complex) modes
                tmp = npa.sqrt(npa.abs(argument))
                kz = npa.where(argument >= 0, tmp, 1j*tmp)
                E = npa.abs(npa.fft.ifft2(npa.fft.ifftshift(c * npa.exp(1j * kz * self.z))))
                return npa.sum(((self.target_amplitude - E)**2))

            self.grad_F = egrad(objective_function)


            def masked_objective_function(phase, mask):

                phase = phase.reshape(self.Ny, self.Nx)                
                c = npa.fft.fftshift(npa.fft.fft2(self.source_amplitude*npa.exp(1j*phase)))
                argument = (2 * npa.pi)**2 * ((1. / self.λ) ** 2 - fxx ** 2 - fyy ** 2)

                #Calculate the propagating and the evanescent (complex) modes
                tmp = npa.sqrt(npa.abs(argument))
                kz = npa.where(argument >= 0, tmp, 1j*tmp)
                E = npa.abs(npa.fft.ifft2(npa.fft.ifftshift(c * npa.exp(1j * kz * self.z))))
                f = (self.target_amplitude - E)[mask]
                return npa.sum(f**2)

            self.grad_F_masked = egrad(masked_objective_function)

        elif propagation_method == 'Fresnel':

            def objective_function(phase):

                phase = phase.reshape(self.Ny, self.Nx)
                E = npa.abs(npa.fft.fftshift( npa.fft.fft2( self.source_amplitude*npa.exp(1j * 2*np.pi/self.λ /(2*self.z) *(self.xx**2 + self.yy**2))*  npa.exp(1j*phase))))
                #print(npa.sum((self.target_amplitude/npa.sum(self.target_amplitude) - (E)/npa.sum(E))**2))
                return npa.sum((self.target_amplitude - E)**2)

            self.grad_F = egrad(objective_function)

            def masked_objective_function(phase, mask):

                phase = phase.reshape(self.Ny, self.Nx)
                E = npa.abs(npa.fft.fftshift( npa.fft.fft2( self.source_amplitude*npa.exp(1j * 2*np.pi/self.λ /(2*self.z) *(self.xx**2 + self.yy**2))*  npa.exp(1j*phase))))
                f = (self.target_amplitude - E)[mask]
                return npa.sum(f**2)

            self.grad_F_masked = egrad(masked_objective_function)




        else:
            raise NotImplementedError(
                f"{method} has not been implemented. Use one of {implemented_propagation_methods}")


        if method == 'Conjugate-Gradient':

            """
            Nonlinear conjugate gradient algorithm by Polak and Ribiere

            Reference: 
            Nocedal, J, and S J Wright. 2006. Numerical Optimization. Springer New York.
            https://docs.scipy.org/doc/scipy/reference/optimize.minimize-cg.html#optimize-minimize-cg
            """

            import scipy
            from scipy.optimize import minimize


            intial_phase = np.array(np.angle(np.fft.ifft2(np.fft.ifftshift(self.target_amplitude))))
            result  = scipy.optimize.minimize(objective_function, intial_phase, method='CG', jac=self.grad_F, tol = 1e-8, options={'maxiter': max_iter})
            retrieved_phase = result.x.reshape(self.Ny, self.Nx)
            self.retrieved_phase = np.where(retrieved_phase < 0, retrieved_phase + np.floor(np.min(retrieved_phase) / (2*np.pi)) * 2*np.pi, retrieved_phase )
            self.retrieved_phase = self.retrieved_phase % (2*np.pi)   -  np.pi

        elif method == 'Stochastic-Gradient-Descent':

            """
            Two batch stochastic gradient descent with momentum.
            """

            mass=0.9

            from scipy.optimize import OptimizeResult

            intial_phase = np.array(np.angle(np.fft.ifft2(np.fft.ifftshift(self.target_amplitude))))

            x = intial_phase
            velocity = np.zeros_like(x)

            bar = progressbar.ProgressBar()
            for i in bar(range(max_iter)):

                mask = np.random.randint(0,2,x.shape, dtype = bool)

                g = self.grad_F_masked(x,mask)

                velocity = mass * velocity - (1.0 - mass) * g
                x = x + learning_rate * velocity


                g = self.grad_F_masked(x,~mask)

                velocity = mass * velocity - (1.0 - mass) * g
                x = x + learning_rate * velocity


                #print(objective_function(x))

            i += 1
            result = OptimizeResult(x=x, fun=objective_function(x), jac=g, nit=i, nfev=i, success=True)
            retrieved_phase = result.x.reshape(self.Ny, self.Nx)
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

            from scipy.optimize import OptimizeResult

            intial_phase = np.array(np.angle(np.fft.ifft2(np.fft.ifftshift(self.target_amplitude))))

            x = intial_phase
            m = np.zeros_like(x)
            v = np.zeros_like(x)

            bar = progressbar.ProgressBar()
            for i in bar(range(max_iter)):
                mask = np.random.randint(0,2,x.shape)

                g = self.grad_F(x)
                m = (1 - beta1) * g + beta1 * m  # first  moment estimate.
                v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
                mhat = m / (1 - beta1**(i + 1))  # bias correction.
                vhat = v / (1 - beta2**(i + 1))
                x = x - learning_rate * mhat / (np.sqrt(vhat) + eps)


                #print(objective_function(x))

            i += 1

            result = OptimizeResult(x=x, fun=objective_function(x), jac=g, nit=i, nfev=i, success=True)
            retrieved_phase = result.x.reshape(self.Ny, self.Nx)
            self.retrieved_phase = np.where(retrieved_phase < 0, retrieved_phase + np.floor(np.min(retrieved_phase) / (2*np.pi)) * 2*np.pi, retrieved_phase )
            self.retrieved_phase = self.retrieved_phase % (2*np.pi)   -  np.pi

        else:
            raise NotImplementedError(
                f"{method} has not been implemented. Use one of {implemented_phase_retrieval_methods}")




        
    def save_retrieved_phase_as_image(self, name, phase_mask_format = 'hsv'):


        if bd == np:
            save_phase_mask_as_image(name, self.retrieved_phase, phase_mask_format = phase_mask_format)
        else:
            save_phase_mask_as_image(name, self.retrieved_phase.get(), phase_mask_format = phase_mask_format)
