import numpy as np
from ..util.backend_functions import backend as bd
from .light_source import LightSource

class SpatialNoise(LightSource):
    def __init__(self, noise_radius, f_mean, f_spread, N = 30, A = 1):
        """
        add spatial noise following a radial normal distribution

        Parameters
        ----------
        noise_radius: maximum spatial radius affected by the spatial noise
        f_mean: mean modulus of the spatial frequency of the spatial noise 
        f_bandwidth: spread spatial frequency of the noise 
        N: number of samples
        A: amplitude of the noise
        """
        global bd
        from ..util.backend_functions import backend as bd

        self.noise_radius = noise_radius 
        self.f_mean = f_mean 
        self.f_spread = f_spread 
        self.N = N
        self.A = A

    def get_E(self, E, xx, yy, λ):

        x_extent = xx[0,-1] - xx[0,0] + xx[0,1] - xx[0,0] 
        y_extent = yy[-1,0] - yy[0,0] + yy[1,0] - yy[0,0] 

        Ny , Nx = xx.shape

        fx_extent = Nx/x_extent
        fy_extent = Ny/y_extent

        dfx = fx_extent/Nx
        dfy = fy_extent/Ny

        fx = dfx*(bd.arange(Nx)-Nx//2)
        fy = dfy*(bd.arange(Ny)-Ny//2)
        fxx, fyy = bd.meshgrid(fx, fy)

        phase = bd.random.rand(Ny, Nx)*2*bd.pi

        fp = bd.sqrt(fxx**2 + fyy**2)
        spatial_noise_freq = bd.exp(- 1/2 * ( (fp - self.f_mean)/ (self.f_spread))**2) * bd.exp(-1j*phase)

        nn, mm = bd.meshgrid(bd.arange(Nx)-Nx//2, bd.arange(Ny)-Ny//2)
        spatial_noise_amp = bd.fft.fftshift(bd.fft.fft2(spatial_noise_freq))*(bd.exp(bd.pi*1j * (nn + mm))) *bd.exp(-(xx**2 + yy**2)/ (self.noise_radius)**2)
        spatial_noise_amp = spatial_noise_amp/bd.amax(bd.abs(spatial_noise_amp))
        self.spatial_noise_amp = spatial_noise_amp *bd.exp(-(xx**2 + yy**2)/ (self.noise_radius)**2) * self.A

        return self.spatial_noise_amp + E
