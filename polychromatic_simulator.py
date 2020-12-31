import numpy as np
from scipy.fftpack import fft2 , ifft2 , fftshift , ifftshift
from pathlib import Path
import colour_functions as cf
import matplotlib.pyplot as plt
import copy
from PIL import Image
from scipy.interpolate import interp2d
import progressbar
nm = 1e-9
mm = 1e-3

class PolychromaticField():
    def __init__(self,spectrum,extent_x,extent_y, Nx, Ny):
        
        self.extent_x = extent_x
        self.extent_y = extent_y

        self.x = np.linspace(-extent_x/2,extent_x/2,Nx)
        self.y = np.linspace(-extent_y/2,extent_y/2,Ny)
        self.xx,self.yy = np.meshgrid(self.x, self.y)
        
        self.Nx = np.int(Nx)
        self.Ny = np.int(Ny)
        self.E = np.ones((int(self.Ny), int(self.Nx)))
        self.spectrum = spectrum
        

    def add_rectangular_slit(self,x0, y0, width, height):
        """
        Creates a slit centered at the point (x0, y0) with width width and height height
        """
        self.E = np.select( [((self.xx > (x0 - width/2) ) & (self.xx < (x0 + width/2) )) & ((self.yy > (y0 - height/2) ) & (self.yy < (y0 + height/2) )),  True], [self.E, 0])
                    
    def add_circular_slit(self,x0, y0, R):
        """
        Crea una rendija centrada en el punto (x0,y0) con anchura lx y altura ly
        """
        
        self.E = np.select( [(self.xx-x0)**2 + (self.yy-y0)**2  < R**2,  True], [self.E, 0])
                
    def add_diffraction_grid(self,D, a, Nx, Ny):
        E0 = np.copy(self.E)
        Ef = 0
        
        b = D-a
        width, height = Nx*a + (Nx-1)*b , Ny*a + (Ny-1)*b
        x0 ,y0 = -width/2 ,  height/2
        
        x0 = -width/2 + a/2
        for i in range(Nx):
            y0 = height/2 - a/2
            for j in range(Ny):
                
                Ef += np.select( [((self.xx > (x0 - a/2) ) & (self.xx < (x0 + a/2) )) & ((self.yy > (y0 - a/2) ) & (self.yy < (y0 + a/2) )),  True], [E0, 0])
                y0 -= D
            x0 += D 
        self.E = Ef


    def add_aperture_from_image(self,path, pad = None , Nx = None, Ny = None):
        # This function load the image specified at "path" as a numpy graymap array. 
        # If Nx and Ny is specified, we interpolate the pattern with interp2d method to the new specified resolution. 
        # If pad is specified, we add zeros (black color) padded to the edges of each axis.

        img = Image.open(Path(path))
        img = img.convert('RGB')
        imgRGB = np.asarray(img)/256.
        imgR = imgRGB[:,:,0]
        imgG = imgRGB[:,:,1]
        imgB = imgRGB[:,:,2]
        self.E = 0.2989 * imgR + 0.5870 * imgG + 0.1140 * imgB

        fun = interp2d(np.linspace(0,1,self.E.shape[1]), np.linspace(0,1,self.E.shape[0]), self.E, kind='cubic')
        self.E = fun(np.linspace(0,1,self.Nx),  np.linspace(0,1,self.Ny))
         
        if pad != None:
            
            Nxpad =  int(np.round(self.Nx / self.extent_x * pad[0]))
            Nypad =  int(np.round(self.Ny / self.extent_y * pad[1]))
            self.E = np.pad(self.E, ((Nypad,Nypad),(Nxpad,Nxpad)), "constant")
            
            scale_ratio = self.E.shape[1]/self.E.shape[0]
            print(scale_ratio)
            if Nx ==None:
                self.Nx = int(np.round(self.E.shape[0]*scale_ratio))
            else:
                self.Nx = Nx
            if Ny ==None:
                self.Ny = self.E.shape[0]
            else:
                self.Ny = Ny

            self.extent_x += 2*pad[0]
            self.extent_y += 2*pad[1]

            fun = interp2d(np.linspace(0,1,self.E.shape[1]), np.linspace(0,1,self.E.shape[0]), self.E, kind='cubic')
            self.E = fun(np.linspace(0,1,self.Nx),  np.linspace(0,1,self.Ny))

        self.x = np.linspace(-self.extent_x/2,self.extent_x/2,self.Nx)
        self.y = np.linspace(-self.extent_y/2,self.extent_y/2,self.Ny)
        self.xx,self.yy = np.meshgrid(self.x, self.y)


        
        
    def compute_colors_at(self,z, spectrum_divisions = 40,grid_divisions = 10):
        

        self.z = z
        fft_c = fft2(self.E)
        c = fftshift(fft_c)
        kx = np.linspace(-np.pi*self.Nx//2 /(self.extent_x/2),  np.pi*self.Nx//2 /(self.extent_x/2),  self.Nx)
        ky = np.linspace(-np.pi*self.Ny//2 /(self.extent_y/2),  np.pi*self.Ny//2 /(self.extent_y/2),  self.Ny)
        kx,ky = np.meshgrid(kx, ky)
        
        
        dλ = (780- 380)/spectrum_divisions 
        sRGB_linear = np.zeros((3,self.Nx*self.Ny))
        λ_list_samples = np.arange(380, 780, dλ)
        
        bar = progressbar.ProgressBar()

        # We compute the pattern of each wavelength separately, and associate it to small spectrum interval dλ = (780- 380)/spectrum_divisions . We approximately the final colour
        # by summing the contribution of each small spectrum interval converting its intensity distribution to a RGB space.
        for i in bar(range(spectrum_divisions)):

            kz = np.sqrt( (2*np.pi/(λ_list_samples[i]*nm))**2 - kx**2 - ky**2)

            E_λ = ifft2(ifftshift(c*np.exp(1j*kz * z)))
            Iλ = np.real(E_λ*np.conjugate(E_λ))
            spec_div = np.where((cf.λ_list > λ_list_samples[i]) & (cf.λ_list < λ_list_samples[i] + dλ) , self.spectrum, 0)
            
            # Its likely that you don't have enough RAM to deal with the Nx x Ny x spectrum_divisions array, so we split the array
            # with grid_divisions 
            if grid_divisions == 1:
                XYZ = cf.spec_to_XYZ(np.outer(Iλ, spec_div))
                sRGB_linear += cf.XYZ_to_sRGB_linear(XYZ)
            else:
                Iλ_split = np.split(Iλ, grid_divisions)
                XYZ_list = []
                for Iλ_ in Iλ_split:
                    XYZ_list += [cf.spec_to_XYZ(np.outer(Iλ_, spec_div))]
                XYZ = np.concatenate(XYZ_list, axis=1)
                sRGB_linear += cf.XYZ_to_sRGB_linear(XYZ)



        sRGB_linear += cf.XYZ_to_sRGB_linear(XYZ)

        rgb = cf.sRGB_linear_to_sRGB(sRGB_linear)
        rgb = (rgb.T).reshape((self.Ny,self.Nx,3))
        return rgb

    def plot(self,rgb, figsize=(6,6), xlim = None, ylim = None):
        plt.style.use('dark_background')

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1,1,1)  

        if xlim != None:
            ax.set_xlim(xlim)
            
        if ylim != None:
            ax.set_ylim(ylim)
        ax.set_xlabel('[mm]')
        ax.set_ylabel('[mm]')

        ax.set_title("Screen distance = " + str(self.z*100) + " cm")
        ax.set_aspect('equal')

        im = ax.imshow((rgb),  extent = [-self.extent_x/2/mm, self.extent_x/2/mm, -self.extent_y/2/mm, self.extent_y/2/mm],  interpolation='spline36')
        plt.show()


