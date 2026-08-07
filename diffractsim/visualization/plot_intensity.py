import matplotlib.pyplot as plt
import numpy as np
from ..util.constants import *

"""

MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.

"""


def plot_intensity(self, I, square_root = False, figsize=(7, 6), 
                  xlim=None, ylim=None, grid = False, text = None, units = mm,
                  slice_y_pos = None, slice_x_pos = None, dark_background = False):
    """visualize the diffraction pattern intesity with matplotlib"""
    
    from ..util.backend_functions import backend as bd
    from ..util.backend_functions import backend_name

    if dark_background == True:
        plt.style.use("dark_background")
    else:
        plt.style.use("default")

    if square_root == False:
        if backend_name == 'cupy':
            I = I.get()
        else:
            I = I

    else:
        if backend_name == 'cupy':
            I = np.sqrt(I.get())
        else:
            I = np.sqrt(I)


    fig = plt.figure(figsize=figsize)

    if (slice_y_pos == None) and (slice_x_pos == None):
        ax = fig.add_subplot(1, 1, 1)
    else:
        ax = fig.add_subplot(1, 2, 1)

    


    if grid == True:
        ax.grid(alpha =0.2)

    if xlim != None:
        ax.set_xlim(np.array(xlim)/units)

    if ylim != None:
        ax.set_ylim(np.array(ylim)/units)

    if units == mm:
        ax.set_xlabel("[mm]")
        ax.set_ylabel("[mm]")
    elif units == um:
        ax.set_xlabel("[um]")
        ax.set_ylabel("[um]")
    elif units == cm:
        ax.set_xlabel("[cm]")
        ax.set_ylabel("[cm]")
    elif units == nm:
        ax.set_xlabel("[nm]")
        ax.set_ylabel("[nm]")
    elif units == m:
        ax.set_xlabel("[m]")
        ax.set_ylabel("[m]")


    if text == None:
        ax.set_title("Screen distance = " + str(self.z * 100) + " cm")
    else: 
        ax.set_title(text)





    im = ax.imshow(
        I, cmap= 'inferno',
        extent=[  # the center of each pixel is exactly the point where the intensity is evaluated
            float(self.x[0] - self.dx/2) / units,
            float(self.x[-1] + self.dx/2) / units,
            float(self.y[0] - self.dy/2)/ units,
            float(self.y[-1] + self.dy/2) / units,
        ],
        interpolation="spline36", origin = "lower"
    )

    
    cb = fig.colorbar(im, orientation = 'vertical')

    if square_root == False:
        cb.set_label(r'Intensity $\left[W / m^2 \right]$', fontsize=10, labelpad =  10 )
    else:
        cb.set_label(r'Square Root Intensity $\left[ \sqrt{W / m^2 } \right]$', fontsize=10, labelpad =  10 )
    ax.set_aspect('equal')
    


    if slice_y_pos != None:
        ax_slice = fig.add_subplot(1, 2, 2)
        plt.subplots_adjust(wspace=0.3)
        ax_slice.set_title("X slice")
        #plt.subplots_adjust(right=2)

        if backend_name == 'cupy':
            x = self.x.get()
            y = self.y.get()
        else:
            x = self.x
            y = self.y

        if square_root == False:
            ax_slice.plot(x/units, I[np.argmin(abs(y-slice_y_pos)),:])
        else:
            ax_slice.plot(x/units, I[np.argmin(abs(y-slice_y_pos)),:]**2)
        ax_slice.set_ylabel(r'Intensity $\left[W / m^2 \right]$')

        if grid == True:
            ax_slice.grid(alpha =0.2)

        if xlim != None:
            ax_slice.set_xlim(np.array(xlim)/units)

        if units == mm:
            ax_slice.set_xlabel("[mm]")
        elif units == um:
            ax_slice.set_xlabel("[um]")
        elif units == cm:
            ax_slice.set_xlabel("[cm]")
        elif units == nm:
            ax_slice.set_xlabel("[nm]")
        elif units == m:
            ax_slice.set_xlabel("[m]")

    if slice_x_pos != None:
        ax_slice = fig.add_subplot(1, 2, 2)
        plt.subplots_adjust(wspace=0.3)
        ax_slice.set_title("Y slice")
        #plt.subplots_adjust(right=2)

        if backend_name == 'cupy':
            x = self.x.get()
            y = self.y.get()
        else:
            x = self.x
            y = self.y

        if square_root == False:
            ax_slice.plot(y/units, I[:, np.argmin(abs(x-slice_x_pos))])
        else:
            ax_slice.plot(y/units, I[:, np.argmin(abs(x-slice_x_pos))]**2)
        ax_slice.set_ylabel(r'Intensity $\left[W / m^2 \right]$')

        if grid == True:
            ax_slice.grid(alpha =0.2)

        if xlim != None:
            ax_slice.set_xlim(np.array(ylim)/units)

        if units == mm:
            ax_slice.set_xlabel("[mm]")
        elif units == um:
            ax_slice.set_xlabel("[um]")
        elif units == cm:
            ax_slice.set_xlabel("[cm]")
        elif units == nm:
            ax_slice.set_xlabel("[nm]")
        elif units == m:
            ax_slice.set_xlabel("[m]")





    plt.show()




def plot_farfield(self, α, β, Irad, figsize=(7, 6), 
                  alpha_lim=None, beta_lim=None, grid = False, text = None, dark_background = False):
    """visualize the far field diffraction pattern intesity with matplotlib"""
    
    from ..util.backend_functions import backend as bd
    from ..util.backend_functions import backend_name
    
    

    if dark_background == True:
        plt.style.use("dark_background")
    else:
        plt.style.use("default")

    if backend_name == 'cupy':
        Irad = Irad.get()
    else:
        Irad = Irad


    fig = plt.figure(figsize=figsize)

    ax = fig.add_subplot(1, 1, 1)


    if grid == True:
        ax.grid(alpha =0.2)

    if alpha_lim != None:
        ax.set_xlim(np.array(alpha_lim))

    if beta_lim != None:
        ax.set_ylim(np.array(beta_lim))



    if text == None:
        ax.set_title("Far field")
    else: 
        ax.set_title(text)


    dα = α[1]-α[0]
    dβ = β[1]-β[0]


    im = ax.imshow(
        Irad, cmap= 'inferno',
        extent=[α[0],α[-1] + dα,  β[0],  β[-1] + dβ],
        interpolation="spline36", origin = "lower"
    )
    ax.set_xlabel('$α$')
    ax.set_ylabel("$β$")

    
    
    cb = fig.colorbar(im, orientation = 'vertical')

    cb.set_label(r"$\frac{\partial P_e(α,β)}{\partial \Omega \cos(\theta)} [a.u.]$", fontsize=10, labelpad =  10 )
    ax.set_aspect('equal')


    plt.show()









def plot_farfield_spherical_coordinates(self, α, β, Irad, figsize=(7, 6), 
                  theta_lim=None,  text = None, dark_background = False):
    """visualize the far field diffraction pattern intesity with matplotlib"""
    
    from ..util.backend_functions import backend as bd
    from ..util.backend_functions import backend_name

    if dark_background == True:
        plt.style.use("dark_background")
    else:
        plt.style.use("default")

    if backend_name == 'cupy':
        Irad = Irad.get()
    else:
        Irad = Irad

    if theta_lim == None:
        theta_lim = 90
        
        
    # interpolate to spherical coordinates
    from scipy.interpolate import interpn
    oldpoints = (α,β)
    Nθ, Nφ  = 4096,4096
    θ = np.linspace(0,theta_lim *np.pi/180 ,Nθ)
    φ = np.linspace(0,2*np.pi,Nφ)
    θθ,φφ = np.meshgrid(θ,φ, indexing='ij')

    αα = np.sin(θθ)*np.cos(φφ) 
    ββ = np.sin(θθ)*np.sin(φφ)
    γγ = np.cos(θθ)
    newpoints = np.array(np.vstack([αα.ravel(),ββ.ravel()]).T)
    Irad_ = interpn(oldpoints, np.array(np.real(Irad).T), newpoints, bounds_error=False,fill_value = 0)  .reshape((Nθ, Nφ))
        
        
        
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(projection='polar')

    if text == None:
        ax.set_title("Far field")
    else: 
        ax.set_title(text)


    ax.grid(False)
    im = ax.pcolormesh(φ, θ* 180 / np.pi, Irad_, cmap = 'inferno')

    ax.set_ylim([0,theta_lim])
    ax.tick_params(axis='y', colors='white')

    label_position=ax.get_rlabel_position()

    import math
    ax.text(math.radians(label_position- label_position*0.5),(ax.get_rmax())/2.,'θ°',ha='center',va='center', color  = 'white')

    ax.grid(alpha =0.2)

    cb1 = fig.colorbar(im, orientation = 'vertical', pad = 0.1, shrink = 0.8)

    cb1.set_label(r'$\frac{\partial P_e(θ,φ)}{\partial \Omega \cos(\theta)}$ [a.u.]', size= 15)

    ax.set_title(r"φ")
    plt.show()    
    
