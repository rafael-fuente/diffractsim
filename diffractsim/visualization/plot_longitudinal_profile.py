import matplotlib.pyplot as plt
import numpy as np
from ..util.constants import *


"""

MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.

"""

def plot_longitudinal_profile_colors(self, longitudinal_profile_rgb, extent, xlim=None, ylim=None, units = mm,  z_units = cm, dark_background = True):
    """visualize the diffraction pattern longitudinal profile colors with matplotlib"""

    from ..util.backend_functions import backend as bd
    if dark_background == True:
        plt.style.use("dark_background")
    else:
        plt.style.use("default")

    if bd != np:
        longitudinal_profile_rgb = longitudinal_profile_rgb.transpose(1,0,2).get()
    else:
        longitudinal_profile_rgb = longitudinal_profile_rgb.transpose(1,0,2)

    fig = plt.figure(figsize=(16/9 *6,6)) 
    ax = fig.add_subplot(1,1,1)  
        
    ax.set_title("Longitudinal Profile")
    
    if xlim != None:
        ax.set_xlim(np.array(xlim)/z_units)
    if ylim != None:
        ax.set_ylim(np.array(ylim)/units)



    if units == mm:
        ax.set_ylabel("[mm]")
    elif units == um:
        ax.set_ylabel("[um]")
    elif units == cm:
        ax.set_ylabel("[cm]")
    elif units == nm:
        ax.set_ylabel("[nm]")
    elif units == m:
        ax.set_ylabel("[m]")



    if z_units == mm:
        ax.set_xlabel('Screen Distance [mm]')
    elif z_units == um:
        ax.set_xlabel('Screen Distance [um]')
    elif z_units == cm:
        ax.set_xlabel('Screen Distance [cm]')
    elif z_units == nm:
        ax.set_xlabel('Screen Distance [nm]')
    elif z_units == m:
        ax.set_xlabel('Screen Distance [m]')


    dz = (extent[3] - extent[2])/ longitudinal_profile_rgb.shape[1]

    im = ax.imshow(longitudinal_profile_rgb,  extent = [(extent[2]- dz/2)/z_units ,  (extent[3]+ dz/2)/z_units, float(extent[0]- self.dx/2) / units, float(extent[1]+ self.dx/2) / units],  interpolation='spline36', aspect = 'auto')
    plt.show()


def plot_longitudinal_profile_intensity(self,  longitudinal_profile_E, extent,  square_root = False, grid = False, xlim=None, ylim=None, units = mm,  z_units = cm, dark_background = True):
    """visualize the diffraction pattern longitudinal profile intensity with matplotlib"""

    from ..util.backend_functions import backend as bd
    if dark_background == True:
        plt.style.use("dark_background")
    else:
        plt.style.use("default")

    I = bd.real(longitudinal_profile_E*np.conjugate(longitudinal_profile_E))
    I = I.transpose(1,0)

    if square_root == False:
        if bd != np:
            I = I.get()
        else:
            I = I

    else:
        if bd != np:
            I = np.sqrt(I.get())
        else:
            I = np.sqrt(I)

    fig = plt.figure(figsize=(16/9 *6,6)) 
    ax = fig.add_subplot(1,1,1)  


    if xlim != None:
        ax.set_xlim(np.array(xlim)/cm)

    if ylim != None:
        ax.set_ylim(np.array(ylim)/units)

    if units == mm:
        ax.set_ylabel("[mm]")
    elif units == um:
        ax.set_ylabel("[um]")
    elif units == cm:
        ax.set_ylabel("[cm]")
    elif units == nm:
        ax.set_ylabel("[nm]")
    elif units == m:
        ax.set_ylabel("[m]")



    if z_units == mm:
        ax.set_xlabel('Screen Distance [mm]')
    elif z_units == um:
        ax.set_xlabel('Screen Distance [um]')
    elif z_units == cm:
        ax.set_xlabel('Screen Distance [cm]')
    elif z_units == nm:
        ax.set_xlabel('Screen Distance [nm]')
    elif z_units == m:
        ax.set_xlabel('Screen Distance [m]')



    ax.set_title("Longitudinal Profile")
    if grid == True:
        ax.grid(alpha =0.2)

    dz = (extent[3] - extent[2])/ I.shape[1]
    im = ax.imshow(I, cmap= 'inferno',  extent = [(extent[2]- dz/2)/z_units ,  (extent[3]+ dz/2)/z_units, float(extent[0]- self.dx/2) / units, float(extent[1]+ self.dx/2) / units],  interpolation='spline36', aspect = 'auto')
    
    cb = fig.colorbar(im, orientation = 'vertical')

    if square_root == False:
        cb.set_label(r'Intensity $\left[W / m^2 \right]$', fontsize=13, labelpad =  14 )
    else:
        cb.set_label(r'Square Root Intensity $\left[ \sqrt{W / m^2 } \right]$', fontsize=13, labelpad =  14 )


    plt.show()
