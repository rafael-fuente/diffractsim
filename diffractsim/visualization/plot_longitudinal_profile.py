import matplotlib.pyplot as plt
import numpy as np
from ..util.constants import *


def plot_longitudinal_profile_colors(self, longitudinal_profile_rgb, start_distance, end_distance, xlim=None, ylim=None, units = mm):
    """visualize the diffraction pattern longitudinal profile colors with matplotlib"""

    from ..util.backend_functions import backend as bd
    plt.style.use("dark_background")

    if bd != np:
        longitudinal_profile_rgb = longitudinal_profile_rgb.transpose(1,0,2).get()
    else:
        longitudinal_profile_rgb = longitudinal_profile_rgb.transpose(1,0,2)

    fig = plt.figure(figsize=(16/9 *6,6)) 
    ax = fig.add_subplot(1,1,1)  
        
    ax.set_xlabel('Screen Distance [cm]')
    ax.set_title("Longitudinal Profile")
    
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

    im = ax.imshow(longitudinal_profile_rgb,  extent = [start_distance/cm,  end_distance/cm, float(self.x[0]) / units, float(self.x[-1] + self.dx) / units],  interpolation='spline36', aspect = 'auto')
    plt.show()


def plot_longitudinal_profile_intensity(self,  longitudinal_profile_E, start_distance, end_distance, square_root = False, grid = False, xlim=None, ylim=None, units = mm):
    """visualize the diffraction pattern longitudinal profile intensity with matplotlib"""

    from ..util.backend_functions import backend as bd
    plt.style.use("dark_background")

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

    if xlim != None:
        ax.set_xlim(np.array(xlim)/cm)

    if ylim != None:
        ax.set_ylim(np.array(ylim)/mm)

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


    fig = plt.figure(figsize=(16/9 *6,6)) 
    ax = fig.add_subplot(1,1,1)  


    
    ax.set_xlabel('Screen Distance [cm]')
    ax.set_title("Longitudinal Profile")
    if grid == True:
        ax.grid(alpha =0.2)

    im = ax.imshow(I, cmap= 'inferno',  extent = [start_distance/cm,  end_distance/cm, float(self.x[0]) / mm, float(self.x[-1]+ self.dx) / mm],  interpolation='spline36', aspect = 'auto')
    
    cb = fig.colorbar(im, orientation = 'vertical')

    if square_root == False:
        cb.set_label(r'Intensity $\left[W / m^2 \right]$', fontsize=13, labelpad =  14 )
    else:
        cb.set_label(r'Square Root Intensity $\left[ \sqrt{W / m^2 } \right]$', fontsize=13, labelpad =  14 )


    plt.show()
