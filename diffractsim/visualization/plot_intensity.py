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
                  slice_y_pos = None, slice_x_pos = None, dark_background = True):
    """visualize the diffraction pattern intesity with matplotlib"""
    
    from ..util.backend_functions import backend as bd
    if dark_background == True:
        plt.style.use("dark_background")
    else:
        plt.style.use("default")

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

        if bd != np:
            x = self.x.get()
            y = self.y.get()
        else:
            x = self.x
            y = self.y

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

        if bd != np:
            x = self.x.get()
            y = self.y.get()
        else:
            x = self.x
            y = self.y

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
