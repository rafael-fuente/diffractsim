import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from .complex_to_rgba import complex_to_rgba
from ..util.constants import *


"""

MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.

"""

def plot_phase(self, E, figsize=(7, 6), xlim=None, ylim=None, grid = False, text = None, max_val = 0.5, units = mm, dark_background = True):
    """visualize the diffraction pattern phase with matplotlib"""
    
    from ..util.backend_functions import backend as bd
    if dark_background == True:
        plt.style.use("dark_background")
    else:
        plt.style.use("default")

    if bd != np:
        E = E.get()
    else:
        E = E

    
    E = E / np.amax(np.abs(E))

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

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

    plt.subplots_adjust(right=0.8)


    im = ax.imshow(
        complex_to_rgba(E, max_val = max_val),
        extent=[
            float(self.x[0] - self.dx/2) / units,
            float(self.x[-1] + self.dx/2) / units,
            float(self.y[0] - self.dy/2)/ units,
            float(self.y[-1] + self.dy/2) / units,
        ],
        interpolation="spline36", origin = "lower"
    )

    colorbar_ax = plt.axes([0.83,0.1,0.018,0.8])
    cb1 = mpl.colorbar.ColorbarBase(colorbar_ax, cmap=mpl.cm.hsv,
                                    norm=mpl.colors.Normalize(vmin=-np.pi, vmax=np.pi),
                                    orientation='vertical')
    cb1.set_label('Phase [radians]')

    plt.show()
