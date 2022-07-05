import matplotlib.pyplot as plt
import numpy as np
from ..util.constants import *


"""

MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.

"""

def plot_colors(self, rgb, figsize=(6, 6), xlim=None, ylim=None, text = None, units = mm, dark_background = True):
    """visualize the diffraction pattern colors with matplotlib"""

    from ..util.backend_functions import backend as bd
    if dark_background == True:
        plt.style.use("dark_background")
    else:
        plt.style.use("default")
    if bd != np:
        rgb = rgb.get()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    if xlim != None:
        ax.set_xlim(np.array(xlim)/units)

    if ylim != None:
        ax.set_ylim(np.array(ylim)/units)

    # we use mm by default
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
        (rgb),
        extent=[
            float(self.x[0] - self.dx/2) / units,
            float(self.x[-1] + self.dx/2) / units,
            float(self.y[0] - self.dy/2)/ units,
            float(self.y[-1] + self.dy/2) / units,
        ],
        interpolation="spline36", origin = "lower"
    )
    plt.show()
