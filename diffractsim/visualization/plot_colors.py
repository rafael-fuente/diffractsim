import matplotlib.pyplot as plt
import numpy as np
from ..util.constants import *

"""

MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.

"""

def save_plot(self, rgb, figsize=(16, 16), xlim=None, ylim=None, text=None, path="pic.png", dark_background=True, tight=True):
    from ..util.backend_functions import backend as bd
    if bd != np:
        rgb = rgb.get()

    if dark_background:
        plt.style.use("dark_background")
    else:
        plt.style.use("default")

    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1, 1, 1)

    if xlim is not None:
        ax.set_xlim(np.array(xlim) / mm)

    if ylim is not None:
        ax.set_ylim(np.array(ylim) / mm)

    ax.set_xlabel("[mm]")
    ax.set_ylabel("[mm]")

    if text is None:
        ax.set_title("Screen distance = " + str(self.z * 100) + " cm")
    else:
        ax.set_title(text)

    im = ax.imshow(
        rgb,
        extent = [
            float(self.x[0] - self.dx / 2) / mm,
            float(self.x[-1] + self.dx / 2) / mm,
            float(self.y[0] - self.dy / 2) / mm,
            float(self.y[-1] + self.dy / 2) / mm,
        ],
        interpolation = "spline36", origin = "lower"
    )
    if tight:
        fig.savefig(path, bbox_inches = 'tight')
    else:
        fig.savefig(path)  # save the figure to file
    plt.close(fig)  # close the figure window


def plot_colors(self, rgb, figsize=(6, 6), xlim=None, ylim=None, text=None, units=mm, dark_background=True):
    """visualize the diffraction pattern colors with matplotlib"""

    from ..util.backend_functions import backend as bd
    if dark_background:
        plt.style.use("dark_background")
    else:
        plt.style.use("default")
    if bd != np:
        rgb = rgb.get()

    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(1, 1, 1)

    if xlim is not None:
        ax.set_xlim(np.array(xlim) / units)

    if ylim is not None:
        ax.set_ylim(np.array(ylim) / units)

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

    if text is None:
        ax.set_title("Screen distance = " + str(self.z * 100) + " cm")
    else:
        ax.set_title(text)

    im = ax.imshow(
        rgb,
        extent = [
            float(self.x[0] - self.dx / 2) / units,
            float(self.x[-1] + self.dx / 2) / units,
            float(self.y[0] - self.dy / 2) / units,
            float(self.y[-1] + self.dy / 2) / units,
        ],
        interpolation = "spline36", origin = "lower"
    )
    plt.show()
