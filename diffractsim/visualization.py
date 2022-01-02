import matplotlib.pyplot as plt
import numpy as np

m = 1.
cm = 1e-2
mm = 1e-3
um = 1e-6
nm = 1e-9
W = 1


def plot_colors(self, rgb, figsize=(6, 6), xlim=None, ylim=None, text = None):
    """visualize the diffraction pattern colors with matplotlib"""

    from .util.backend_functions import backend as bd
    plt.style.use("dark_background")
    if bd != np:
        rgb = rgb.get()

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    if xlim != None:
        ax.set_xlim(np.array(xlim)/mm)

    if ylim != None:
        ax.set_ylim(np.array(ylim)/mm)

    # we use mm by default
    ax.set_xlabel("[mm]")
    ax.set_ylabel("[mm]")

    if text == None:
        ax.set_title("Screen distance = " + str(self.z * 100) + " cm")
    else: 
        ax.set_title(text)

    im = ax.imshow(
        (rgb),
        extent=[
            float(self.x[0]) / mm,
            float(self.x[-1] + self.dx) / mm,
            float(self.y[0] )/ mm,
            float(self.y[-1] + self.dy) / mm,
        ],
        interpolation="spline36", origin = "lower"
    )
    plt.show()



def plot_intensity(self, square_root = False, figsize=(7, 6), xlim=None, ylim=None, grid = False, text = None):
    """visualize the diffraction pattern intesity with matplotlib"""
    
    from .util.backend_functions import backend as bd
    plt.style.use("dark_background")

    if square_root == False:
        if bd != np:
            print("hola")
            I = self.I.get()
        else:
            I = self.I

    else:
        if bd != np:
            I = np.sqrt(self.I.get())
        else:
            I = np.sqrt(self.I)


    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)

    if grid == True:
        ax.grid(alpha =0.2)

    if xlim != None:
        ax.set_xlim(np.array(xlim)/mm)

    if ylim != None:
        ax.set_ylim(np.array(ylim)/mm)

    # we use mm by default
    ax.set_xlabel("[mm]")
    ax.set_ylabel("[mm]")


    if text == None:
        ax.set_title("Screen distance = " + str(self.z * 100) + " cm")
    else: 
        ax.set_title(text)


    im = ax.imshow(
        I, cmap= 'inferno',
        extent=[
            float(self.x[0]) / mm,
            float(self.x[-1] + self.dx) / mm,
            float(self.y[0] )/ mm,
            float(self.y[-1] + self.dy) / mm,
        ],
        interpolation="spline36", origin = "lower"
    )

    cb = fig.colorbar(im, orientation = 'vertical')

    if square_root == False:
        cb.set_label(r'Intensity $\left[W / m^2 \right]$', fontsize=13, labelpad =  14 )
    else:
        cb.set_label(r'Square Root Intensity $\left[ \sqrt{W / m^2 } \right]$', fontsize=13, labelpad =  14 )


    plt.show()



def plot_longitudinal_profile_colors(self, longitudinal_profile_rgb, start_distance, end_distance, xlim=None, ylim=None):
    """visualize the diffraction pattern longitudinal profile colors with matplotlib"""

    from .util.backend_functions import backend as bd
    plt.style.use("dark_background")

    if bd != np:
        longitudinal_profile_rgb = longitudinal_profile_rgb.transpose(1,0,2).get()
    else:
        longitudinal_profile_rgb = longitudinal_profile_rgb.transpose(1,0,2)

    fig = plt.figure(figsize=(16/9 *6,6)) 
    ax1 = fig.add_subplot(1,1,1)  
        
    ax1.set_xlabel('Screen Distance [cm]')
    ax1.set_title("Longitudinal Profile")
    ax1.set_ylabel("[mm]")
    if xlim != None:
        ax1.set_xlim(np.array(xlim)/cm)

    if ylim != None:
        ax1.set_ylim(np.array(ylim)/mm)

    im = ax1.imshow(longitudinal_profile_rgb,  extent = [start_distance/cm,  end_distance/cm, float(self.x[0]) / mm, float(self.x[-1] + self.dx) / mm],  interpolation='spline36', aspect = 'auto')
    plt.show()


def plot_longitudinal_profile_intensity(self,  longitudinal_profile_E, start_distance, end_distance, square_root = False, grid = False, xlim=None, ylim=None):
    """visualize the diffraction pattern longitudinal profile intensity with matplotlib"""

    from .util.backend_functions import backend as bd
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
        ax1.set_xlim(np.array(xlim)/cm)

    if ylim != None:
        ax1.set_ylim(np.array(ylim)/mm)

    fig = plt.figure(figsize=(16/9 *6,6)) 
    ax1 = fig.add_subplot(1,1,1)  


    ax1.set_ylabel("[mm]")
    ax1.set_xlabel('Screen Distance [cm]')
    ax1.set_title("Longitudinal Profile")
    if grid == True:
        ax1.grid(alpha =0.2)

    im = ax1.imshow(I, cmap= 'inferno',  extent = [start_distance/cm,  end_distance/cm, float(self.x[0]) / mm, float(self.x[-1]+ self.dx) / mm],  interpolation='spline36', aspect = 'auto')
    
    cb = fig.colorbar(im, orientation = 'vertical')

    if square_root == False:
        cb.set_label(r'Intensity $\left[W / m^2 \right]$', fontsize=13, labelpad =  14 )
    else:
        cb.set_label(r'Square Root Intensity $\left[ \sqrt{W / m^2 } \right]$', fontsize=13, labelpad =  14 )


    plt.show()

