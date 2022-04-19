from matplotlib.colors import hsv_to_rgb
import numpy as np

"""

MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.

"""

def complex_to_rgba(Z: np.ndarray, max_val: float = 1.0) -> np.ndarray:
    r = np.abs(Z)
    arg = np.angle(Z)
    
    h = (arg + np.pi)  / (2 * np.pi)
    s = np.ones(h.shape)
    v = np.ones(h.shape)  #alpha
    rgb = hsv_to_rgb(   np.moveaxis(np.array([h,s,v]) , 0, -1)  ) # --> tuple

    abs_z = np.abs(Z)/ max_val
    abs_z = np.where(abs_z> 1., 1. ,abs_z)
    return np.concatenate((rgb, abs_z.reshape((*abs_z.shape,1))), axis= (abs_z.ndim))
