from PIL import Image
import numpy as np
from pathlib import Path
from scipy.ndimage import map_coordinates
from scipy.interpolate import RegularGridInterpolator as _RGI


"""

MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.

"""

def load_graymap_image_as_array(string, new_size = None):

    from PIL import Image
    img = Image.open(Path(string))
    img = img.convert("RGB")
    if new_size != None:
        img = img.resize(new_size)

    imgRGB = np.asarray(img) / 255.0
    imgR = imgRGB[:, :, 0]
    imgG = imgRGB[:, :, 1]
    imgB = imgRGB[:, :, 2]
    t = np.array(0.2990 * imgR + 0.5870 * imgG + 0.1140 * imgB)
    t = np.flip(t, axis = 0)
    return t



def save_phase_mask_as_image(string, phase_mask, phase_mask_format = 'hsv'):

    if phase_mask_format =='hsv':

        from matplotlib.colors import hsv_to_rgb

        h = ((np.flip(phase_mask, axis = 0) + np.pi)  / (2 * np.pi))
        s = np.ones_like(h)
        v = np.ones_like(h)
        rgb = hsv_to_rgb(   np.moveaxis(np.array([h,s,v]) , 0, -1))

        img_RGB = [Image.fromarray((np.round(255 * rgb[:,:,0])).astype(np.uint8), "L"),
                   Image.fromarray((np.round(255 * rgb[:,:,1])).astype(np.uint8), "L"),
                   Image.fromarray((np.round(255 * rgb[:,:,2])).astype(np.uint8), "L")]

        img = Image.merge("RGB", img_RGB)
        img.save(Path(string))

    else:
        #save as grayscale
        h = ((np.flip(phase_mask, axis = 0) + np.pi)  / (2 * np.pi))
        img = Image.fromarray(np.uint8(h* 255) , 'L')
        img.save(Path(string))


    
def save_amplitude_mask_as_image(string, amplitude_mask):
    img = Image.fromarray(np.uint8( np.flip(amplitude_mask/np.amax(amplitude_mask), axis = 0)* 255) , 'L')
    img.save(Path(string))


def create_interpolator(
    data,
    *,
    x_interval=None,   # [xmin, xmax]
    y_interval=None,   # [ymin, ymax]
    method="linear",
    bounds_error=False,
    fill_value=0.0,
):
    """
    Return f(x, y) that interpolates *data*.

    Parameters
    ----------
    data : 2-D np.ndarray
        Sample values laid out as data[yi, xi].
    x_interval : iterable of length 2, optional
        [xmin, xmax] span associated with the x dimension (length nx).
        Defaults to pixel indices [0, Nx-1].
    y_interval : iterable of length 2, optional
        [ymin, ymax] span associated with the y dimension (length ny).
        Defaults to pixel indices [0, Ny-1].
    method : {"linear", "nearest"}, default "linear"
        Interpolation scheme passed to RegularGridInterpolator.
    bounds_error, fill_value : see scipy.interpolate.RegularGridInterpolator

    Returns
    -------
    f : callable
        *Scalar or array* = f(x, y).  Accepts
        1. two arrays of matching shape (broadcast like NumPy), or
        2. an (N, 2) array of point coordinates.
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError("`data` must be 2-D")

    ny, nx = data.shape

    if x_interval is None:
        x_interval = (0, nx - 1)
    if y_interval is None:
        y_interval = (0, ny - 1)

    if len(x_interval) != 2 or len(y_interval) != 2:
        raise ValueError("`x_interval` must be [xmin, xmax] and `y_interval` must be [ymin, ymax]")

    xmin, xmax = map(float, x_interval)
    ymin, ymax = map(float, y_interval)

    # --- Your original spacing/centering logic (kept exactly) ---
    # Using the intervals only to set spacing; grid is centered via (i - n//2)
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    xs = dx * (np.arange(nx) - nx // 2)
    ys = dy * (np.arange(ny) - ny // 2)
    # -------------------------------------------------------------

    # Note grid order: (y, x) — the same order as data[yi, xi]
    _interp = _RGI(
        (ys, xs),
        data,
        method=method,
        bounds_error=bounds_error,
        fill_value=fill_value,
    )

    def f(x, y=None):
        """Evaluate the interpolator."""
        if y is None:
            # Expect x to be (..., 2) array of points [[x, y], ...]
            pts = np.asarray(x, dtype=float)
            return _interp(pts)
        else:
            x = np.asarray(x, dtype=float)
            y = np.asarray(y, dtype=float)
            pts = np.stack([y, x], axis=-1)  # (..., 2)
            return _interp(pts)

    return f


def load_file_as_function(
    path,
    *,
    x_interval=None,   # [xmin, xmax]
    y_interval=None,   # [ymin, ymax]
    method="linear",
    bounds_error=False,
    fill_value=0.0):
    
    """
    Return f(x, y) that interpolates *data* from a file

    Parameters
    ----------
    path : string with the path to the 2-D np.ndarray
        Sample values laid out as data[yi, xi].
    x_interval : iterable of length 2, optional
        [xmin, xmax] span associated with the x dimension (length nx).
        Defaults to pixel indices [0, Nx-1].
    y_interval : iterable of length 2, optional
        [ymin, ymax] span associated with the y dimension (length ny).
        Defaults to pixel indices [0, Ny-1].
    method : {"linear", "nearest"}, default "linear"
        Interpolation scheme passed to RegularGridInterpolator.
    bounds_error, fill_value : see scipy.interpolate.RegularGridInterpolator

    Returns
    -------
    f : callable
        *Scalar or array* = f(x, y).  Accepts
        1. two arrays of matching shape (broadcast like NumPy), or
        2. an (N, 2) array of point coordinates.
    """

    
    return create_interpolator(np.load(path), x_interval=x_interval,  y_interval=y_interval,  method=method,bounds_error=bounds_error,fill_value=fill_value)




def load_phase_as_function(    
    path,
    *,
    x_interval=None,   # [xmin, xmax]
    y_interval=None,   # [ymin, ymax]
    method="linear",
    bounds_error=False,
    fill_value=0.0):
    """
    Return f(x, y) that interpolates phas data from a file

    Parameters
    ----------
    path : string with the path to the 2-D np.ndarray
        Sample values laid out as data[yi, xi].
    x_interval : iterable of length 2, optional
        [xmin, xmax] span associated with the x dimension (length nx).
        Defaults to pixel indices [0, Nx-1].
    y_interval : iterable of length 2, optional
        [ymin, ymax] span associated with the y dimension (length ny).
        Defaults to pixel indices [0, Ny-1].
    method : {"linear", "nearest"}, default "linear"
        Interpolation scheme passed to RegularGridInterpolator.
    bounds_error, fill_value : see scipy.interpolate.RegularGridInterpolator

    Returns
    -------
    f : callable
        *Scalar or array* = f(x, y).  Accepts
        1. two arrays of matching shape (broadcast like NumPy), or
        2. an (N, 2) array of point coordinates.
    """

    sin_fun = create_interpolator(np.sin(np.load(path)),  x_interval = x_interval,   y_interval = y_interval, method=method,bounds_error=bounds_error,fill_value=fill_value)
    cos_fun = create_interpolator(np.cos(np.load(path)),  x_interval = x_interval,   y_interval = y_interval, method=method,bounds_error=bounds_error,fill_value=fill_value)

    def phase_function(xx,yy):
        return np.arctan2(sin_fun(xx,yy), cos_fun(xx,yy))
        
    return phase_function
