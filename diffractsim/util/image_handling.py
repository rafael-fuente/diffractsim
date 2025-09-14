from PIL import Image
import numpy as np
from scipy.ndimage import map_coordinates
from pathlib import Path


"""

MPL 2.0 License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.

"""


def rescale_img_to_custom_coordinates(img, image_size, extent_x,extent_y, Nx, Ny):

    img_pixels_width, img_pixels_height = img.size

    if image_size != None:
        new_img_pixels_width, new_img_pixels_height = int(np.round(image_size[0] / extent_x  * Nx)),  int(np.round(image_size[1] / extent_y  * Ny))
    else:
        #by default, the image fills the entire aperture plane
        new_img_pixels_width, new_img_pixels_height = Nx, Ny

    img = img.resize((new_img_pixels_width, new_img_pixels_height))

    dst_img = Image.new("RGB", (Nx, Ny), "black" )
    dst_img_pixels_width, dst_img_pixels_height = dst_img.size

    Ox, Oy = (dst_img_pixels_width-new_img_pixels_width)//2, (dst_img_pixels_height-new_img_pixels_height)//2
    
    dst_img.paste( img , box = (Ox, Oy ))
    return dst_img


def convert_graymap_image_to_hsvmap_image(img):
    imgRGB = np.asarray(img) / 255.0
    imgR = imgRGB[:, :, 0]
    imgG = imgRGB[:, :, 1]
    imgB = imgRGB[:, :, 2]
    graymap_array = np.array(0.2990 * imgR + 0.5870 * imgG + 0.1140 * imgB)

    from matplotlib.colors import hsv_to_rgb

    h = graymap_array
    s = np.ones_like(h)
    v = np.ones_like(h)
    rgb = hsv_to_rgb(   np.moveaxis(np.array([h,s,v]) , 0, -1))

    img_RGB = [Image.fromarray((np.round(255 * rgb[:,:,0])).astype(np.uint8), "L"),
               Image.fromarray((np.round(255 * rgb[:,:,1])).astype(np.uint8), "L"),
               Image.fromarray((np.round(255 * rgb[:,:,2])).astype(np.uint8), "L")]

    return Image.merge("RGB", img_RGB)



def resize_array(img_array, new_shape):

    Ny, Nx = img_array.shape
    
    from scipy.interpolate import RectBivariateSpline
    fun = RectBivariateSpline(
        np.linspace(0, 1, Nx),
        np.linspace(0, 1, Ny),
        img_array)
    resize_img_array = fun(np.linspace(0, 1, new_shape[1]), np.linspace(0, 1, new_shape[0]))
    return resize_img_array






def load_image_as_array(path):

    """Load *path* into a 2‑D float array.

    The function performs **three** independent preprocessing steps:

    1. Read the file with *Pillow* and convert to 8‑bit sRGB.
    2. Convert RGB to luminance (*Y*) using the ITU‑R BT.601 weights
       (0.299R+0.587G+0.114B).

    Parameters
    ----------
    path
        Path to any raster image supported by *Pillow* (PNG, JPEG, TIFF, …).
    image_size
        ``(width, height)`` tuple to which the image should be resized.  ``None``
        (default) keeps the original resolution.

    Returns
    -------
    np.ndarray
        2‑D array of shape ``(height, width)`` with values in the range
        :math:`[0, 1]`.

    Examples
    --------
    >>> from pathlib import Path
    >>> img = load_image("input/photo.jpg")
    
    """
    img = Image.open(Path(path))
    img = img.convert("RGB")
    imgRGB = np.asarray(img) / 255.0

    t = 0.2990 * imgRGB[:, :, 0] + 0.5870 * imgRGB[:, :, 1] + 0.1140 * imgRGB[:, :, 2]
    t = np.array(np.flip(t, axis = 0))
    
    return t





def load_image_as_function(
    path,
    *,
    x_interval=None,   # [xmin, xmax]
    y_interval=None,   # [ymin, ymax]
    method="linear",
    bounds_error=False,
    fill_value=0.0,
):
    """
    Return f(x, y) that interpolates *data* using scipy.ndimage.map_coordinates.

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
        Interpolation scheme: order=1 (linear) or order=0 (nearest).
    bounds_error : bool, default False
        If True, raise a ValueError when evaluating outside the domain implied
        by the centered grid (see Notes).
        If False, samples outside the domain use `fill_value` if provided,
        otherwise edge values are used (nearest-like).
    fill_value : float or None, default None
        Value to use outside the domain when bounds_error is False.
        If None, falls back to nearest-edge behavior (since ndimage does not
        do linear extrapolation like RGI).

    Notes
    -----
    This keeps your original spacing/centering logic:
        dx = (xmax - xmin) / nx
        xs = dx * (i - nx//2)   for i = 0..nx-1
    and similarly for y. That means the interpolation *domain* is
    [xs[0], xs[-1]] × [ys[0], ys[-1]], which is generally centered near 0 and
    not exactly [xmin, xmax] × [ymin, ymax]. `x_interval`/`y_interval` only set
    the pixel spacing via dx, dy.

    Returns
    -------
    f : callable
        *Scalar or array* = f(x, y). Accepts
        1) two arrays of matching/broadcastable shape (NumPy broadcasting), or
        2) an (..., 2) array of point coordinates [[x, y], ...].
    """
    data = load_image_as_array(path)
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

    # Interpolation order from method
    if method == "linear":
        order = 1
    elif method == "nearest":
        order = 0
    else:
        raise ValueError("`method` must be 'linear' or 'nearest'")

    # --- Your original spacing/centering logic (preserved) ---
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny
    # World-domain edges implied by the centered grid
    x_lo = dx * (0 - nx // 2)
    x_hi = dx * ((nx - 1) - nx // 2)
    y_lo = dy * (0 - ny // 2)
    y_hi = dy * ((ny - 1) - ny // 2)
    # ---------------------------------------------------------

    # Convert world (x, y) to index-space (ix, iy) for ndimage
    # Using: world_x = dx*(i - nx//2)  =>  i = (world_x - x_lo)/dx
    def _to_indices(x, y):
        ix = (np.asarray(x, dtype=float) - x_lo) / dx
        iy = (np.asarray(y, dtype=float) - y_lo) / dy
        return ix, iy

    def _check_bounds(px, py):
        if bounds_error:
            oob = (px < x_lo) | (px > x_hi) | (py < y_lo) | (py > y_hi)
            if np.any(oob):
                raise ValueError("Points outside the interpolation domain")

    # Choose ndimage edge handling
    if bounds_error:
        mode = "nearest"   # won't be used if we raise; safe placeholder
        cval = 0.0
    else:
        if fill_value is None:
            # RGI(None) extrapolates; ndimage can't. Approximate with edge values.
            mode = "nearest"
            cval = 0.0
        else:
            mode = "constant"
            cval = float(fill_value)

    def _eval(ix, iy):
        coords = np.vstack([iy.ravel(), ix.ravel()])  # (2, N) with order (y, x)
        out = map_coordinates(
            data,
            coords,
            order=order,
            mode=mode,
            cval=cval,
            prefilter=False,  # fine (and faster) for order 0/1
        )
        return out.reshape(ix.shape)

    def f(x, y=None):
        """Evaluate the interpolator."""
        if y is None:
            pts = np.asarray(x, dtype=float)
            if pts.shape[-1] != 2:
                raise ValueError("When y is None, x must be an array of [..., 2] points")
            px = pts[..., 0]
            py = pts[..., 1]
            _check_bounds(px, py)
            ix, iy = _to_indices(px, py)
            return _eval(ix, iy)
        else:
            px = np.asarray(x, dtype=float)
            py = np.asarray(y, dtype=float)
            _check_bounds(px, py)
            px, py = np.broadcast_arrays(px, py)
            ix, iy = _to_indices(px, py)
            return _eval(ix, iy)

    return f




