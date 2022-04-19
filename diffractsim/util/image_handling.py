from PIL import Image
import numpy as np


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
    
    from scipy.interpolate import interp2d
    fun = interp2d(
        np.linspace(0, 1, Nx),
        np.linspace(0, 1, Ny),
        img_array,
        kind="cubic",
    )
    resize_img_array = fun(np.linspace(0, 1, new_shape[1]), np.linspace(0, 1, new_shape[0]))
    return resize_img_array
