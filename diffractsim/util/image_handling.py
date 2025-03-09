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



from scipy.interpolate import RegularGridInterpolator

def resize_array(img_array, new_shape):
    Ny, Nx = img_array.shape
    x = np.linspace(0, 1, Nx)
    y = np.linspace(0, 1, Ny)
    
    interpolator = RegularGridInterpolator((y, x), img_array, method="cubic", bounds_error=False, fill_value=0)
    
    new_x = np.linspace(0, 1, new_shape[1])
    new_y = np.linspace(0, 1, new_shape[0])
    new_grid = np.meshgrid(new_y, new_x, indexing='ij')
    
    resize_img_array = interpolator(np.array([new_grid[0].flatten(), new_grid[1].flatten()]).T).reshape(new_shape)
    
    return resize_img_array

