from PIL import Image
import numpy as np
from pathlib import Path


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