import numpy as np
from ..util.backend_functions import backend as bd
import numpy as np
from .diffractive_element import DOE
from PIL import Image,ImageDraw

class HexagonalAperture(DOE):
    def __init__(self, radius):
        """
        Creates a hexagonal slit
        """
        global bd
        from ..util.backend_functions import backend as bd
        self.radius = radius # circumscribed circle radius

    def get_transmittance(self, xx, yy, λ):


        Ny,Nx = xx.shape
        dx = xx[0,1]-xx[0,0]
        dy = yy[1,0]-yy[0,0]

        img = Image.new("RGB", (Nx, Ny))

        vertex = [(self.radius/dx*np.cos(phi) + Nx//2, self.radius/dy*np.sin(phi) + Ny//2) for phi in np.arange(0,2*np.pi, 2*np.pi/6)]
        img_draw = ImageDraw.Draw(img) 
        img_draw.polygon(vertex, fill ="white")
        imgRGB = np.asarray(img) / 255.0
        t = 0.2990 * imgRGB[:, :, 0] + 0.5870 * imgRGB[:, :, 1] + 0.1140 * imgRGB[:, :, 2]
        t = bd.array(np.flip(t, axis = 0))


        return t
