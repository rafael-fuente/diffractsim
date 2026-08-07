import numpy as np
from ..util.backend_functions import backend as bd
from .diffractive_element import DOE

class LinearPolarizer(DOE):
    def __init__(self, angle):
        """
        Creates a linear polarizer at a given angle (in degrees)
        """
        self.angle = angle
        self.theta = np.radians(angle)

    def apply_to_vectorial_field(self, field):
        """
        Applies the Jones matrix of a linear polarizer to the vectorial field
        """
        c = bd.cos(self.theta)
        s = bd.sin(self.theta)
        
        Ex_new = (c**2) * field.Ex + (s*c) * field.Ey
        Ey_new = (s*c) * field.Ex + (s**2) * field.Ey
        
        field.Ex = Ex_new
        field.Ey = Ey_new
        field.Ez = bd.zeros_like(field.Ez) # Polarizer is usually a planar element

    def get_transmittance(self, xx, yy, λ):
        # By default, a polarizer doesn't change the amplitude of a scalar field 
        # (or it could be argued it should reduce it by 50%, but scalar fields don't have polarization info)
        return bd.ones_like(xx)
