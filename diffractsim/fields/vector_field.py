"""
VectorField representation for electromagnetic simulations.

This module introduces a vectorial field abstraction without
modifying or interfering with existing scalar-field logic.
"""

import numpy as np

class VectorField:
    """
    Represents a monochromatic electromagnetic field using
    transverse electric field components.
    """

    def __init__(self, Ex, Ey, wavelength, x, y):
        self.Ex = np.asarray(Ex, dtype=complex)
        self.Ey = np.asarray(Ey, dtype=complex)
        self.wavelength = wavelength
        self.x = x
        self.y = y

        if self.Ex.shape != self.Ey.shape:
            raise ValueError("Ex and Ey must have identical shapes")

    @property
    def shape(self):
        return self.Ex.shape
