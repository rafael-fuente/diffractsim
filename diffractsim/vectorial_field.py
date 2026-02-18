import numpy as np

class VectorialField:
    def __init__(self, x, y, z, Ex=None, Ey=None):
        self.x = x
        self.y = y
        self.z = z
        self.Ex = Ex
        self.Ey = Ey

    def magnitude(self):
        return (self.x**2 + self.y**2 + self.z**2)**0.5

    def add(self, other):
        return VectorialField(self.x + other.x, self.y + other.y, self.z + other.z)

    def subtract(self, other):
        return VectorialField(self.x - other.x, self.y - other.y, self.z - other.z)

    def scale(self, scalar):
        return VectorialField(self.x * scalar, self.y * scalar, self.z * scalar)

    def dot_product(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross_product(self, other):
        return VectorialField(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def __str__(self):
        return f"VectorialField({self.x}, {self.y}, {self.z})"

def initialize_fields(self, Ex=None, Ey=None):
        if Ex is None:
            Ex = np.zeros_like(self.x)
        if Ey is None:
            Ey = np.zeros_like(self.y)
        self.Ex = Ex
        self.Ey = Ey

def propagate_via_angular_spectrum_method(self, wavelength, kx, ky):
        # Angular spectrum method implementation here
        pass