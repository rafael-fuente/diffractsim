import numpy as np

class JonesElement:
    def __init__(self, J):
        self.J = np.asarray(J, dtype=complex)
        if self.J.shape != (2, 2):
            raise ValueError('J must be 2x2')

    def get_E_components(self, Ex, Ey, xx, yy, lam):
        shape = Ex.shape
        v = np.vstack([np.asarray(Ex).reshape(-1), np.asarray(Ey).reshape(-1)])
        v2 = self.J @ v
        return v2[0].reshape(shape), v2[1].reshape(shape)
