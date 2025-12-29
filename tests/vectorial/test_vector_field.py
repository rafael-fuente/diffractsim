import numpy as np
from diffractsim.fields.vector_field import VectorField

def test_vector_field_initialization():
    Ex = np.ones((32, 32), dtype=complex)
    Ey = np.zeros((32, 32), dtype=complex)

    field = VectorField(
        Ex=Ex,
        Ey=Ey,
        wavelength=500e-9,
        x=np.linspace(-1,1,32),
        y=np.linspace(-1,1,32)
    )

    assert field.Ex.shape == field.Ey.shape
