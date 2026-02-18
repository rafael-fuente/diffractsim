import unittest
from vectorial_field import VectorialField

class TestVectorialField(unittest.TestCase):
    def test_init(self):
        field = VectorialField(wavelength=1e-6, extent_x=10e-3, extent_y=10e-3, Nx=100, Ny=100)
        self.assertTrue(isinstance(field, VectorialField))
        self.assertEqual(field.wavelength, 1e-6)
        self.assertEqual(field.extent_x, 10e-3)
        self.assertEqual(field.extent_y, 10e-3)
        self.assertEqual(field.Nx, 100)
        self.assertEqual(field.Ny, 100)

    def test_get_intensity(self):
        field = VectorialField(wavelength=1e-6, extent_x=10e-3, extent_y=10e-3, Nx=100, Ny=100)
        intensity = field.get_intensity()
        expected_shape = (field.Nx, field.Ny)
        self.assertEqual(intensity.shape, expected_shape)

if __name__ == '__main__':
    unittest.main()

def test_get_intensity(self):
    field = VectorialField(wavelength=1e-6, extent_x=10e-3, extent_y=10e-3, Nx=100, Ny=100)
    intensity = field.get_intensity()
    self.assertEqual(intensity.shape, (field.Nx, field.Ny))
    expected_max_intensity = 1.0  # Assuming the maximum intensity is 1.0
    self.assertTrue(np.all(intensity <= expected_max_intensity))
    self.assertTrue(np.any(intensity > 0))  # Ensure there are non-zero intensities
