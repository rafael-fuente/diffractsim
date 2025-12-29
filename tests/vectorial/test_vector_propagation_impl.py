import unittest
import numpy as np
from diffractsim.fields.vector_field import VectorField
from diffractsim.propagation.vector_angular_spectrum import propagate_vector_angular_spectrum

class TestVectorPropagation(unittest.TestCase):

    def test_vector_propagation_runs(self):
        """Verify that propagation runs without raising NotImplementedError."""
        Ex = np.zeros((32, 32), dtype=complex)
        Ey = np.zeros((32, 32), dtype=complex)
        
        # Simple Gaussian
        x = np.linspace(-10e-6, 10e-6, 32)
        y = np.linspace(-10e-6, 10e-6, 32)
        XX, YY = np.meshgrid(x, y)
        Ex = np.exp(-(XX**2 + YY**2)/(2e-6)**2).astype(complex)

        field = VectorField(
            Ex=Ex,
            Ey=Ey,
            wavelength=500e-9,
            x=x,
            y=y
        )

        # Should not raise
        new_field = propagate_vector_angular_spectrum(field, 100e-6)
        
        self.assertIsInstance(new_field, VectorField)
        self.assertTrue(hasattr(new_field, 'Ez'))
        self.assertEqual(new_field.Ex.shape, Ex.shape)
        self.assertEqual(new_field.Ez.shape, Ex.shape)

    def test_divergence_free(self):
        """Verify that the propagated field satisfies div(E) = 0 in k-space."""
        N = 64
        L = 20e-6
        wavelength = 500e-9
        x = np.linspace(-L/2, L/2, N)
        y = np.linspace(-L/2, L/2, N)
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        
        # Create a random field to test robustness
        np.random.seed(42)
        Ex = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        Ey = np.random.randn(N, N) + 1j * np.random.randn(N, N)
        
        # Band-limit
        mask_freq = np.abs(np.fft.fftfreq(N))[:, None] < 0.2
        Ex = np.fft.ifft2(np.fft.fft2(Ex) * mask_freq).copy()
        Ey = np.fft.ifft2(np.fft.fft2(Ey) * mask_freq).copy()

        field = VectorField(Ex, Ey, wavelength, x, y)
        
        # Propagate
        z = 10e-6
        new_field = propagate_vector_angular_spectrum(field, z)
        
        # Check divergence in k-space
        fx = np.fft.fftfreq(N, dx)
        fy = np.fft.fftfreq(N, dy)
        KX, KY = np.meshgrid(2 * np.pi * fx, 2 * np.pi * fy)
        
        k = 2 * np.pi / wavelength
        kz = np.sqrt((k**2 - KX**2 - KY**2).astype(complex))
        
        Ex_k = np.fft.fft2(new_field.Ex)
        Ey_k = np.fft.fft2(new_field.Ey)
        Ez_k = np.fft.fft2(new_field.Ez)
        
        div_k = KX * Ex_k + KY * Ey_k + kz * Ez_k
        
        mask = np.abs(kz) > 1e-5
        if np.any(mask):
            max_div = np.max(np.abs(div_k[mask]))
            field_mag = np.max(np.abs(Ex_k))
            # Relative divergence should be small
            self.assertLess(max_div / (field_mag + 1e-15), 1e-7)

if __name__ == '__main__':
    unittest.main()