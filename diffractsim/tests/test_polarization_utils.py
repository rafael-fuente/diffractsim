import unittest
from diffractsim.util.polarization_utils import polarization_state

class TestPolarizationUtils(unittest.TestCase):
    def test_polarization_state(self):
        # Test s-polarized state
        self.assertEqual(polarization_state('s'), 's')
        
        # Test p-polarized state
        self.assertEqual(polarization_state('p'), 'p')
        
        # Test circularly polarized state
        self.assertEqual(polarization_state('circular'), 'circular')

if __name__ == '__main__':
    unittest.main()