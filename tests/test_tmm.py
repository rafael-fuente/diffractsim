"""
Unit tests for Transfer Matrix Method (TMM) implementation.

MPL 2.0 Clause License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from diffractsim.tmm import Layer, Stack, fresnel_coefficients
from diffractsim.tmm.materials import Material, DrudeMaterial
from diffractsim.util.constants import nm


class TestFresnelCoefficients:
    """Test Fresnel coefficient calculations."""
    
    def test_normal_incidence_air_glass(self):
        """Test Fresnel coefficients for air-glass interface at normal incidence."""
        n1 = 1.0  # Air
        n2 = 1.5  # Glass
        θ1 = 0.0  # Normal incidence
        
        r_s, t_s, θ2_s = fresnel_coefficients(n1, n2, θ1, "s")
        r_p, t_p, θ2_p = fresnel_coefficients(n1, n2, θ1, "p")
        
        # At normal incidence, s and p should be identical
        assert np.isclose(r_s, r_p, rtol=1e-10)
        assert np.isclose(t_s, t_p, rtol=1e-10)
        
        # Analytical formula: r = (n1 - n2) / (n1 + n2)
        r_expected = (n1 - n2) / (n1 + n2)
        assert np.isclose(r_s, r_expected, rtol=1e-10)
        
        # t = 2*n1 / (n1 + n2)
        t_expected = 2 * n1 / (n1 + n2)
        assert np.isclose(t_s, t_expected, rtol=1e-10)
    
    def test_energy_conservation_lossless(self):
        """Test energy conservation for lossless media."""
        n1 = 1.0
        n2 = 1.5
        θ1 = np.pi / 6  # 30 degrees
        
        r_s, t_s, θ2_s = fresnel_coefficients(n1, n2, θ1, "s")
        r_p, t_p, θ2_p = fresnel_coefficients(n1, n2, θ1, "p")
        
        # For lossless media, R + T should equal 1
        # R = |r|², T = |t|² * (n2*cos(θ2)) / (n1*cos(θ1))
        cos_θ1 = np.cos(θ1)
        cos_θ2_s = np.cos(θ2_s)
        cos_θ2_p = np.cos(θ2_p)
        
        R_s = np.abs(r_s)**2
        T_s = np.abs(t_s)**2 * (n2 * cos_θ2_s) / (n1 * cos_θ1)
        
        R_p = np.abs(r_p)**2
        T_p = np.abs(t_p)**2 * (n2 * cos_θ2_p) / (n1 * cos_θ1)
        
        # Energy conservation: R + T = 1 (within numerical precision)
        assert np.isclose(R_s + T_s, 1.0, rtol=1e-6)
        assert np.isclose(R_p + T_p, 1.0, rtol=1e-6)
    
    def test_total_internal_reflection(self):
        """Test total internal reflection (glass to air)."""
        n1 = 1.5  # Glass
        n2 = 1.0  # Air
        θ1 = np.arcsin(n2 / n1) + 0.1  # Just above critical angle
        
        r_s, t_s, _ = fresnel_coefficients(n1, n2, θ1, "s")
        
        # For TIR, |r| = 1
        assert np.isclose(np.abs(r_s), 1.0, rtol=1e-6)


class TestLayer:
    """Test Layer class."""
    
    def test_constant_index(self):
        """Test layer with constant refractive index."""
        layer = Layer(n=1.5, k=0.0)
        assert np.isclose(layer.get_index(500*nm), 1.5)
        assert np.isclose(layer.get_index(600*nm), 1.5)
    
    def test_dispersive_index(self):
        """Test layer with wavelength-dependent index."""
        def n_func(λ):
            # Simple dispersion: n = 1.5 + 0.01/λ (in meters)
            return 1.5 + 0.01 / λ
        
        layer = Layer(n=n_func)
        n1 = layer.get_index(500*nm)
        n2 = layer.get_index(600*nm)
        
        # Compare real parts (for lossless media)
        assert np.real(n1) > np.real(n2)  # Shorter wavelength has higher index (normal dispersion)
    
    def test_lossy_medium(self):
        """Test lossy medium (complex index)."""
        layer = Layer(n=1.5, k=0.1)
        n_complex = layer.get_index(500*nm)
        
        assert np.isclose(np.real(n_complex), 1.5)
        assert np.isclose(np.imag(n_complex), 0.1)


class TestStack:
    """Test Stack class and TMM calculations."""
    
    def test_single_interface_air_glass(self):
        """Test single interface (air-glass)."""
        air = Layer(n=1.0, name="air")
        glass = Layer(n=1.5, name="glass")
        
        stack = Stack([air, glass])
        result = stack.solve(500*nm, θ_incident=0.0, polarization="s")
        
        # At normal incidence, R = ((n1-n2)/(n1+n2))²
        R_expected = ((1.0 - 1.5) / (1.0 + 1.5))**2
        assert np.isclose(result['R'], R_expected, rtol=1e-6)
        
        # Energy conservation for lossless
        assert np.isclose(result['R'] + result['T'], 1.0, rtol=1e-6)
        assert np.isclose(result['A'], 0.0, atol=1e-10)
    
    def test_ar_coating_quarter_wave(self):
        """Test quarter-wave anti-reflection coating."""
        # Single-layer AR coating: n_coating = sqrt(n_air * n_glass)
        # For air (n=1) and glass (n=1.5), optimal n_coating = sqrt(1.5) ≈ 1.225
        λ0 = 500 * nm
        n_coating = np.sqrt(1.0 * 1.5)
        d_coating = λ0 / (4 * n_coating)  # Quarter-wave thickness
        
        air = Layer(n=1.0, name="air")
        coating = Layer(n=n_coating, d=d_coating, name="coating")
        glass = Layer(n=1.5, name="glass")
        
        stack = Stack([air, coating, glass])
        result = stack.solve(λ0, θ_incident=0.0, polarization="s")
        
        # At design wavelength, reflectance should be minimized
        # For ideal quarter-wave coating: R = ((n_air - n_glass²/n_coating) / (n_air + n_glass²/n_coating))²
        # This should be very small
        assert result['R'] < 0.01  # Reflectance < 1%
    
    def test_energy_conservation_multilayer(self):
        """Test energy conservation for multilayer stack."""
        # Three-layer stack: air - coating - glass
        air = Layer(n=1.0, name="air")
        coating = Layer(n=1.3, d=100*nm, name="coating")
        glass = Layer(n=1.5, name="glass")
        
        stack = Stack([air, coating, glass])
        result = stack.solve(500*nm, θ_incident=0.0, polarization="s")
        
        # Energy conservation: R + T + A = 1
        # For lossless media, A should be ~0
        energy_sum = result['R'] + result['T'] + result['A']
        assert np.isclose(energy_sum, 1.0, rtol=1e-3)
    
    def test_glass_metal_interface(self):
        """Test glass-metal interface (lossy)."""
        glass = Layer(n=1.5, name="glass")
        # Simple metal: n = 0.1 + 3j (typical for visible light)
        metal = Layer(n=0.1, k=3.0, name="metal")
        
        stack = Stack([glass, metal])
        result = stack.solve(500*nm, θ_incident=0.0, polarization="s")
        
        # For lossy media, R + T + A = 1
        energy_sum = result['R'] + result['T'] + result['A']
        assert np.isclose(energy_sum, 1.0, rtol=0.02)  # Allow 2% error for lossy
        assert result['A'] > 0.0  # Should have absorption
    
    def test_lossy_metal_film(self):
        """Test lossy metal film."""
        # Air - metal - air
        air = Layer(n=1.0, name="air")
        # Simple metal: n = 0.1 + 3j (typical for visible light)
        metal = Layer(n=0.1, k=3.0, d=50*nm, name="metal")
        air2 = Layer(n=1.0, name="air")
        
        stack = Stack([air, metal, air2])
        result = stack.solve(500*nm, θ_incident=0.0, polarization="s")
        
        # For lossy media, R + T + A = 1, and A > 0
        energy_sum = result['R'] + result['T'] + result['A']
        assert np.isclose(energy_sum, 1.0, rtol=0.02)  # Allow 2% error for lossy
        assert result['A'] > 0.0  # Should have absorption


class TestPolarizers:
    """Test polarizer and retarder Jones matrices."""
    
    def test_malus_law(self):
        """Test Malus law with ideal polarizers."""
        from diffractsim.vectorial_field import VectorialField
        from diffractsim.tmm.thin_elements import IdealPolarizer
        
        # Create unpolarized field (equal Ex and Ey)
        Ex = np.ones((10, 10), dtype=complex)
        Ey = np.ones((10, 10), dtype=complex)
        field = VectorialField(Ex, Ey, 500*nm, 1e-6, 1e-6)
        
        # First polarizer at 0°
        pol1 = IdealPolarizer(angle=0.0)
        field1 = pol1.apply(field)
        
        # Second polarizer (analyzer) at angle θ
        angles = np.linspace(0, np.pi/2, 10)
        intensities = []
        
        for θ in angles:
            pol2 = IdealPolarizer(angle=θ)
            field2 = pol2.apply(field1)
            I = np.mean(field2.intensity())
            intensities.append(I)
        
        # Malus law: I(θ) = I(0) * cos²(θ)
        I0 = intensities[0]
        for i, θ in enumerate(angles):
            I_expected = I0 * np.cos(θ)**2
            I_actual = intensities[i]
            # Allow 1% error as specified in issue
            assert np.abs(I_actual - I_expected) / I0 < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

