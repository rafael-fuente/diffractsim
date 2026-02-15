import numpy as np
from diffractsim.monochromatic_simulator import MonochromaticField
from diffractsim.light_sources.plane_wave import PlaneWave
from diffractsim.diffractive_elements.jones_element import JonesElement

def _make_field(vectorial=True, intensity=1.0):
    return MonochromaticField(
        wavelength=532e-9, extent_x=32e-6, extent_y=32e-6,
        Nx=32, Ny=32, intensity=intensity, vectorial=vectorial)

def test_linear_polarizer_extinction():
    F = _make_field(vectorial=True, intensity=1.0)
    src = PlaneWave(Ex_amplitude=1.0, Ey_amplitude=0.0, phase_diff=0.0)
    F.add(src)
    J = np.array([[0, 0], [0, 1]], dtype=complex)
    pol = JonesElement(J)
    F.add(pol)
    I = F.get_intensity()
    assert np.allclose(I, 0.0, atol=1e-12)

def test_qwp_adds_pi_over_2_phase():
    F = _make_field(vectorial=True, intensity=1.0)
    src = PlaneWave(Ex_amplitude=1.0, Ey_amplitude=1.0, phase_diff=0.0)
    F.add(src)
    J = np.array([[1, 0], [0, 1j]], dtype=complex)
    qwp = JonesElement(J)
    F.add(qwp)
    ex = F.Ex.flat[0]
    ey = F.Ey.flat[0]
    assert np.allclose(ey / ex, 1j, atol=1e-12)

def test_scalar_backward_compatibility():
    Fs = _make_field(vectorial=False)
    src = PlaneWave()
    Fs.add(src)
    I = Fs.get_intensity()
    assert I.shape == (Fs.Ny, Fs.Nx)
    assert np.isfinite(I).all()
