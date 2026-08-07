# Vectorial Electromagnetic Field Support

This extension adds vectorial EM field representation to `diffractsim`, enabling simulation of polarization effects in optical systems.

## Installation

Install diffractsim with all dependencies:

```bash
pip install -e .
```

## Quick Start

```python
from diffractsim import VectorialMonochromaticField, mm, cm, nm
from diffractsim.diffractive_elements import CircularAperture

# Create vectorial field
F = VectorialMonochromaticField(
    wavelength=632.8*nm,  # HeNe laser
    extent_x=4*mm,
    extent_y=4*mm,
    Nx=500,
    Ny=500
)

# Set polarization state
F.set_linear_polarization(angle=0)  # Horizontal polarization

# Add optical element
F.add(CircularAperture(radius=0.5*mm))

# Propagate
F.propagate(50*cm)

# Get results
intensity = F.get_intensity()
S0, S1, S2, S3 = F.get_stokes_parameters()
```

## Features

### Polarization States

**Linear polarization:**
```python
F.set_linear_polarization(angle=np.pi/4)  # 45° linear
```

**Circular polarization:**
```python
F.set_circular_polarization(handedness='right')  # or 'left'
```

**Elliptical polarization:**
```python
F.set_elliptical_polarization(a=1.0, b=0.5, angle=0)
```

### Polarization Analysis

**Stokes parameters:**
```python
S0, S1, S2, S3 = F.get_stokes_parameters()
# S0: total intensity
# S1: linear H-V
# S2: linear ±45°
# S3: circular R-L
```

**Jones vector:**
```python
jones = F.get_jones_vector()  # [Ex, Ey] at center
```

**Degree of polarization:**
```python
DOP = F.get_degree_of_polarization()
```

## Examples

See `examples/` directory:
- `polarization_linear.py` - Linear polarization through circular aperture
- `polarization_circular.py` - Circular polarization through rectangular slit

## Applications

- Fusion plasma diagnostics (Thomson scattering polarimetry)
- Astronomical polarimetry
- Optical communications
- Birefringent material characterization
- Laser system design

## Implementation Details

The `VectorialMonochromaticField` class extends `MonochromaticField` with three field components:
- `Ex`, `Ey`, `Ez` (complex 2D arrays)

All propagation methods (angular spectrum, Fresnel, Bluestein) work with vector fields by applying the same diffraction integral to each component independently.

Total intensity: `I = |Ex|² + |Ey|² + |Ez|²`

## Backward Compatibility

The original scalar `MonochromaticField` class remains unchanged. Use `VectorialMonochromaticField` only when polarization effects are important.

## References

1. Born & Wolf, "Principles of Optics" (Chapter 10: Interference and Diffraction with Partially Polarized Light)
2. Goodman, "Introduction to Fourier Optics" (Chapter 6: Frequency Analysis)
3. Hecht, "Optics" (Chapter 8: Polarization)
