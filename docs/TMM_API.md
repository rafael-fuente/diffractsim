# Transfer Matrix Method (TMM) API Reference

## Overview

The TMM module provides tools for analyzing planar multilayer optical stacks, including:
- Fresnel coefficient calculations
- Multilayer stack analysis
- Material models (including Drude model for metals)
- Thin optical elements (polarizers, retarders)
- Surface Plasmon Polariton (SPP) calculations

## Core Classes

### `Layer`

Represents a single layer in a multilayer stack.

```python
from diffractsim.tmm import Layer

# Constant index
layer = Layer(n=1.5, k=0.0, d=100*nm, name="coating")

# Wavelength-dependent index
def n_func(λ):
    return 1.5 + 0.01 / λ

layer = Layer(n=n_func, d=100*nm)

# Semi-infinite layer (substrate)
substrate = Layer(n=1.5, name="glass")
```

**Parameters:**
- `n`: Refractive index (float, callable, or Material object)
- `k`: Extinction coefficient (float or callable), default 0.0
- `d`: Layer thickness in meters (None for semi-infinite)
- `name`: Optional layer name

### `Stack`

Multilayer stack for TMM calculations.

```python
from diffractsim.tmm import Stack, Layer

# Create stack: air - coating - glass
air = Layer(n=1.0, name="air")
coating = Layer(n=1.3, d=100*nm, name="coating")
glass = Layer(n=1.5, name="glass")

stack = Stack([air, coating, glass])

# Solve for reflectance/transmittance
result = stack.solve(wavelength=500*nm, θ_incident=0.0, polarization="s")
print(f"Reflectance: {result['R']*100:.2f}%")
print(f"Transmittance: {result['T']*100:.2f}%")
```

**Methods:**
- `solve(wavelength, θ_incident=0.0, polarization="s")`: Calculate R, T, A
  - Returns dict with keys: 'R', 'T', 'A', 'r', 't'
- `get_field_profile(...)`: Compute field profile through stack (TODO)

## Materials

### `Material`

Base class for optical materials.

```python
from diffractsim.tmm import Material

def n_func(λ):
    return 1.5 + 0.01 / λ

material = Material(n_func=n_func)
```

### `DrudeMaterial`

Drude model for metals.

```python
from diffractsim.tmm import DrudeMaterial

# Gold parameters
ωp = 1.37e16  # Plasma frequency (rad/s)
γ = 4.05e13   # Damping constant (rad/s)
gold = DrudeMaterial(ωp, γ, ε_inf=1.0)

layer = Layer(gold, d=50*nm, name="gold")
```

## Fresnel Coefficients

```python
from diffractsim.tmm import fresnel_coefficients

n1, n2 = 1.0, 1.5
θ1 = np.pi / 6  # 30 degrees

r, t, θ2 = fresnel_coefficients(n1, n2, θ1, polarization="s")
```

**Parameters:**
- `n1`, `n2`: Refractive indices
- `θ1`: Angle of incidence (radians)
- `polarization`: "s" (TE) or "p" (TM)

**Returns:**
- `r`: Reflection coefficient (complex)
- `t`: Transmission coefficient (complex)
- `θ2`: Angle of refraction (radians)

## Thin Optical Elements

### `IdealPolarizer`

Ideal linear polarizer (Jones matrix).

```python
from diffractsim.tmm import IdealPolarizer
from diffractsim import VectorialField

polarizer = IdealPolarizer(angle=np.pi/4)  # 45 degrees
field_out = polarizer.apply(field_in)
```

### `IdealRetarder`

Ideal waveplate/retarder.

```python
from diffractsim.tmm import IdealRetarder, quarter_wave_plate, half_wave_plate

# Quarter-wave plate
qwp = quarter_wave_plate(axis_angle=0.0)

# Half-wave plate
hwp = half_wave_plate(axis_angle=np.pi/4)

# Custom retarder
retarder = IdealRetarder(retardance=np.pi/2, axis_angle=0.0)
```

## Surface Plasmon Polaritons (SPP)

### `kretschmann_configuration`

Kretschmann configuration for SPP excitation.

```python
from diffractsim.tmm.spp import kretschmann_configuration

# Angle sweep
result = kretschmann_configuration(
    prism_n=1.515,
    metal_layer=metal_layer,
    dielectric_n=1.0,
    wavelength=633*nm,
    angle_range=(θ_min, θ_max, n_points)
)

print(f"Resonance angle: {np.rad2deg(result['resonance_angle']):.2f}°")
```

### `single_interface_spp`

Single interface SPP properties.

```python
from diffractsim.tmm.spp import single_interface_spp

spp_props = single_interface_spp(
    metal_n=0.1 + 3j,
    dielectric_n=1.0,
    wavelength=633*nm
)

print(f"Effective index: {spp_props['n_eff']:.4f}")
print(f"Penetration depth (metal): {spp_props['penetration_depth_metal']/nm:.2f} nm")
```

## VectorialField

Vectorial electromagnetic field with Ex and Ey components.

```python
from diffractsim import VectorialField

# Create field
Ex = np.ones((100, 100), dtype=complex)
Ey = np.zeros((100, 100), dtype=complex)
field = VectorialField(Ex, Ey, wavelength=500*nm, dx=1e-6, dy=1e-6)

# Basis transforms
field_sp = field.to_basis("sp", angle=0.0)

# Stokes parameters
stokes = field.stokes()
print(f"S0 (intensity): {stokes['S0']}")

# Intensity
I = field.intensity()
```

## Examples

See:
- `examples/ar_coating_example.py` - Anti-reflection coating design
- `examples/spp_kretschmann_example.py` - SPP in Kretschmann configuration

## References

- Transfer Matrix Method: Born & Wolf, "Principles of Optics"
- SPP: Raether, "Surface Plasmons"
- Jones matrices: Jones, "A new calculus for the treatment of optical systems"

