"""
Example demonstrating circular polarization with vectorial EM field.
Shows conversion from circular to elliptical polarization through aperture.
"""

from diffractsim import mm, cm, nm, VectorialMonochromaticField
from diffractsim.diffractive_elements import RectangularSlit
import matplotlib.pyplot as plt

# Create vectorial field
F = VectorialMonochromaticField(
    wavelength=632.8*nm,
    extent_x=4*mm,
    extent_y=4*mm,
    Nx=500,
    Ny=500
)

# Set right circular polarization
F.set_circular_polarization(handedness='right')

# Add rectangular slit (breaks circular symmetry)
F.add(RectangularSlit(width=0.4*mm, height=0.8*mm))

# Propagate
F.propagate(50*cm)

# Get Stokes parameters
S0, S1, S2, S3 = F.get_stokes_parameters()

# Plot results showing conversion from circular to elliptical
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Total intensity
axes[0, 0].imshow(S0, extent=[-2*mm, 2*mm, -2*mm, 2*mm], cmap='hot')
axes[0, 0].set_title('Total Intensity (S0)')
axes[0, 0].set_xlabel('x (mm)')
axes[0, 0].set_ylabel('y (mm)')

# Linear horizontal-vertical
axes[0, 1].imshow(S1/S0, extent=[-2*mm, 2*mm, -2*mm, 2*mm], cmap='RdBu', vmin=-1, vmax=1)
axes[0, 1].set_title('S1/S0 (Linear H-V)')
axes[0, 1].set_xlabel('x (mm)')
axes[0, 1].set_ylabel('y (mm)')

# Linear diagonal
axes[1, 0].imshow(S2/S0, extent=[-2*mm, 2*mm, -2*mm, 2*mm], cmap='RdBu', vmin=-1, vmax=1)
axes[1, 0].set_title('S2/S0 (Linear ±45°)')
axes[1, 0].set_xlabel('x (mm)')
axes[1, 0].set_ylabel('y (mm)')

# Circular polarization
axes[1, 1].imshow(S3/S0, extent=[-2*mm, 2*mm, -2*mm, 2*mm], cmap='RdBu', vmin=-1, vmax=1)
axes[1, 1].set_title('S3/S0 (Circular R-L)')
axes[1, 1].set_xlabel('x (mm)')
axes[1, 1].set_ylabel('y (mm)')

plt.tight_layout()
plt.savefig('polarization_circular.png', dpi=150)
plt.show()

print("Simulation complete!")
print("Slit breaks circular symmetry, converting circular → elliptical polarization")
