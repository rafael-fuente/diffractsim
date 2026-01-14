"""
Example demonstrating linear polarization with vectorial EM field.
Shows how polarization state changes through diffraction.
"""

from diffractsim import mm, cm, nm, VectorialMonochromaticField
from diffractsim.diffractive_elements import CircularAperture
import matplotlib.pyplot as plt

# Create vectorial field
F = VectorialMonochromaticField(
    wavelength=632.8*nm,  # HeNe laser
    extent_x=4*mm,
    extent_y=4*mm,
    Nx=500,
    Ny=500
)

# Set horizontal linear polarization
F.set_linear_polarization(angle=0)

# Add circular aperture
F.add(CircularAperture(radius=0.5*mm))

# Propagate
F.propagate(50*cm)

# Get intensity and polarization data
intensity = F.get_intensity()
S0, S1, S2, S3 = F.get_stokes_parameters()
degree_of_pol = F.get_degree_of_polarization()

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

# Total intensity
axes[0, 0].imshow(intensity, extent=[-2*mm, 2*mm, -2*mm, 2*mm], cmap='hot')
axes[0, 0].set_title('Total Intensity')
axes[0, 0].set_xlabel('x (mm)')
axes[0, 0].set_ylabel('y (mm)')

# S1 (horizontal vs vertical)
axes[0, 1].imshow(S1/S0, extent=[-2*mm, 2*mm, -2*mm, 2*mm], cmap='RdBu', vmin=-1, vmax=1)
axes[0, 1].set_title('S1/S0 (H-V polarization)')
axes[0, 1].set_xlabel('x (mm)')
axes[0, 1].set_ylabel('y (mm)')

# S2 (diagonal polarization)
axes[1, 0].imshow(S2/S0, extent=[-2*mm, 2*mm, -2*mm, 2*mm], cmap='RdBu', vmin=-1, vmax=1)
axes[1, 0].set_title('S2/S0 (diagonal polarization)')
axes[1, 0].set_xlabel('x (mm)')
axes[1, 0].set_ylabel('y (mm)')

# Degree of polarization
axes[1, 1].imshow(degree_of_pol, extent=[-2*mm, 2*mm, -2*mm, 2*mm], cmap='hot', vmin=0, vmax=1)
axes[1, 1].set_title('Degree of Polarization')
axes[1, 1].set_xlabel('x (mm)')
axes[1, 1].set_ylabel('y (mm)')

plt.tight_layout()
plt.savefig('polarization_linear.png', dpi=150)
plt.show()

print("Simulation complete!")
print(f"Mean degree of polarization: {degree_of_pol.mean():.3f}")
