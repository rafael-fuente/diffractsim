import diffractsim
diffractsim.set_backend("CPU")
from diffractsim import mm, nm, cm, VectorialField, CircularAperture, LinearPolarizer, QuarterWavePlate, HalfWavePlate
import matplotlib.pyplot as plt
import numpy as np

# wavelength of the field (Red laser)
wavelength = 633 * nm

# 1. Setup a horizontally polarized field
print("Setting up horizontally polarized field...")
F = VectorialField(wavelength, 10 * mm, 10 * mm, 512, 512, pol_state=[1, 0])

# 2. Add a circular aperture
F.add(CircularAperture(radius=1.5 * mm))

# 3. Add a Quarter Wave Plate at 45 degrees to convert to Circular Polarization
print("Applying Quarter Wave Plate at 45 degrees...")
F.add(QuarterWavePlate(fast_axis_angle=45))

# Verify S3 (circular polarization parameter) is high
S0, S1, S2, S3 = F.get_stokes_parameters()
avg_S3 = np.mean(S3[F.Ny//2-10:F.Ny//2+10, F.Nx//2-10:F.Nx//2+10])
print(f"Average S3 in center (should be ~S0 for circular): {avg_S3}")

# 4. Propagate
print("Propagating...")
F.propagate(50 * cm)

# 5. Add a vertical polarizer to see if we can filter components
print("Applying vertical polarizer...")
F.add(LinearPolarizer(angle=90))

# 6. Final propagation to screen
F.propagate(50 * cm)

print("Vectorial simulation with polarization elements finished successfully.")
print(f"Final intensity in center: {np.max(F.get_intensity())}")
