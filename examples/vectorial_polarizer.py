import diffractsim
diffractsim.set_backend("CPU") # Use CPU for compatibility in this example
from diffractsim import mm, nm, cm, VectorialField, CircularAperture, LinearPolarizer, QuarterWavePlate

# wavelength of the field
wavelength = 633 * nm

# setup the simulation
# VectorialField(wavelength, extent_x, extent_y, Nx, Ny, intensity, pol_state)
# pol_state [1, 1] means 45 degree linear polarization
F = VectorialField(wavelength, 10 * mm, 10 * mm, 512, 512, pol_state=[1, 1])

# add a circular aperture
F.add(CircularAperture(radius=1.5 * mm))

# add a horizontal polarizer (0 degrees)
F.add(LinearPolarizer(angle=0))

# propagate the field to 1 meter
F.propagate(100 * cm)

# plot the result
# Since this is a head-less environment, we might not see the plot, 
# but we can save it if needed or just verify it runs.
# F.plot_colors(saveas = "vectorial_polarizer.png")
print("Vectorial simulation finished successfully.")
print(f"Final z: {F.z/cm} cm")
