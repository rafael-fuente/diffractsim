import diffractsim
diffractsim.set_backend("JAX")

from diffractsim import MonochromaticField, mm, nm, cm
from diffractsim import SLM, load_image_as_function, load_file_as_function, load_phase_as_function
from diffractsim import CustomPhaseRetrieval

#Note: CustomPhaseRetrieval requires autograd which is not installed by default with diffractsim. 
# To install autograd, type: 'pip install -U jax'


# Generate a 30cm plane phase hologram
distance = 30*cm
PR = CustomPhaseRetrieval(wavelength=532 * nm, z = distance, extent_x=30 * mm, extent_y=30 * mm, Nx=2048, Ny=2048)

PR.set_source_amplitude(load_image_as_function("./apertures/white_background.png", 
                                               x_interval = (-15*mm/2, +15*mm/2),   y_interval = (-15*mm/2, +15*mm/2)))
PR.set_target_amplitude(load_image_as_function("./apertures/USAF_test.png", 
                                               x_interval = (-15*mm/2, +15*mm/2),   y_interval = (-15*mm/2, +15*mm/2)))

PR.retrieve_phase_mask(max_iter = 1000, method = 'Adam-JAX')
PR.save_retrieved_phase_as_image('USAF_hologram.png')
PR.save_retrieved_phase_as_file('USAF_hologram.npy')



F = MonochromaticField(
    wavelength=532 * nm, extent_x=30 * mm, extent_y=30 * mm, Nx=2048, Ny=2048, intensity = 0.001
)


# add Spatial Light Modulator
F.add(SLM(
      phase_mask_function = load_phase_as_function("USAF_hologram.npy", 
      x_interval = (-15*mm, +15*mm),   y_interval = (-15*mm, +15*mm)), 
      size_x =15*mm, size_y =15*mm, 
      simulation = F)
)


# plot phase at z = 0
E = F.get_field()
F.plot_phase(E, grid = True, units = mm)

# propagate field 30*cm
F.propagate(distance)


# plot reconstructed image
rgb = F.get_colors()
F.plot_colors(rgb)
