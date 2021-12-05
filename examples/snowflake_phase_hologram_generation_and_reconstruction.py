import diffractsim
diffractsim.set_backend("CPU") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, mm, nm, cm, PhaseRetrieval


# Generate a Fourier plane phase hologram
PR = PhaseRetrieval(target_amplitude_path = './apertures/snowflake.png', new_size= (600,600), pad = 0)
PR.retrieve_phase_mask(max_iter = 200, method = 'Gerchberg-Saxton')
PR.save_retrieved_phase_as_image('snowflake_phase_hologram.png')


#Add a plane wave
F = MonochromaticField(
    wavelength=632.8 * nm, extent_x=30 * mm, extent_y=30 * mm, Nx=1800, Ny=1800, intensity = 0.005
)


# load the hologram as a phase mask aperture
F.add_aperture_from_image(
     amplitude_mask_path= "./apertures/white_background.png", 
     phase_mask_path= "snowflake_phase_hologram.png", image_size=(10.0 * mm, 10.0 * mm)
)

# plot colors at z = 0
rgb = F.get_colors()
F.plot_colors(rgb)


# propagate field to Fourier plane
F.add_lens(f = 80*cm) 
F.propagate(80*cm)


# plot colors (reconstructed image) at z = 80*cm (Fourier plane)
rgb = F.get_colors()
F.plot_colors(rgb)
