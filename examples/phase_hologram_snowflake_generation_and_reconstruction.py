import diffractsim
diffractsim.set_backend("CUDA") #Change the string to "CUDA" to use GPU acceleration

from diffractsim import MonochromaticField, ApertureFromImage, Lens, mm, nm, cm, FourierPhaseRetrieval


# Generate a Fourier plane phase hologram
PR = FourierPhaseRetrieval(target_amplitude_path = './apertures/snowflake.png', new_size= (400,400), pad = (200,200))
PR.retrieve_phase_mask(max_iter = 200, method = 'Conjugate-Gradient')
PR.save_retrieved_phase_as_image('snowflake_phase_hologram.png')


#Add a plane wave
F = MonochromaticField(
    wavelength=632.8 * nm, extent_x=30 * mm, extent_y=30 * mm, Nx=2400, Ny=2400, intensity = 0.005
)


# load the hologram as a phase mask aperture
F.add(ApertureFromImage(
     amplitude_mask_path= "./apertures/white_background.png", 
     phase_mask_path= "snowflake_phase_hologram.png", image_size=(10.0 * mm, 10.0 * mm), simulation = F))

# plot colors at z = 0
rgb = F.get_colors()
F.plot_colors(rgb)

# propagate field to Fourier plane
F.add(Lens(f = 80*cm))
F.propagate(80*cm)


# plot colors (reconstructed image) at z = 80*cm (Fourier plane)
rgb = F.get_colors()
F.plot_colors(rgb)
