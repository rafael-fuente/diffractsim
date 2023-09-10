import diffractsim

try:
    diffractsim.set_backend("CUDA")  # Change the string to "CUDA" to use GPU acceleration
except Exception as e:
    print(f"Error setting CUDA backend: {e}")

from diffractsim import MonochromaticField, ApertureFromImage, Lens, mm, nm, cm, FourierPhaseRetrieval

try:
    # Try to generate a Fourier plane phase hologram
    PR = FourierPhaseRetrieval(target_amplitude_path='./apertures/snowflake.png', new_size=(400, 400), pad=(200, 200))
    PR.retrieve_phase_mask(max_iter=200, method='Conjugate-Gradient')
    PR.save_retrieved_phase_as_image('snowflake_phase_hologram.png')
except Exception as e:
    print(f"Error generating phase hologram: {e}")

try:
    # Try to create a monochromatic field
    F = MonochromaticField(
        wavelength=632.8 * nm, extent_x=30 * mm, extent_y=30 * mm, Nx=2400, Ny=2400, intensity=0.005
    )
except Exception as e:
    print(f"Error creating monochromatic field: {e}")

try:
    # Try to load the hologram as a phase mask aperture
    F.add(ApertureFromImage(
        amplitude_mask_path="./apertures/white_background.png",
        phase_mask_path="snowflake_phase_hologram.png", image_size=(10.0 * mm, 10.0 * mm), simulation=F))
except Exception as e:
    print(f"Error loading hologram as phase mask aperture: {e}")

try:
    # Try to propagate the field to the Fourier plane
    F.add(Lens(f=80 * cm))
    F.propagate(80 * cm)
except Exception as e:
    print(f"Error propagating the field to the Fourier plane: {e}")

try:
    # Try to visualize the colors in the Fourier plane
    rgb = F.get_colors()
    F.plot_colors(rgb)
except Exception as e:
    print(f"Error visualizing colors in the Fourier plane: {e}")
