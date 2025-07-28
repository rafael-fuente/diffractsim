import argparse
import diffractsim
from diffractsim import PolychromaticField, MonochromaticField, ApertureFromImage, Lens, mm, nm, cm, colour_functions as cf

# Set the backend
diffractsim.set_backend("CPU")


def run_simulation(frame: int, monochromatic: bool, wavelength: float, spectrum: str, phase_mask: str, spectrum_size: int,
                    spectrum_divisions: int, image_size_x: float, image_size_y: float,
                    output_filename_pattern: str):

    match spectrum:
        case "illuminant_d65":
            light_spectrum = cf.illuminant_d65
        case "high_pressure_sodium":
            light_spectrum = cf.high_pressure_sodium
        case "incandescent_tugsten":
            light_spectrum = cf.incandescent_tugsten
        case "compact_fluorescent_lamp":
            light_spectrum = cf.compact_fluorescent_lamp
        case "mercury_vapor":
            light_spectrum = cf.mercury_vapor
        case "LED_6770K":
            light_spectrum = cf.LED_6770K
        case "ceramic_metal_halide":
            light_spectrum = cf.ceramic_metal_halide
        case "cie_cmf":
            light_spectrum = cf.cie_cmf

    # Handle monochromatic or polychromatic fields
    if monochromatic:
        F = MonochromaticField(
            wavelength = wavelength * nm, extent_x = 30 * mm, extent_y = 30 * mm, Nx = 2400, Ny = 2400, intensity = 0.0075)
    else:
        F = PolychromaticField(
            spectrum = light_spectrum, extent_x = 30 * mm, extent_y = 30 * mm, Nx = 2400, Ny = 2400,
            spectrum_size = spectrum_size, spectrum_divisions = spectrum_divisions)

    # Load the hologram as a phase mask aperture
    F.add(ApertureFromImage(
        amplitude_mask_path = "./apertures/white_background.png",
        phase_mask_path = phase_mask,
        image_size = (image_size_x * mm, image_size_y * mm), simulation = F))

    # Add a lens with a focal length
    F.add(Lens(f = 80 * cm))

    # Propagate the field based on the frame argument
    F.propagate(frame * cm)

    # Get the reconstructed image (colors) at the specified z position
    rgb = F.get_colors()

    # Format the filename using the provided pattern
    imagefile = output_filename_pattern.format(frame = frame)

    print(imagefile + " done")

    # Save the plot to the specified path
    F.save_plot(rgb, figsize = (16, 16), path = imagefile, tight = True)


if __name__ == "__main__":
    # Set up argument parser to accept all configurable arguments
    parser = argparse.ArgumentParser(description = "Run a phase hologram reconstruction with customizable options.")

    # Adding arguments for frame, monochromatic field, and the new parameters
    parser.add_argument('--frame', required=True, type = int, default = 1, help = "Set the frame number (default is 1).")
    parser.add_argument('--monochromatic', action=argparse.BooleanOptionalAction, default = False, help = "Use monochromatic field instead of polychromatic.")
    parser.add_argument('--wavelength', required=True, type = float, default = 591.0, help = "Wavelength in nm for the monochromatic field.")
    parser.add_argument('--light_spectrum', required=True, type = str, default = "illuminant_d65", help = "Light spectrum to use. Options are:"
                                                                                                "illuminant_d65, high_pressure_sodium, incandescent_tugsten,"
                                                                                                "compact_fluorescent_lamp, LED_6770K, ceramic_metal_halide,"
                                                                                                "mercury_vapor, cie_cmfSee ./diffractsim/data/*.")
    parser.add_argument('--phase_mask', required=True, type = str, default = "github_logo.png", help = "Path to the phase mask image.")
    parser.add_argument('--spectrum_size', required=True, type = int, default = 400, help = "Spectrum size for the polychromatic field.")
    parser.add_argument('--spectrum_divisions', required=True, type = int, default = 400, help = "Number of spectrum divisions for the polychromatic field.")
    parser.add_argument('--image_size', required=True, nargs = 2, type = float, default = [23.0, 23.0], help = "Size of the image in mm (x, y).")
    parser.add_argument('--output_filename_pattern', required=True, type = str, default = "./animation/frame_{frame:06d}.png",
                        help = "Pattern for naming the output files. Use '{frame}' to insert the frame number.")

    # Parse the arguments from the command line
    args = parser.parse_args()

    # Run the simulation with the provided arguments
    run_simulation(frame = args.frame,
                    monochromatic = args.monochromatic,
                    wavelength = args.wavelength,
                    spectrum = args.light_spectrum,
                    phase_mask = args.phase_mask,
                    spectrum_size = args.spectrum_size,
                    spectrum_divisions = args.spectrum_divisions,
                    image_size_x = args.image_size[0],
                    image_size_y = args.image_size[1],
                    output_filename_pattern = args.output_filename_pattern)
