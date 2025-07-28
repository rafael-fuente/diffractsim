import subprocess
from concurrent.futures import ThreadPoolExecutor
import ffmpeg


def run_simulation_for_frame(slave, frame_number, is_monochromatic, wave_length, spectrum, phase_mask_name, spectrum_sz, spectrum_div,
                                img_size_x, img_size_y, out_filename_pattern):
    # Build the command to run diffractsim_simulation.py with all configurable arguments
    cmd = [
        "python", slave,
        "--frame", str(frame_number),
        "--wavelength", str(wave_length),
        "--light_spectrum", spectrum,
        "--phase_mask", phase_mask_name,
        "--spectrum_size", str(spectrum_sz),
        "--spectrum_divisions", str(spectrum_div),
        "--image_size", str(img_size_x), str(img_size_y),
        "--output_filename_pattern", out_filename_pattern
    ]

    # Add the --monochromatic flag if necessary
    if is_monochromatic:
        cmd.append("--monochromatic")

    # Run the command using subprocess
    print(f"Computing frame {frame_number}")
    subprocess.run(cmd)


if __name__ == "__main__":
    # Number of concurrent instances to run
    max_workers = 11

    # Animation start and end frame as position on z axis (ie distance to lens/sensor)
    start_frame = 1
    end_frame = 100
    encode_to_video = True
    framerate = 20

    # Simulation slave to use
    sim_slave = "phase_hologram_reconstruction_slave.py"

    # Configuration for the simulation (outer scope variables)
    monochromatic = False
    wavelength = 591.0
    light_spectrum = "illuminant_d65"
    phase_mask = "github_logo.png"
    spectrum_size = 400  # by default spectrum has a size of 400. If new size, we interpolate
    spectrum_divisions = 400
    image_size_x = 23.0  # in mm
    image_size_y = 23.0  # in mm
    # Options for light_spectrum are:
    # illuminant_d65, high_pressure_sodium, incandescent_tugsten,
    # compact_fluorescent_lamp, LED_6770K, ceramic_metal_halide,
    # mercury_vapor, cie_cmf

    path = "./animation/"
    filename = "frame_"
    filetype = ".png"  # Can save as jpg or png
    output_filename_pattern = path + filename + "{frame:06d}" + filetype

    # ThreadPoolExecutor to handle parallel execution
    with ThreadPoolExecutor(max_workers = max_workers) as executor:
        # Submitting tasks for frames
        futures = []
        for frame in range(start_frame, end_frame + 1):
            futures.append(executor.submit(
                run_simulation_for_frame, sim_slave, str(frame), monochromatic, str(wavelength), light_spectrum, phase_mask,
                str(spectrum_size), str(spectrum_divisions), str(image_size_x), str(image_size_y), output_filename_pattern))

        # Waiting for all tasks to complete
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred: {e}")

    if encode_to_video:
        (
            ffmpeg
            .input(path + filename + '%06d' + filetype, start_number = start_frame, framerate = framerate)  # Note: pattern_type='glob' and "*" wildcard only work in linux
            .output(path + 'animation.mp4', vcodec = 'png', movflags = 'faststart')  # crf=1, **{'qscale:v': 1}
            .run(overwrite_output = True)
            # for Android use: vcodec='libvpx-vp9', pix_fmt='yuv420p', video_bitrate=200000000, movflags='faststart'
        )
