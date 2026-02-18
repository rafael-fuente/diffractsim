import numpy as np
import matplotlib.pyplot as plt

def simulate_polarized_light(aperture_size, wavelength, polarization_angle):
    # Create a grid of points in space
    x = np.linspace(-10, 10, 400)
    y = np.linspace(-10, 10, 400)
    X, Y = np.meshgrid(x, y)

    # Calculate the distance from the origin (center of aperture)
    r = np.sqrt(X**2 + Y**2)

    # Create an aperture
    aperture = np.zeros_like(r)
    aperture[r < aperture_size / 2] = 1

    # Simulate polarized light
    theta = np.arctan2(Y, X)
    polarization_factor = np.cos(2 * (theta - polarization_angle))
    light_intensity = aperture * polarization_factor**2

    return light_intensity

def plot_light_intensity(light_intensity):
    plt.imshow(light_intensity, extent=[-10, 10, -10, 10], origin='lower', cmap='viridis')
    plt.colorbar(label='Intensity')
    plt.title('Polarized Light Intensity Through Aperture')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.show()

# Example usage
aperture_size = 2.0
wavelength = 1.0
polarization_angle = np.pi / 4

light_intensity = simulate_polarized_light(aperture_size, wavelength, polarization_angle)
plot_light_intensity(light_intensity)