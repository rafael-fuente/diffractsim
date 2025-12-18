"""
Vectorial EM Field Example - Polarization Effects

This example demonstrates the vectorial field simulator with polarization support:
1. Create polarized beams
2. Apply polarization optical elements (polarizers, waveplates)
3. Propagate vectorial fields
4. Analyze polarization state via Stokes parameters

Author: Solari Systems (Algora Bounty #69)
"""

import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path for development
import sys
sys.path.insert(0, '..')

from diffractsim import VectorialField, JonesVector, JonesMatrix
from diffractsim import create_polarized_beam, apply_polarization_element
from diffractsim import mm, nm, cm, W, m


def example_1_basic_polarization_states():
    """Demonstrate basic polarization states and their Stokes parameters."""
    print("\n=== Example 1: Basic Polarization States ===")

    wavelength = 632.8 * nm  # He-Ne laser
    extent_x = 10 * mm
    extent_y = 10 * mm
    Nx, Ny = 256, 256

    # Create different polarization states
    polarizations = ['horizontal', 'vertical', 'diagonal', 'right_circular', 'left_circular']

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, pol in enumerate(polarizations):
        field = create_polarized_beam(wavelength, extent_x, extent_y, Nx, Ny,
                                      polarization=pol)

        # Get Stokes parameters at center
        S0, S1, S2, S3 = field.get_stokes_parameters()
        S0_center = float(S0[Ny//2, Nx//2])
        S1_center = float(S1[Ny//2, Nx//2])
        S2_center = float(S2[Ny//2, Nx//2])
        S3_center = float(S3[Ny//2, Nx//2])

        print(f"{pol:15}: S0={S0_center:.3f}, S1={S1_center:.3f}, S2={S2_center:.3f}, S3={S3_center:.3f}")

        # Plot intensity
        I = field.get_intensity()
        if hasattr(I, 'get'):
            I = I.get()  # Convert from GPU if needed

        axes[i].imshow(I, extent=[-extent_x/mm/2, extent_x/mm/2, -extent_y/mm/2, extent_y/mm/2])
        axes[i].set_title(f'{pol}\nS=({S1_center:.2f},{S2_center:.2f},{S3_center:.2f})')
        axes[i].set_xlabel('x (mm)')
        axes[i].set_ylabel('y (mm)')

    axes[5].axis('off')
    plt.tight_layout()
    plt.savefig('polarization_states.png', dpi=150)
    print("Saved: polarization_states.png")


def example_2_polarizer_effect():
    """Show effect of linear polarizer on different polarization states."""
    print("\n=== Example 2: Polarizer Effect ===")

    wavelength = 632.8 * nm
    extent = 10 * mm
    N = 256

    # Create diagonally polarized beam
    field = create_polarized_beam(wavelength, extent, extent, N, N,
                                  polarization='diagonal')

    print("Initial (diagonal polarization):")
    I_initial = float(field.get_intensity()[N//2, N//2])
    print(f"  Intensity: {I_initial:.4f}")

    # Apply horizontal polarizer
    apply_polarization_element(field, 'polarizer', angle=0)

    I_after = float(field.get_intensity()[N//2, N//2])
    print(f"After horizontal polarizer:")
    print(f"  Intensity: {I_after:.4f}")
    print(f"  Transmission: {I_after/I_initial:.2%}")


def example_3_quarter_wave_plate():
    """Convert linear to circular polarization using quarter-wave plate."""
    print("\n=== Example 3: Quarter-Wave Plate ===")

    wavelength = 632.8 * nm
    extent = 10 * mm
    N = 256

    # Create diagonally polarized beam
    field = create_polarized_beam(wavelength, extent, extent, N, N,
                                  polarization='diagonal')

    S0, S1, S2, S3 = field.get_stokes_parameters()
    print("Initial (diagonal):")
    print(f"  S3 (circular): {float(S3[N//2, N//2]):.4f}")

    # Apply quarter-wave plate at 0 degrees
    apply_polarization_element(field, 'quarter_wave', angle=0)

    S0, S1, S2, S3 = field.get_stokes_parameters()
    print("After QWP at 0°:")
    print(f"  S3 (circular): {float(S3[N//2, N//2]):.4f}")
    print("  -> Circular polarization achieved!")


def example_4_propagation():
    """Propagate vectorial field and observe polarization preservation."""
    print("\n=== Example 4: Vectorial Propagation ===")

    wavelength = 632.8 * nm
    extent = 10 * mm
    N = 256

    # Create circularly polarized beam
    field = create_polarized_beam(wavelength, extent, extent, N, N,
                                  polarization='right_circular')

    S0, S1, S2, S3 = field.get_stokes_parameters()
    print("Before propagation:")
    print(f"  S3 (circular): {float(S3[N//2, N//2]):.4f}")
    print(f"  DOP: {float(field.get_degree_of_polarization()[N//2, N//2]):.4f}")

    # Propagate 10 cm
    field.propagate(10 * cm)

    S0, S1, S2, S3 = field.get_stokes_parameters()
    print("After 10cm propagation:")
    print(f"  S3 (circular): {float(S3[N//2, N//2]):.4f}")
    print(f"  DOP: {float(field.get_degree_of_polarization()[N//2, N//2]):.4f}")
    print("  -> Polarization state preserved during free-space propagation!")


def example_5_jones_calculus():
    """Direct Jones calculus demonstration."""
    print("\n=== Example 5: Jones Calculus ===")

    # Create Jones vectors
    H = JonesVector(1, 0)  # Horizontal
    D = JonesVector(1/np.sqrt(2), 1/np.sqrt(2))  # Diagonal

    print("Horizontal polarization:")
    print(f"  Jones vector: ({H.Ex:.3f}, {H.Ey:.3f})")
    print(f"  Stokes: {H.stokes_parameters()}")

    # Apply half-wave plate at 22.5 degrees to rotate H to D
    HWP = JonesMatrix.half_wave_plate(angle=np.pi/8)  # 22.5 degrees
    result = HWP @ H

    print("\nAfter HWP at 22.5°:")
    print(f"  Jones vector: ({result.Ex:.3f}, {result.Ey:.3f})")
    print(f"  Stokes: {result.stokes_parameters()}")


if __name__ == "__main__":
    print("=" * 60)
    print(" Diffractsim Vectorial Field Examples")
    print("=" * 60)

    example_1_basic_polarization_states()
    example_2_polarizer_effect()
    example_3_quarter_wave_plate()
    example_4_propagation()
    example_5_jones_calculus()

    print("\n" + "=" * 60)
    print(" All examples completed successfully!")
    print("=" * 60)
