"""
Surface Plasmon Polariton (SPP) - Kretschmann Configuration

This example demonstrates Surface Plasmon Polariton (SPP) excitation using the 
Kretschmann configuration: prism-metal-dielectric.

Theory:
SPPs are electromagnetic waves that propagate along a metal-dielectric interface. 
In the Kretschmann configuration:
- A prism (high index) is used to couple light into a metal film
- The metal film is in contact with a dielectric
- At the resonance angle, light couples to SPPs, causing a dip in reflectance

We use a Drude model for the metal permittivity.
"""

import numpy as np
import matplotlib.pyplot as plt
from diffractsim.tmm import Layer, Stack
from diffractsim.tmm.materials import DrudeMaterial
from diffractsim.tmm.spp import kretschmann_configuration, single_interface_spp, spp_effective_index
from diffractsim.util.constants import nm, um


def main():
    # Parameters
    λ = 633 * nm  # He-Ne laser wavelength
    
    # Prism (BK7 glass)
    n_prism = 1.515
    
    # Metal (gold) - Drude model parameters
    # Typical values for gold at visible wavelengths
    ωp = 1.37e16  # Plasma frequency (rad/s)
    γ = 4.05e13   # Damping constant (rad/s)
    ε_inf = 1.0  # High-frequency permittivity
    
    gold = DrudeMaterial(ωp, γ, ε_inf)
    metal_layer = Layer(gold, d=50*nm, name="gold")
    
    # Dielectric (air or water)
    n_dielectric = 1.0  # Air
    
    print(f"Wavelength: {λ/nm:.1f} nm")
    print(f"Prism index: {n_prism:.3f}")
    print(f"Dielectric index: {n_dielectric:.3f}")
    print(f"Metal film thickness: {metal_layer.d/nm:.1f} nm")
    
    # Calculate SPP properties for metal-dielectric interface
    print("\nCalculating SPP properties...")
    n_metal = gold.get_index(λ)
    ε_metal = n_metal**2
    ε_dielectric = n_dielectric**2
    
    spp_props = single_interface_spp(n_metal, n_dielectric, λ)
    
    print(f"Metal complex index: {n_metal:.4f}")
    print(f"SPP effective index: {spp_props['n_eff']:.4f}")
    print(f"Penetration depth in metal: {spp_props['penetration_depth_metal']/nm:.2f} nm")
    print(f"Penetration depth in dielectric: {spp_props['penetration_depth_dielectric']/nm:.2f} nm")
    
    # Create Kretschmann stack: prism - metal - dielectric
    prism = Layer(n=n_prism, name="prism")
    dielectric = Layer(n=n_dielectric, name="dielectric")
    
    stack = Stack([prism, metal_layer, dielectric])
    
    # Angle sweep
    print("\nComputing Kretschmann configuration (angle sweep)...")
    θ_min = np.deg2rad(30)
    θ_max = np.deg2rad(80)
    n_points = 200
    
    result = kretschmann_configuration(
        n_prism, metal_layer, n_dielectric, λ,
        angle_range=(θ_min, θ_max, n_points)
    )
    
    angles_deg = np.rad2deg(result['angles'])
    R = result['R']
    T = result['T']
    A = result['A']
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(angles_deg, R * 100, 'b-', label='Reflectance', linewidth=2)
    plt.plot(angles_deg, T * 100, 'g-', label='Transmittance', linewidth=2)
    plt.plot(angles_deg, A * 100, 'r-', label='Absorbance', linewidth=2)
    plt.axvline(np.rad2deg(result['resonance_angle']), color='k', linestyle='--', 
                label=f"Resonance: {np.rad2deg(result['resonance_angle']):.2f}°")
    plt.xlabel('Angle of Incidence (degrees)', fontsize=12)
    plt.ylabel('Power (%)', fontsize=12)
    plt.title(f'Kretschmann Configuration: R/T/A vs Angle (λ = {λ/nm:.1f} nm)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(30, 80)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('spp_kretschmann_angle.png', dpi=150)
    print("Saved: spp_kretschmann_angle.png")
    
    print(f"\nSPP Resonance Angle: {np.rad2deg(result['resonance_angle']):.2f}°")
    print(f"Minimum Reflectance: {result['min_R']*100:.2f}%")
    
    # Wavelength sweep at fixed angle
    print("\nComputing wavelength dependence...")
    θ_fixed = result['resonance_angle']
    
    wavelengths = np.linspace(400, 800, 200) * nm
    R_wl = []
    T_wl = []
    A_wl = []
    
    for λ_wl in wavelengths:
        # Update metal layer for each wavelength
        metal_layer_wl = Layer(gold, d=50*nm, name="gold")
        stack_wl = Stack([prism, metal_layer_wl, dielectric])
        
        result_wl = stack_wl.solve(λ_wl, θ_incident=θ_fixed, polarization="p")
        R_wl.append(result_wl['R'])
        T_wl.append(result_wl['T'])
        A_wl.append(result_wl['A'])
    
    R_wl = np.array(R_wl)
    T_wl = np.array(T_wl)
    A_wl = np.array(A_wl)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(wavelengths/nm, R_wl * 100, 'b-', label='Reflectance', linewidth=2)
    plt.plot(wavelengths/nm, T_wl * 100, 'g-', label='Transmittance', linewidth=2)
    plt.plot(wavelengths/nm, A_wl * 100, 'r-', label='Absorbance', linewidth=2)
    plt.axvline(λ/nm, color='k', linestyle='--', alpha=0.5, label=f'Design λ = {λ/nm:.0f} nm')
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Power (%)', fontsize=12)
    plt.title(f'Kretschmann Configuration: R/T/A vs Wavelength (θ = {np.rad2deg(θ_fixed):.2f}°)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(400, 800)
    plt.ylim(0, 100)
    plt.tight_layout()
    plt.savefig('spp_kretschmann_wavelength.png', dpi=150)
    print("Saved: spp_kretschmann_wavelength.png")
    
    # SPP dispersion: effective index vs wavelength
    print("\nComputing SPP dispersion...")
    wavelengths_disp = np.linspace(400, 1000, 300) * nm
    n_eff_spp = []
    
    for λ_disp in wavelengths_disp:
        n_metal_disp = gold.get_index(λ_disp)
        ε_metal_disp = n_metal_disp**2
        n_eff = spp_effective_index(ε_metal_disp, ε_dielectric)
        n_eff_spp.append(n_eff)
    
    n_eff_spp = np.array(n_eff_spp)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths_disp/nm, np.real(n_eff_spp), 'b-', label='Re(n_eff)', linewidth=2)
    plt.plot(wavelengths_disp/nm, np.imag(n_eff_spp), 'r--', label='Im(n_eff)', linewidth=2)
    plt.axhline(n_prism, color='g', linestyle=':', alpha=0.7, label=f'Prism index = {n_prism:.3f}')
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Effective Index', fontsize=12)
    plt.title('SPP Dispersion: Effective Index vs Wavelength', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(400, 1000)
    plt.tight_layout()
    plt.savefig('spp_dispersion.png', dpi=150)
    print("Saved: spp_dispersion.png")
    
    # Find where Re(n_eff) crosses prism index (phase matching condition)
    crossing_idx = np.where(np.real(n_eff_spp) > n_prism)[0]
    if len(crossing_idx) > 0:
        print(f"\nPhase matching possible for λ > {wavelengths_disp[crossing_idx[0]]/nm:.1f} nm")
    
    print("\nExample complete!")


if __name__ == "__main__":
    main()

