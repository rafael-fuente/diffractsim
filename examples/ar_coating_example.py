"""
Anti-Reflection (AR) Coating Example

This example demonstrates the design and analysis of a quarter-wave anti-reflection 
coating using the Transfer Matrix Method (TMM).

Theory:
A single-layer AR coating works by destructive interference. For optimal performance at wavelength λ₀:
- Coating thickness: d = λ₀ / (4n_coating)
- Optimal index: n_coating = √(n_air × n_substrate)

For air (n=1.0) and glass (n=1.5), the optimal coating index is √1.5 ≈ 1.225.
"""

import numpy as np
import matplotlib.pyplot as plt
from diffractsim.tmm import Layer, Stack
from diffractsim.util.constants import nm, um


def main():
    # Design wavelength
    λ0 = 500 * nm
    
    # Materials
    n_air = 1.0
    n_glass = 1.5
    n_coating_optimal = np.sqrt(n_air * n_glass)  # ≈ 1.225
    
    # Quarter-wave thickness
    d_coating = λ0 / (4 * n_coating_optimal)
    
    print(f"Design wavelength: {λ0/nm:.1f} nm")
    print(f"Optimal coating index: {n_coating_optimal:.4f}")
    print(f"Coating thickness: {d_coating/nm:.2f} nm")
    
    # Create stack: air - coating - glass
    air = Layer(n=n_air, name="air")
    coating = Layer(n=n_coating_optimal, d=d_coating, name="coating")
    glass = Layer(n=n_glass, name="glass")
    
    stack = Stack([air, coating, glass])
    
    # Wavelength sweep
    print("\nComputing reflectance vs wavelength...")
    wavelengths = np.linspace(400, 700, 300) * nm
    R = []
    T = []
    
    for λ in wavelengths:
        result = stack.solve(λ, θ_incident=0.0, polarization="s")
        R.append(result['R'])
        T.append(result['T'])
    
    R = np.array(R)
    T = np.array(T)
    
    # Plot reflectance and transmittance
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths/nm, R * 100, 'b-', label='Reflectance', linewidth=2)
    plt.plot(wavelengths/nm, T * 100, 'g-', label='Transmittance', linewidth=2)
    plt.axvline(λ0/nm, color='r', linestyle='--', alpha=0.5, label=f'Design λ = {λ0/nm:.0f} nm')
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Power (%)', fontsize=12)
    plt.title('AR Coating: Reflectance and Transmittance vs Wavelength', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(400, 700)
    plt.ylim(0, 5)
    plt.tight_layout()
    plt.savefig('ar_coating_wavelength.png', dpi=150)
    print("Saved: ar_coating_wavelength.png")
    
    # Comparison with bare glass
    print("\nComparing with bare glass...")
    stack_bare = Stack([air, glass])
    
    R_bare = []
    for λ in wavelengths:
        result = stack_bare.solve(λ, θ_incident=0.0, polarization="s")
        R_bare.append(result['R'])
    
    R_bare = np.array(R_bare)
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths/nm, R_bare * 100, 'r--', label='Bare glass (no coating)', linewidth=2)
    plt.plot(wavelengths/nm, R * 100, 'b-', label='With AR coating', linewidth=2)
    plt.axvline(λ0/nm, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Wavelength (nm)', fontsize=12)
    plt.ylabel('Reflectance (%)', fontsize=12)
    plt.title('AR Coating Performance Comparison', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(400, 700)
    plt.ylim(0, 5)
    plt.tight_layout()
    plt.savefig('ar_coating_comparison.png', dpi=150)
    print("Saved: ar_coating_comparison.png")
    
    print(f"\nReflectance reduction at {λ0/nm:.0f} nm:")
    idx_design = np.argmin(np.abs(wavelengths - λ0))
    print(f"  Without coating: {R_bare[idx_design]*100:.2f}%")
    print(f"  With coating: {R[idx_design]*100:.4f}%")
    print(f"  Reduction: {(1 - R[idx_design]/R_bare[idx_design])*100:.1f}%")
    
    # Angular dependence
    print("\nComputing angular dependence...")
    angles_deg = np.linspace(0, 60, 100)
    angles_rad = np.deg2rad(angles_deg)
    
    R_angle = []
    R_angle_bare = []
    
    for θ in angles_rad:
        result = stack.solve(λ0, θ_incident=θ, polarization="s")
        R_angle.append(result['R'])
        
        result_bare = stack_bare.solve(λ0, θ_incident=θ, polarization="s")
        R_angle_bare.append(result_bare['R'])
    
    R_angle = np.array(R_angle)
    R_angle_bare = np.array(R_angle_bare)
    
    # Plot angular dependence
    plt.figure(figsize=(10, 6))
    plt.plot(angles_deg, R_angle_bare * 100, 'r--', label='Bare glass', linewidth=2)
    plt.plot(angles_deg, R_angle * 100, 'b-', label='With AR coating', linewidth=2)
    plt.xlabel('Angle of Incidence (degrees)', fontsize=12)
    plt.ylabel('Reflectance (%)', fontsize=12)
    plt.title(f'AR Coating: Reflectance vs Angle (λ = {λ0/nm:.0f} nm)', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 60)
    plt.tight_layout()
    plt.savefig('ar_coating_angle.png', dpi=150)
    print("Saved: ar_coating_angle.png")
    
    print("\nExample complete!")


if __name__ == "__main__":
    main()

