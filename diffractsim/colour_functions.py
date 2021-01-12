import numpy as np
from scipy.interpolate import CubicSpline
from pathlib import Path

# import CIE XYZ standard observer color matching functions
cmf = np.loadtxt(Path(__file__).parent / "./data/cie-cmf.txt", usecols=(1, 2, 3))
λ_list = np.loadtxt(Path(__file__).parent / "./data/cie-cmf.txt", usecols=(0))

# cubic spline of CIE XYZ standard observer color matching functions
cs_x = CubicSpline(λ_list, cmf.T[0], bc_type="natural")
cs_y = CubicSpline(λ_list, cmf.T[1], bc_type="natural")
cs_z = CubicSpline(λ_list, cmf.T[2], bc_type="natural")

# import illuminant_d65 spectrum
illuminant_d65 = np.loadtxt(Path(__file__).parent / "./data/illuminant_d65.txt", usecols=(1))
λ_list = np.loadtxt(Path(__file__).parent / "./data/illuminant_d65.txt", usecols=(0))
illuminant_d65_spline = CubicSpline(λ_list, illuminant_d65, bc_type="natural")


# XYZ to linear sRGB matrix
T = np.vstack(
    [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]
)

CLIP_CLAMP_TO_ZERO = 0
CLIP_ADD_WHITE = 1

# default clip method
clip_method = CLIP_ADD_WHITE


def XYZ_to_sRGB_linear(XYZ):
    """
    Convert a XYZ to a linear RGB color.


    XYZ: numpy array with multiple XYZ colors in the form: np.array([[X1,X2,X3...],
                                                                     [Y1,Y2,Y3...],
                                                                     [Z1,Z2,Z3...]])
    """

    rgb = np.tensordot(T, XYZ, axes=([1, 0]))

    if clip_method == CLIP_CLAMP_TO_ZERO:
        # set negative rgb values to zero
        rgb = np.where(rgb < 0, 0, rgb)
        return rgb

    if clip_method == CLIP_ADD_WHITE:
        # add enough white to make all rgb values nonnegative
        # find max negative rgb (or 0.0 if all non-negative), we need that much white
        rgb_min = np.amin(rgb, axis=0)
        # get max positive component
        rgb_max = np.amax(rgb, axis=0)

        # get scaling factor to maintain max rgb after adding white
        scaling = np.where(rgb_max > 0.0, rgb_max / (rgb_max - rgb_min + 0.00001), 1.0)

        # add enough white to cancel this out, maintaining the maximum of rgb
        rgb = np.where(rgb_min < 0.0, scaling * (rgb - rgb_min), rgb)
        return rgb


def sRGB_linear_to_sRGB(rgb_linear):

    """
    Convert a linear RGB color to a non linear RGB color (gamma correction).


    RGB: numpy array with multiple RGB colors in the form: np.array([[R1,R2,R3...],
                                                                     [G1,G2,G3...],
                                                                     [B1,B2,B3...]])
    """

    """sRGB standard for gamma inverse correction."""
    rgb = np.where(
        rgb_linear <= 0.00304,
        12.92 * rgb_linear,
        1.055 * np.power(rgb_linear, 1.0 / 2.4) - 0.055,
    )

    # clip intensity if needed (rgb values > 1.0) by scaling
    rgb_max = np.amax(rgb, axis=0) + 0.00001  # avoid division by zero
    intensity_cutoff = 1.0
    rgb = np.where(rgb_max > intensity_cutoff, rgb * intensity_cutoff / (rgb_max), rgb)

    return rgb


def sRGB_to_sRGB_linear(rgb):
    """
    Convert a RGB color to a linear RGB color.


    RGB: numpy array with multiple RGB colors in the form: np.array([[R1,R2,R3...],
                                                                     [G1,G2,G3...],
                                                                     [B1,B2,B3...]])
    """
    return np.where(rgb <= 0.03928, rgb / 12.92, np.power((rgb + 0.055) / 1.055, 2.4))


def XYZ_to_sRGB(XYZ):
    """
    Convert a XYZ to an RGB color.


    XYZ: numpy array with multiple XYZ colors in the form: np.array([[X1,X2,X3...],
                                                                     [Y1,Y2,Y3...],
                                                                     [Z1,Z2,Z3...]])
    """

    rgb = XYZ_to_sRGB_linear(XYZ)
    rgb = sRGB_linear_to_sRGB(rgb)

    return rgb


def spec_to_XYZ(spec):
    """
    Convert a spectrum to an XYZ color.

    spec: numpy array in the form: np.array([spec1, spec2,...,specN)
    where spec1,spec2,...,specN  are lists with spectral intensities sampled on 380-780 nm interval.
    Number of samples of each spectral intensity list doesn't matter, but they must be equally spaced.
    """

    if spec.ndim == 1:

        Δλ = (780 - 380) / (spec.shape[0] - 1)
        λ_list = np.arange(380.0, 780.00001, Δλ)

        X = np.dot(spec, cs_x(λ_list)) * Δλ * 0.003975 * 683.002
        Y = np.dot(spec, cs_y(λ_list)) * Δλ * 0.003975 * 683.002
        Z = np.dot(spec, cs_z(λ_list)) * Δλ * 0.003975 * 683.002
        return np.array([X, Y, Z])

    else:
        Δλ = (780 - 380) / (spec.shape[1] - 1)
        λ_list = np.arange(380.0, 780.00001, Δλ)

        cs_xyz = np.array([cs_x(λ_list), cs_y(λ_list), cs_z(λ_list)])
        return np.tensordot(spec, cs_xyz, axes=([1, 1])).T * Δλ * 0.003975 * 683.002


def spec_to_sRGB(spec):
    """
    Convert a spectrum to an RGB color.

    spec: numpy array in the form: np.array([spec1, spec2,...,specN)
    where spec1,spec2,...,specN  are lists with spectral intensities sampled on 380-780 nm interval.
    Number of samples of each spectral intensity list doesn't matter, but they must be equally spaced.

    """

    XYZ = spec_to_XYZ(spec)
    return XYZ_to_sRGB(XYZ)


def wavelength_to_XYZ(wavelength, intensity):

    Δλ = (780 - 380) / (len(λ_list) - 1)

    if (wavelength > 380) and (wavelength < 780):
        X = intensity * cs_x(wavelength) * Δλ * 0.003975 * 683.002
        Y = intensity * cs_y(wavelength) * Δλ * 0.003975 * 683.002
        Z = intensity * cs_z(wavelength) * Δλ * 0.003975 * 683.002
    else:
        X = intensity * 0.0
        Y = intensity * 0.0
        Z = intensity * 0.0

    return np.array([X, Y, Z])


def wavelength_to_sRGB(wavelength, intensity):

    XYZ = wavelength_to_XYZ(wavelength, intensity)
    return XYZ_to_sRGB(XYZ)
