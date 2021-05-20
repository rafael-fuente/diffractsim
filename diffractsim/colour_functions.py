import numpy as np
from pathlib import Path
from scipy.interpolate import CubicSpline
from .monochromatic_simulator import MonochromaticField

from .backend_functions import backend as bd

illuminant_d65 = np.loadtxt(Path(__file__).parent / "./data/illuminant_d65.txt", usecols=(1))

class ColourSystem:
    def __init__(self, spectrum_size = 400, spec_divisions = 40, clip_method = 1):
        global bd
        from .backend_functions import backend as bd

        self.spectrum_size = spectrum_size
        # import CIE XYZ standard observer color matching functions

        cmf = np.loadtxt(Path(__file__).parent / "./data/cie-cmf.txt", usecols=(1, 2, 3))
        
        self.Δλ = (779-380)/spectrum_size
        self.λ_list = np.linspace(380,779, spectrum_size)

        if spectrum_size == 400: 
            
            # CIE XYZ standard observer color matching functions
            self.cie_x =  cmf.T[0]
            self.cie_y =  cmf.T[1]
            self.cie_z =  cmf.T[2]

        else: #by default spectrum has a size of 400. If new size, we interpolate
            λ_list_old = np.linspace(380,779, 400)
            self.cie_x = np.interp(self.λ_list, λ_list_old, cmf.T[0])
            self.cie_y = np.interp(self.λ_list, λ_list_old, cmf.T[1])
            self.cie_z = np.interp(self.λ_list, λ_list_old, cmf.T[2])

        # if cupy backend:
        if bd != np:
            self.cie_x = bd.array(self.cie_x)
            self.cie_y = bd.array(self.cie_y)
            self.cie_z = bd.array(self.cie_z)

        self.cie_xyz = bd.array([self.cie_x, self.cie_y, self.cie_z])

        # used in spec_to_XYZ
        self.spec_divisions = spec_divisions
        self.cie_xyz_partitions = bd.hsplit(self.cie_xyz, self.spec_divisions)

        # XYZ to linear sRGB matrix
        self.T = bd.vstack(
            [[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.0570]]
        )
        
        #clip methods for negative sRGB values:

        self.CLIP_CLAMP_TO_ZERO = 0
        self.CLIP_ADD_WHITE = 1

        # default clip method
        self.clip_method = clip_method

    def XYZ_to_sRGB_linear(self, XYZ):
        """
        Convert a XYZ to a linear RGB color.


        XYZ: array with multiple XYZ colors in the form: bd.array([[X1,X2,X3...],
                                                                         [Y1,Y2,Y3...],
                                                                         [Z1,Z2,Z3...]])
        """

        rgb = bd.tensordot(self.T, XYZ, axes=([1, 0]))

        if self.clip_method == self.CLIP_CLAMP_TO_ZERO:
            # set negative rgb values to zero
            rgb = bd.where(rgb < 0, 0, rgb)
            return rgb

        if self.clip_method == self.CLIP_ADD_WHITE:
            # add enough white to make all rgb values nonnegative
            # find max negative rgb (or 0.0 if all non-negative), we need that much white
            rgb_min = bd.amin(rgb, axis=0)
            # get max positive component
            rgb_max = bd.amax(rgb, axis=0)

            # get scaling factor to maintain max rgb after adding white
            scaling = bd.where(rgb_max > 0.0, rgb_max / (rgb_max - rgb_min + 0.00001), bd.ones(rgb.shape))

            # add enough white to cancel this out, maintaining the maximum of rgb
            rgb = bd.where(rgb_min < 0.0, scaling * (rgb - rgb_min), rgb)
            return rgb


    def sRGB_linear_to_sRGB(self,rgb_linear):

        """
        Convert a linear RGB color to a non linear RGB color (gamma correction).


        RGB: numpy array with multiple RGB colors in the form: bd.array([[R1,R2,R3...],
                                                                         [G1,G2,G3...],
                                                                         [B1,B2,B3...]])
        """

        """sRGB standard for gamma inverse correction."""
        rgb = bd.where(
            rgb_linear <= 0.00304,
            12.92 * rgb_linear,
            1.055 * bd.power(rgb_linear, 1.0 / 2.4) - 0.055,
        )

        # clip intensity if needed (rgb values > 1.0) by scaling
        rgb_max = bd.amax(rgb, axis=0) + 0.00001  # avoid division by zero
        intensity_cutoff = 1.0
        rgb = bd.where(rgb_max > intensity_cutoff, rgb * intensity_cutoff / (rgb_max), rgb)

        return rgb


    def sRGB_to_sRGB_linear(self,rgb):
        """
        Convert a RGB color to a linear RGB color.


        RGB: numpy array with multiple RGB colors in the form: bd.array([[R1,R2,R3...],
                                                                         [G1,G2,G3...],
                                                                         [B1,B2,B3...]])
        """
        return bd.where(rgb <= 0.03928, rgb / 12.92, bd.power((rgb + 0.055) / 1.055, 2.4))


    def XYZ_to_sRGB(self, XYZ):
        """
        Convert a XYZ to an RGB color.


        XYZ: numpy array with multiple XYZ colors in the form: bd.array([[X1,X2,X3...],
                                                                         [Y1,Y2,Y3...],
                                                                         [Z1,Z2,Z3...]])
        """

        rgb_linear = self.XYZ_to_sRGB_linear(XYZ)
        rgb = self.sRGB_linear_to_sRGB(rgb_linear)

        return rgb


    def spec_to_XYZ(self, spec):
        """
        Convert a spectrum to an XYZ color.

        spec: numpy array in the form: bd.array([spec1, spec2,...,specN)
        where spec1,spec2,...,specN  are lists with spectral intensities sampled on 380-780 nm interval whose sampled are separated by 1nm
        Number of samples of each spectral intensity list doesn't matter, but they must be equally spaced.
        """

        
        
        if spec.ndim == 1:

            X = bd.dot(spec, self.cie_x) * self.Δλ * 0.003975 * 683.002
            Y = bd.dot(spec, self.cie_y) * self.Δλ * 0.003975 * 683.002
            Z = bd.dot(spec, self.cie_z) * self.Δλ * 0.003975 * 683.002
            return bd.array([X, Y, Z])

        else:
            return bd.tensordot(spec, self.cie_xyz, axes=([1, 1])).T * self.Δλ * 0.003975 * 683.002


    def spec_partition_to_XYZ(self, spec_partition, index = 0):
        """
        Convert a spectrum to an XYZ color.

        spec: numpy array in the form: bd.array([spec1, spec2,...,specN)
        where spec1,spec2,...,specN  are lists with spectral intensities sampled on 380-780 nm interval whose sampled are separated by 1nm
        Number of samples of each spectral intensity list doesn't matter, but they must be equally spaced.
        """

        if spec_partition.ndim == 1:
            X = bd.dot(spec_partition, self.cie_xyz_partitions[index][0]) * self.Δλ * 0.003975 * 683.002
            Y = bd.dot(spec_partition, self.cie_xyz_partitions[index][1]) * self.Δλ * 0.003975 * 683.002
            Z = bd.dot(spec_partition, self.cie_xyz_partitions[index][2]) * self.Δλ * 0.003975 * 683.002
            return bd.array([X, Y, Z])

        else:
            return bd.tensordot(spec_partition, self.cie_xyz_partitions[index], axes=([1, 1])).T * self.Δλ * 0.003975 * 683.002



    def spec_to_sRGB(self, spec):
        """
        Convert a spectrum to an RGB color.

        spec: numpy array in the form: bd.array([spec1, spec2,...,specN)
        where spec1,spec2,...,specN  are lists with spectral intensities sampled on 380-780 nm interval.
        Number of samples of each spectral intensity list doesn't matter, but they must be equally spaced.

        """

        XYZ = self.spec_to_XYZ(spec)
        return self.XYZ_to_sRGB(XYZ)


    def wavelength_to_XYZ(self,wavelength, intensity):


        if (wavelength > 380) and (wavelength < 780):
            index = int(wavelength-380)
            X = intensity * self.cie_x[index] * self.Δλ * 0.003975 * 683.002
            Y = intensity * self.cie_y[index] * self.Δλ * 0.003975 * 683.002
            Z = intensity * self.cie_z[index] * self.Δλ * 0.003975 * 683.002
        else:
            X = intensity * 0.0
            Y = intensity * 0.0
            Z = intensity * 0.0

        return bd.array([X, Y, Z])


    def wavelength_to_sRGB(self, wavelength, intensity):

        XYZ = self.wavelength_to_XYZ(wavelength, intensity)
        return self.XYZ_to_sRGB(XYZ)

    def wavelength_to_sRGB_linear(self, wavelength, intensity):

        XYZ = self.wavelength_to_XYZ(wavelength, intensity)
        return self.XYZ_to_sRGB_linear(XYZ)