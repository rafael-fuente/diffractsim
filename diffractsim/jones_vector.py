"""
MPL 2.0 Clause License 

Copyright (c) 2022, Rafael de la Fuente
All rights reserved.
"""

import numpy as np

class JonesVector:
    def __init__(self, s11, s12, s21, s22):
        """
        Initializes the Jones vector representing a linear polarizer or other optical element.

        Parameters
        ----------
        s11 : float
            Element of the Jones matrix.
        s12 : float
            Element of the Jones matrix.
        s21 : float
            Element of the Jones matrix.
        s22 : float
            Element of the Jones matrix.
        """
        self.matrix = np.array([[s11, s12], [s21, s22]])

    def apply_to(self, electric_field):
        """
        Applies the Jones vector to an electric field.

        Parameters
        ----------
        electric_field : numpy.ndarray
            The electric field as a 2D array where each row represents a point in space and each column represents the electric field components (Ex, Ey).

        Returns
        -------
        numpy.ndarray
            The transformed electric field.
        """
        return np.dot(self.matrix, electric_field)

    @staticmethod
    def linear_polarizer(angle):
        """
        Creates a Jones vector for a linear polarizer.

        Parameters
        ----------
        angle : float
            The angle of the polarizer in radians.

        Returns
        -------
        JonesVector
            The Jones vector representing the linear polarizer.
        """
        s11 = np.cos(2 * angle)
        s12 = 0
        s21 = 0
        s22 = -np.cos(2 * angle)
        return JonesVector(s11, s12, s21, s22)

    @staticmethod
    def circular_polarizer():
        """
        Creates a Jones vector for a circular polarizer.

        Returns
        -------
        JonesVector
            The Jones vector representing the circular polarizer.
        """
        s11 = 0.5
        s12 = -0.5j
        s21 = 0.5j
        s22 = 0.5
        return JonesVector(s11, s12, s21, s22)