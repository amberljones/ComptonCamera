# -*- coding: utf-8 -*-
"""

@author: Amber
"""

import numpy as np
import scipy.constants as constants

# Constants
ELECTRON_MASS_KEV = (constants.electron_mass * constants.c ** 2) / (constants.electron_volt * 10 ** 3)  


class DetectionPair:
    """
    Class representing a detection pair for a Compton scatter event.
    """

    def __init__(self, scatter_position, absorption_position, absorption_energy, scatter_energy):
        """
        Initialize a DetectionPair object.

        :param scatter_position: Coordinates of scatter (array-like)
        :param absorption_position: Coordinates of absorption (array-like)
        :param absorption_energy: Absorption energy in keV
        :param scatter_energy: Scatter energy in keV
        """
        self.scatter_position = np.float64(scatter_position)
        self.absorption_position = np.float64(absorption_position)
        self.line_vector = (self.scatter_position - self.absorption_position) / np.linalg.norm(
            self.scatter_position - self.absorption_position)
        self.scatter_energy = np.float64(scatter_energy)
        self.initial_energy = np.float64(absorption_energy)
        self.scatter_angle = self.calculate_scatter_angle(absorption_energy, scatter_energy)
    
    def __repr__(self):
        return f"Person(scatter energy={self.initial_energy}, scatter locations={self.scatter_position}, absorption energies={self.scatter_energy}, absorption locations={self.absorption_position})"

    @staticmethod
    def calculate_scatter_angle(absorption_energy, scatter_energy):
        """
        Calculate Compton Scattering Angle in radians.

        :param initial_energy: Initial energy in keV
        :param scatter_energy_deposited: Scatter energy deposited in keV
        :return: Scattering angle in radians
        """
        return np.arccos(
        (1 - (1 / absorption_energy) - 1 / (absorption_energy - scatter_energy)) * ELECTRON_MASS_KEV)