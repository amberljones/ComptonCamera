# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 19:29:28 2024

@author: Amber
"""

import math
import multiprocessing
from datetime import datetime
from multiprocessing import Pool
import numpy as np
import scipy.constants as constants
from tqdm import tqdm
import warnings

# Ignore specific warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in double_scalars")
warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")
warnings.filterwarnings("ignore", message="invalid value encountered in arcsin")

# Constants
ELECTRON_MASS_KEV = (constants.electron_mass * constants.c ** 2) / (constants.electron_volt * 10 ** 3)  


class DetectionPair:
    """
    Class representing a detection pair for a Compton scatter event.
    """

    def __init__(self, scatter_position, absorption_position, initial_energy, scatter_energy):
        """
        Initialize a DetectionPair object.

        :param scatter_position: Coordinates of scatter (array-like)
        :param absorption_position: Coordinates of absorption (array-like)
        :param initial_energy: Initial energy in keV
        :param scatter_energy: Scatter energy in keV
        """
        self.scatter_position = np.array(scatter_position, dtype=np.float64)
        self.absorption_position = np.array(absorption_position, dtype=np.float64)
        self.line_vector = (self.scatter_position - self.absorption_position) / np.linalg.norm(
            self.scatter_position - self.absorption_position)
        self.scatter_energy = scatter_energy
        self.scatter_angle = self.calculate_scatter_angle(initial_energy, scatter_energy)
    
    def __repr__(self):
        return f"Person(scatter energy={self.initial_energy}, scatter locations={self.scatter_position}, absorption energies={self.scatter_energy}, absorption locations={self.absorption_position})"

    @staticmethod
    def calculate_scatter_angle(initial_energy, scatter_energy):
        """
        Calculate Compton Scattering Angle in radians.

        :param initial_energy: Initial energy in keV
        :param scatter_energy_deposited: Scatter energy deposited in keV
        :return: Scattering angle in radians
        """
        return np.arccos(
        1 - ((1 / (initial_energy - scatter_energy)) - 1 / initial_energy) * ELECTRON_MASS_KEV)


class ConeCalculator:
    """
    Class responsible for calculating the cone points based on detection pairs.
    """

    def __init__(self, imaging_area, voxel_length):
        """
        Initialize ConeCalculator object.

        :param imaging_area: 3D imaging area dimensions
        :param voxel_length: Size of each voxel
        """
        self.imaging_area = imaging_area
        self.voxel_length = voxel_length

    def rotation_matrix(self, vec1, vec2):
        """
        Calculate the rotation matrix from one vector to another.

        :param vec1: Initial vector
        :param vec2: Target vector
        :return: 3x3 rotation matrix
        """
        if np.allclose(vec1, vec2 / np.linalg.norm(vec2)):
            return np.eye(3)

        a = vec1 / np.linalg.norm(vec1)
        b = vec2 / np.linalg.norm(vec2)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

    def calculate_cone_polars(self, detection_pair):
        """
        Calculate the cone points for a detection pair.

        :param detection_pair: DetectionPair object
        :return: List of translated and rotated points
        """
        vect1 = [0, 0, 1]  # Default z-axis vector
        vect2 = detection_pair.line_vector
        theta_c = detection_pair.scatter_angle

        # Find max value of R (distance to imaging area boundaries)
        edge_points = [[0, 0, 0], [self.imaging_area[0], 0, 0], [0, self.imaging_area[1], 0],
                       [self.imaging_area[0], self.imaging_area[1], 0],
                       [0, 0, self.imaging_area[2]], [self.imaging_area[0], 0, self.imaging_area[2]],
                       [0, self.imaging_area[1], self.imaging_area[2]],
                       [self.imaging_area[0], self.imaging_area[1], self.imaging_area[2]]]

        R_max = max(np.linalg.norm(np.array(edge_point) - detection_pair.scatter_position) for edge_point in edge_points)
        R_min = 0

        R = np.arange(R_min, R_max, self.voxel_length / 2)
        points = []
        sin_t_c = np.sin(theta_c)
        cos_t_c = np.cos(theta_c)

        Rot = self.rotation_matrix(vect1, vect2)

        for r in R:
            Theta_size = 2 * np.arcsin((self.voxel_length / 2) / (r * sin_t_c))
            Theta_size = np.pi if Theta_size == 0 else Theta_size
            Theta = np.linspace(0, 2 * np.pi, 2 * int(2 * np.pi // Theta_size + 1))

            for t in Theta:
                rotated_point = Rot.dot([r * sin_t_c * np.cos(t), r * sin_t_c * np.sin(t), r * cos_t_c])
                translated_point = rotated_point + detection_pair.scatter_position

                if np.all(0 <= translated_point) and np.all(translated_point < self.imaging_area):
                    points.append(translated_point)

        return np.array(points)


class VoxelCube:
    """
    Class representing a voxel cube that stores cone intersections for imaging.
    """

    def __init__(self, imaging_area, voxel_length):
        """
        Initialize VoxelCube object.

        :param imaging_area: 3D imaging area dimensions
        :param voxel_length: Size of each voxel
        """
        self.imaging_area = np.array(imaging_area)
        self.voxel_length = voxel_length
        self.voxels_per_side = np.array(imaging_area / voxel_length, dtype=int)
        self.voxel_cube = np.zeros(self.voxels_per_side, dtype=int)

    def add_cone_to_cube(self, detection_pair):
        """
        Add the cone for a detection pair to the voxel cube.

        :param detection_pair: DetectionPair object
        """
        calculator = ConeCalculator(self.imaging_area, self.voxel_length)
        cone_points = calculator.calculate_cone_polars(detection_pair)

        for point in cone_points:
            self.voxel_cube[int(point[0] // self.voxel_length),
                            int(point[1] // self.voxel_length),
                            int(point[2] // self.voxel_length)] = 1

    def save_matrix(self, file_name):
        """
        Save the voxel cube matrix to a text file.

        :param file_name: Name of the file to save the matrix
        """
        reshaped_arr = self.voxel_cube.reshape(self.voxel_cube.shape[0], -1)
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        file_path = f"SavedVoxelCubes/{file_name}_{timestamp}_{self.voxel_cube.shape}.txt"

        np.savetxt(file_path, reshaped_arr)
        loaded_arr = np.loadtxt(file_path)
        loaded_arr_reshaped = loaded_arr.reshape(loaded_arr.shape[0], self.voxel_cube.shape[1], self.voxel_cube.shape[2])

        if np.array_equal(loaded_arr_reshaped, self.voxel_cube):
            print("Matrix saved and verified successfully.")
        else:
            print("Matrix verification failed.")


class VoxelCubeGenerator:
    """
    Class responsible for generating the voxel cube based on multiple detection pairs.
    """

    def __init__(self, imaging_area, voxel_length, pairs):
        """
        Initialize VoxelCubeGenerator object.

        :param imaging_area: 3D imaging area dimensions
        :param voxel_length: Size of each voxel
        :param pairs: List of DetectionPair objects
        """
        self.imaging_area = imaging_area
        self.voxel_length = voxel_length
        self.pairs = pairs
        self.voxel_cube = VoxelCube(imaging_area, voxel_length)

    def generate_voxel_cube_multi(self):
        """
        Generate the voxel cube by processing detection pairs.
        """
        with Pool(multiprocessing.cpu_count()) as pool:
            pbar = tqdm(total=len(self.pairs))
            args = [(self.imaging_area, self.voxel_length, self.voxel_cube.voxels_per_side, pair) for pair in self.pairs]

            results = pool.imap(self.calculate_voxel_cone_cube, args)
            for result in results:
                self.voxel_cube.voxel_cube += result
                pbar.update(1)
            pbar.close()
            
    def generate_voxel_cube(self):
        """
        Generate the voxel cube by processing detection pairs.
        """
        pbar = tqdm(total=len(self.pairs))
        for pair in self.pairs:
            # Prepare arguments for calculate_voxel_cone_cube
            args = (self.imaging_area, self.voxel_length, self.voxel_cube.voxels_per_side, pair)
            
            # Call the function directly without multiprocessing
            result = self.calculate_voxel_cone_cube(args)
            
            # Update the voxel cube with the result
            self.voxel_cube.voxel_cube += result
            pbar.update(1)
            
        pbar.close()

    def calculate_voxel_cone_cube(self, args):
        """
        Create a voxel cone cube for a given detection pair.

        :param args: Tuple containing imaging_area, voxel_length, voxels_per_side, and detection_pair
        :return: Voxel cube for the detection pair
        """
        imaging_area, voxel_length, voxels_per_side, detection_pair = args
        cone_calculator = ConeCalculator(imaging_area, voxel_length)
        cone_points = cone_calculator.calculate_cone_polars(detection_pair)

        voxel_cube_local = np.zeros(voxels_per_side, dtype=int)
        for point in cone_points:
            voxel_cube_local[int(point[0] // voxel_length),
                             int(point[1] // voxel_length),
                             int(point[2] // voxel_length)] = 1
        return voxel_cube_local

    def save_result(self, file_name):
        """
        Save the generated voxel cube matrix.

        :param file_name: Name of the file to save the matrix
        """
        self.voxel_cube.save_matrix(file_name)



