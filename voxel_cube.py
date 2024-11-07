# -*- coding: utf-8 -*-
"""

@author: Amber
"""
import numpy as np
from cone_calculator import ConeCalculator
from datetime import datetime

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