# -*- coding: utf-8 -*-
"""

@author: Amber
"""
from voxel_cube import VoxelCube
from cone_calculator import ConeCalculator
import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from datetime import datetime


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
