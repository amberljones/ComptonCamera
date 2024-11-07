# -*- coding: utf-8 -*-
"""

@author: Amber
"""
import numpy as np 
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
        print(vect2)
        print(theta_c)

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
            Theta_size = Theta_size if not np.isnan(Theta_size) else np.pi/2
            Theta = np.linspace(0, 2 * np.pi, 2 * int(2 * np.pi // Theta_size + 1))

            for t in Theta:
                rotated_point = Rot.dot([r * sin_t_c * np.cos(t), r * sin_t_c * np.sin(t), r * cos_t_c])
                translated_point = rotated_point + detection_pair.scatter_position

                if np.all(0 <= translated_point) and np.all(translated_point < self.imaging_area):
                    points.append(translated_point)

        return np.array(points)

