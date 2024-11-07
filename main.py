# -*- coding: utf-8 -*-
"""

@author: Amber
"""
from detection_pair import DetectionPair
from voxel_cube_generator import VoxelCubeGenerator
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import ast 


df = pd.read_csv('konbert-output-4ef693f9.csv')
#get rid of channel numbers 
df = df.iloc[0:, :]

#convert positions to numpy arrays 
df['scatter locations'] = df['scatter locations'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float64))
df['absorption locations'] = df['absorption locations'].apply(lambda x: np.array(ast.literal_eval(x), dtype=np.float64))

#create DetectionPair object instances
pairs = df.apply(lambda row: DetectionPair(row['scatter locations'], row['absorption locations'], row['absorption energies'], row['scatter energy']), axis=1)

generator = VoxelCubeGenerator(imaging_area=np.array([80, 80, 40]), voxel_length=0.5 , pairs=pairs)
generator.generate_voxel_cube()
generator.save_result("FinalMatrix") 

df[['x', 'y', 'z']] = pd.DataFrame(df['absorption locations'].tolist(), index=df.index)

# Extract x, y, z columns
x = df['x']
y = df['y']
z = df['z']

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='b', marker='o')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
