import csv
import numpy as np
from plot_joints import scatter3d
import os

directory_path = './data'
csvfiles = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
for filename in csvfiles:
    wrists = np.genfromtxt(os.path.join(directory_path,filename), skip_header=1, delimiter=',')

    wrist_pos = wrists[:,1:4]

    scatter3d(wrist_pos[:,0],wrist_pos[:,1],wrist_pos[:,2],range(wrist_pos.shape[0]))