import zipfile
import os
import sys
sys.path.append('./')
from plot_joints import scatter3d
import numpy as np
import csv
from tqdm import tqdm
from VirtualIMU.utils.extract_coordinates import extractWristCoordinatesBVH, extractAllCoordinatesBVH

# Directory containing the bvh files
directory_path = './data'

# Find all files containing 'bvh' in filename
bvh_files = [f for f in os.listdir(directory_path) if 'bvh' in f]

# List of all already extracted bvh files
wrist_csvfiles = [f.split('.')[0] for f in os.listdir(f'{directory_path}/WristCoords') if f.endswith('.csv')]
full_csvfiles = [f.split('.')[0] for f in os.listdir(f'{directory_path}/AllCoords') if f.endswith('.csv')]

for bvh_filename in bvh_files:
    print(f'Reading: {bvh_filename}')
    path = os.path.join(directory_path, bvh_filename)

    # Currently not fully implemented for unzipped
    if bvh_filename.endswith('.bvh'):
        with open(path,'r') as f:
            data = f.read().decode('ASCII')
            extractWristCoordinatesBVH(f,f'data/WristCoords/{bvh_filename.split(".")[0]}.csv')

    if zipfile.is_zipfile(path):
        zip = zipfile.ZipFile(path)

        for name in tqdm(zip.namelist()):
            print(f'Parsing {name.split(".")[0]}')
            if name.endswith('.bvh'):
                with zip.open(name,'r') as f:
                    # BVH files we used were byte strings, needed to be decoded to ascii first
                    data = f.read().decode('ASCII')

                # Extract wrist pos and orientation
                if name.split('.')[0] in wrist_csvfiles:
                    print(f'{name.split(".")[0]} wrist is already parsed')
                else:   
                    extractWristCoordinatesBVH(data,f'data/WristCoords/{name.split(".")[0]}.csv')
                
                # Extract ground truth data
                if name.split('.')[0] in full_csvfiles:
                    print(f'{name.split(".")[0]} full is already parsed')
                else: 
                    extractAllCoordinatesBVH(data,f'data/AllCoords/{name.split(".")[0]}.csv')