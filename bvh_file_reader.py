import zipfile
from bvh import Bvh
import os
from plot_joints import scatter3d
from bvh_helper import process_bvhfile
import numpy as np

def get_children(joint):
    return [i for i in joint.filter('JOINT')]

def get_offset(joint):
    x,y,z = next(joint.filter('OFFSET')).value[1:]
    return np.array([float(x),float(y),float(z)])

def process_bvh(data):
    mocap = Bvh(data)
    root = next(mocap.root.filter('ROOT'))
    currJoints = get_children(root)
    offset = get_offset(root)
    offsets = {root.name:offset}
    while len(currJoints)>0:
        joint = currJoints.pop()
        parent = joint.parent.name
        offsets[joint.name] = offsets[parent] + get_offset(joint)
        currJoints.extend(get_children(joint))

    for joint in offsets.keys():
        magnitude = np.sqrt(np.sum(offsets[joint]**2))
        theta_x = np.arctan2(offsets[joint][1],offsets[joint][2])
        theta_y = np.arctan2(offsets[joint][0],offsets[joint][2])
        theta_z = np.arctan2(offsets[joint][0],offsets[joint][1])
        x = magnitude * np.cos(theta_x)-np.sin(theta_x)
        y = magnitude * np.cos(theta_y)-np.sin(theta_y)
        z = magnitude * np.cos(theta_z)-np.sin(theta_z)
        offsets[joint] = np.array([x,y,z])
    p = np.array([offsets[joint] for joint in offsets.keys()])
    scatter3d(p[:,0],p[:,1],p[:,1],range(22),labels=offsets.keys())



# Directory containing the .tar.bz2 files
directory_path = './data'

# List all .tar.bz2 files in the directory
bvh_files = [f for f in os.listdir(directory_path) if 'bvh' in f]

for bvh_filename in bvh_files:
    print(f'Reading: {bvh_filename}')
    path = os.path.join(directory_path, bvh_filename)
    if bvh_filename.endswith('.bvh'):
        with open(path,'r') as f:
            data = f.read()
            process_bvhfile(data)
    if zipfile.is_zipfile(path):
        zip = zipfile.ZipFile(path)
        for name in zip.namelist():
            if name.endswith('.bvh'):
                with zip.open(name,'r') as f:
                    skeleton = process_bvhfile(f)
                    coords = np.array(skeleton.get_frames_worldpos()[1])[:,1:].reshape((64,27,3))
                    for i in range(coords.shape[0]):
                        scatter3d(coords[0,:,0],coords[0,:,2],coords[0,:,1],range(27))

