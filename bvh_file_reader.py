import zipfile
import os
from plot_joints import scatter3d
from bvh_helper import process_bvhfile
import numpy as np
import csv
from tqdm import tqdm


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

csvfile = open('mocap_data.csv','w')
csvwriter = csv.writer(csvfile, delimiter=',')
header_written = False

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
        for name in tqdm(zip.namelist()):
            if name.endswith('.bvh'):
                with zip.open(name,'r') as f:
                    skeleton = process_bvhfile(f)
                    header, coords = skeleton.get_frames_worldpos()
                    rot_header, rot = skeleton.get_frames_rotations()

                    # Rename joints 
                    rot_header = [h.replace('Hand.','Wrist.') for h in rot_header]
                    rot_header = [h.replace('ForeArm','Elbow') for h in rot_header]
                    header = [h.replace('Hand.','Wrist.') for h in header]
                    header = [h.replace('ForeArm','Elbow') for h in header]

                    # Add rot or pos to name
                    header = [h.replace('.','Pos.') for h in header]
                    rot_header = [h.replace('.','Rot.') for h in rot_header]

                    # Find forearm pos
                    elbow_idx = [i for i, h in enumerate(header) if 'Elbow' in h]
                    elbow = np.array(coords)[:,elbow_idx]
                    wrist_idx = [i for i, h in enumerate(header) if 'Wrist' in h]
                    wrist = np.array(coords)[:,wrist_idx]
                    forearm = (wrist + elbow)/2
                    forearm_name = [h.replace('Elbow','ForeArm') for h in np.array(header)[elbow_idx]]
                    header = np.concatenate((header,forearm_name))
                    coords = np.hstack((coords,forearm))

                    # Find knuckle pos
                    hand_end_idx = [i for i, h in enumerate(header) if 'HandEnd' in h]
                    hand_end = np.array(coords)[:,hand_end_idx]
                    knuckle = (wrist + hand_end)/2
                    knuckle_name = [h.replace('Elbow','Knuckle') for h in np.array(header)[elbow_idx]]
                    header = np.concatenate((header,knuckle_name))
                    coords = np.hstack((coords,knuckle))

                    
                    # Write header
                    if not header_written:
                        header_combo = np.concatenate((header,rot_header[1:]))
                        csvwriter.writerow(header)
                        header_written = True

                    # coords_and_rot = list(np.hstack((coords,rot[1:])))
                    csvwriter.writerows(coords)
                    # hand_idx = [i for i, label in enumerate(header) if 'Time' == label or 'Hand' in label]
                    # header = np.array(header)[hand_idx]

                    # coords = np.array(coords)[:,1:].reshape((64,31,3))
                    # for coord in coords:
                    #     scatter3d(coord[:,0],coord[:,2],coord[:,1],range(31),labels=header[1::3])

csvfile.close()


