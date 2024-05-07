import sys
import os
sys.path.append('./')
from utils.virtualIMU_loader import VirtualIMU_Loader
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from plot_joints import scatter3d


imu_directory_path = 'data/VirtualIMU'
full_dir_path = 'data/AllCoords'
folder = 'Male1_B26_WalkToSkip'
file = 'synthAug0VidIMU.csv'

window_size = 15
overlap = 0.5


mocap = np.genfromtxt(os.path.join(full_dir_path,f'{folder}.csv'), skip_header=1, delimiter=',')
data = np.genfromtxt(os.path.join(imu_directory_path,folder,file), skip_header=1, delimiter=',')


dataset = []
mocap_dataset = []
start = 1
end = start + window_size
windows = []
mocap_windows = []
while end < data.shape[0]:
    max_acc = np.round(np.max(np.abs(data[start:end,1:7]),axis=0),2)
    max_gyro = np.round(np.max(np.abs(data[start:end,7:]),axis=0),2)
    # if np.any(max_acc > 250) or np.any(max_gyro > 360):
    #      print(max_acc)
    #      print(max_gyro)

    windows.append(np.transpose(data[start:end,1:]))
    start_time = data[start,0]
    diff = np.abs(mocap[:,1]-start_time)
    mocap_start = mocap[np.argmin(diff),1:]
    end_time = data[end,0]
    diff = np.abs(mocap[:,1]-end_time)
    mocap_end = mocap[np.argmin(diff),1:]
    mocap_windows.append(np.transpose((mocap_start,mocap_end)))
    start += int(window_size * overlap)
    end += int(window_size * overlap)

mocap_windows = np.array(mocap_windows)
fig, scatter = scatter3d(mocap_windows[0,0::3,1],mocap_windows[0,2::3,1],mocap_windows[0,1::3,1],np.ones(6))

def update(idx, data, scatter):
    scatter._offsets3d = (data[idx,0::3,0],data[idx,2::3,0],data[idx,1::3,0])

ani = anim.FuncAnimation(fig, update, len(mocap_windows), fargs=(mocap_windows,scatter)).save('ani.gif','ffmpeg',1)
plt.show()