import numpy as np
import os

imu_directory_path = 'data/VirtualIMU'
full_dir_path = 'data/AllCoords'

class VirtualIMU_Loader():
    def __init__(self) -> None:
          pass

    def load_files(self):
        for folder in os.listdir(imu_directory_path):
            mocap_data = np.genfromtxt(os.path.join(full_dir_path,f'{folder}.csv'), skip_header=1, delimiter=',')
            for file in os.listdir(os.path.join(imu_directory_path,folder)):
                    imu_data = np.genfromtxt(os.path.join(imu_directory_path,folder,file), skip_header=1, delimiter=',')
                    yield imu_data, mocap_data

    def extract_dataset(self, window_size=None, overlap=None):
        dataset = []
        mocap_dataset = []
        for data, mocap in self.load_files():
            if window_size and overlap:
                start = 1
                end = start + window_size
                windows = []
                mocap_windows = []
                while end < data.shape[0]:
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
                
                if len(windows) > 0:
                    dataset.append(windows)
                    mocap_dataset.append(mocap_windows)
            else:
                dataset.append([np.transpose(data)])
        if not window_size or not overlap:
             return dataset
        return np.concatenate(dataset), np.concatenate(mocap_dataset)