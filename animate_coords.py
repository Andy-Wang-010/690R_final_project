import sys
import os
sys.path.append('./')
from model.IMU_CNN import IMU_CNN
from model.IMU_Transformer import IMU_Transformer
from utils.virtualIMU_loader import VirtualIMU_Loader
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from plot_joints import scatter3d
import torch


imu_directory_path = 'data/VirtualIMU'
full_dir_path = 'data/AllCoords'
folder = 'Male1_B26_WalkToSkip'
file = 'synthAug0VidIMU.csv'

window_size = 25
overlap = 0.5
holdout = 0.2
device = 'cuda'
dtype = torch.float32



vIMU_loader = VirtualIMU_Loader()
dataset, mocap_dataset = vIMU_loader.extract_dataset(window_size,overlap)
    
data = torch.tensor(dataset,device=device,dtype=dtype)
mocap_data = torch.tensor(mocap_dataset,device=device,dtype=dtype)
X_train = data[int(holdout*data.shape[0]):]
X_test = data[:int(holdout*data.shape[0])]
y_train = mocap_data[int(holdout*data.shape[0]):]
y_test = mocap_data[:int(holdout*data.shape[0])]

X_test = X_test[:1000]
y_test = y_test[:1000]

fig, scatter, plots = scatter3d(mocap_dataset[0,0::3,1],mocap_dataset[0,2::3,1],mocap_dataset[0,1::3,1],np.ones(6),buffer=0.6)

def update(idx, data, scatter):
    scatter._offsets3d = (data[idx,0::3,0],data[idx,2::3,0],data[idx,1::3,0])

def y_pred_update(idx, data, scatter, plots):
    mid_point = (data[idx,0:3] + data[idx,3:6])/2
    x,y,z = data[idx,0::3],data[idx,2::3],data[idx,1::3]
    x -= mid_point[0]
    y -= mid_point[1]
    z -= mid_point[2]
    scatter._offsets3d = (x,y,z)
    

    plots[0][0].set_data([x[0],x[1]],[y[0],y[1]])
    plots[1][0].set_data([x[0],x[2]],[y[0],y[2]])
    plots[2][0].set_data([x[1],x[3]],[y[1],y[3]])
    plots[3][0].set_data([x[2],x[4]],[y[2],y[4]])
    plots[4][0].set_data([x[3],x[5]],[y[3],y[5]])
    plots[0][0].set_3d_properties([z[0],z[1]])
    plots[1][0].set_3d_properties([z[0],z[2]])
    plots[2][0].set_3d_properties([z[1],z[3]])
    plots[3][0].set_3d_properties([z[2],z[4]])
    plots[4][0].set_3d_properties([z[3],z[5]])

predict = True
compounding = True
predictions = []
if not predict:
    ani = anim.FuncAnimation(fig, y_pred_update, len(y_test.detach().cpu().numpy()), fargs=(y_test[:,:,1].detach().cpu().numpy(),scatter,plots))
    ani.save('groud_truth.gif','ffmpeg',5)
else:
    model = IMU_Transformer(64).to(device,dtype)
    model.load_state_dict(torch.load('model/best_transformer_2'))
    if compounding:
        with torch.no_grad():
            y_pred = model.forward(X_test[0:1],y_test[0:1,:,0])
            for i in range(1000):
                predictions.append(y_pred.flatten().detach().cpu().numpy())
                y_pred = model.forward(X_test[i:i+1],y_pred)
        ani = anim.FuncAnimation(fig, y_pred_update, len(X_test), fargs=(np.array(predictions),scatter,plots))
        ani.save('pred_compound_cnn.gif','ffmpeg',5)
    else:
        with torch.no_grad():
            y_pred = model.forward(X_test,y_test[:,:,0])
        ani = anim.FuncAnimation(fig, y_pred_update, len(X_test), fargs=(y_pred.detach().cpu().numpy(),scatter,plots))
        ani.save('pred_one_cnn.gif','ffmpeg',5)

plt.show()