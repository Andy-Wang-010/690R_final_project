import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./')
from model.IMU_Encoder import IMU_Encoder
from utils.virtualIMU_loader import VirtualIMU_Loader
from tqdm import tqdm
import math
import numpy as np

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

print(f'Using Device: {device}')

class IMU_CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = IMU_Encoder(100,[1,1,1],[12,64,64,64])
        self.encoder.load_state_dict(torch.load('model/encoder_weights'))
        self.conv1 = nn.Conv1d(64,128,5)
        self.conv2 = nn.Conv1d(128,128,3)
        self.conv3 = nn.Conv1d(128,128,3)
        self.conv4 = nn.Conv1d(128,128,3)
        self.conv4 = nn.Conv1d(128,128,3)
        size = 15
        self.lin1 = nn.Linear(size*128,512)
        self.lin2 = nn.Linear(512,18)
        self.lin3 = nn.Linear(256,64)
        self.lin4 = nn.Linear(64,18)

        self.drop = nn.Dropout1d()
        self.relu = nn.LeakyReLU()
        self.flat = nn.Flatten()

    def forward(self, x, start):
        with torch.no_grad():
            x = self.encoder(x)
            x = self.relu(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.flat(x)
        x = self.drop(x)
        x = self.lin1(x)
        x = self.drop(x)
        x = self.relu(x)
        x = self.lin2(x)
        leftShoulder = x[:,:3]
        rightShoulder = x[:,3:6]

        leftElbowLen = torch.sqrt(torch.sum((start[:,:3] - start[:,6:9])**2,axis=1))
        leftElbow = leftShoulder + F.normalize(x[:,6:9]) * leftElbowLen.repeat(3,1).transpose(0,1)
        
        rightElbowLen = torch.sqrt(torch.sum((start[:,3:6] - start[:,9:12])**2,axis=1))
        rightElbow = rightShoulder + F.normalize(x[:,9:12]) * rightElbowLen.repeat(3,1).transpose(0,1)

        leftWristLen = torch.sqrt(torch.sum((start[:,6:9] - start[:,12:15])**2,axis=1))
        leftWrist = leftElbow + F.normalize(start[:,12:15]) * leftWristLen.repeat(3,1).transpose(0,1)

        rightWristLen = torch.sqrt(torch.sum((start[:,9:12] - start[:,15:18])**2,axis=1))
        rightWrist = rightElbow + F.normalize(x[:,15:18]) * rightWristLen.repeat(3,1).transpose(0,1)

        x = torch.cat((
            leftShoulder,
            rightShoulder,
            leftElbow,
            rightElbow,
            leftWrist,
            rightWrist
        ),dim=1)


        return x


# Tuneable Parameters

window_size = 25
overlap = 0.5
kernel_sizes = [1,1,1]
channels = [12,64,64,64]
num_epochs = 1e4
weight_decay = 0.001
holdout = 0.2
batch_size = 7368
tokens = 64
learning_rate = 1e-3

dtype = torch.float32
eps = torch.tensor(1e-7,device=device,dtype=dtype)

if __name__ == '__main__':
    # encoder = IMU_Encoder(window_size,kernel_sizes,channels)
    transformer = IMU_CNN().to(device,dtype=dtype)
    optim = torch.optim.Adam(transformer.parameters(),weight_decay=weight_decay, lr=learning_rate,amsgrad=True)
    loss_criterion = torch.nn.MSELoss()

    vIMU_loader = VirtualIMU_Loader()
    dataset, mocap_dataset = vIMU_loader.extract_dataset(window_size,overlap)
        
    data = torch.tensor(dataset,device=device,dtype=dtype)
    mocap_data = torch.tensor(mocap_dataset,device=device,dtype=dtype)
    X_train = data[int(holdout*data.shape[0]):]
    X_test = data[:int(holdout*data.shape[0])]
    y_train = mocap_data[int(holdout*data.shape[0]):]
    y_test = mocap_data[:int(holdout*data.shape[0])]

    torch.cuda.empty_cache()
    epochs = tqdm(range(int(num_epochs)))
    # with torch.autograd.detect_anomaly():
    best_test = np.infty
    best_model = None
    for i in epochs:
        for i in range(int(X_train.shape[0]/batch_size)):
            X = X_train[i*batch_size:(i+1)*batch_size]
            y = y_train[i*batch_size:(i+1)*batch_size]
            optim.zero_grad()
            transformer.train()
            y_pred = transformer(X,y[:,:,0])
            # y_pred += torch.rand_like(y_pred)*0.0001
            delta_penalty = F.relu(torch.abs(y[:,12:,1] - y_pred[:,12:]) - (0.6/30))
            loss = loss_criterion(y[:,:,1],y_pred) + delta_penalty.mean()
            loss.backward()
            optim.step()

        with torch.no_grad():
            transformer.eval()
            y_pred = transformer(X_test,y_test[:,:,0])
            test_loss = loss_criterion(y_test[:,:,1],y_pred)
            if test_loss.detach().cpu().item() < best_test and not torch.isnan(y_pred).any():
                best_model = transformer.state_dict()
                best_test = test_loss.detach().cpu().item()
            epochs.set_description('TRAIN: {:.7e}, TEST: {:.7e}'.format(loss.cpu().item(),test_loss.cpu().item()))

    # torch.save(encoder.state_dict(),'model/encoder_weights')
    # encoder.load_state_dict(torch.load('model/encoder_weights'))
    # x = encoder(data[0])
    pass