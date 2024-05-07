import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./')
from model.IMU_Encoder import IMU_Encoder
from utils.virtualIMU_loader import VirtualIMU_Loader
from tqdm import tqdm
import math

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
        size = 5
        self.flat = nn.Flatten()
        self.lin1 = nn.Linear(size*128,2048)
        self.lin2 = nn.Linear(2048,512)
        self.lin3 = nn.Linear(512,64)
        self.lin4 = nn.Linear(64,18)

        self.relu = nn.ReLU()

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
        x = self.lin1(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = self.relu(x)
        x = self.lin4(x)
        x = x + start
        return x


# Tuneable Parameters

window_size = 15
overlap = 0.5
kernel_sizes = [1,1,1]
channels = [12,64,64,64]
num_epochs = 1e4
weight_decay = 0.01
holdout = 0.2
batch_size = 128
tokens = 64
learning_rate = 1e-4

dtype = torch.float32

# encoder = IMU_Encoder(window_size,kernel_sizes,channels)
transformer = IMU_CNN().to(device,dtype=dtype)
optim = torch.optim.Adam(transformer.parameters(),weight_decay=weight_decay, lr=learning_rate)
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
for i in epochs:
    for i in range(int(X_train.shape[0]/batch_size)):
        X = X_train[i*batch_size:(i+1)*batch_size]
        y = y_train[i*batch_size:(i+1)*batch_size]
        optim.zero_grad()
        y_pred = transformer(X,y[:,:,0])
        loss = loss_criterion(y[:,:,1],y_pred)
        loss.backward()
        optim.step()

    with torch.no_grad():
        transformer.eval()
        y_pred = transformer(X_test,y_test[:,:,0])
        loss = loss_criterion(y_test[:,:,1],y_pred)
        epochs.set_description('LOSS: {:.7e}'.format(loss.cpu().item()))

# torch.save(encoder.state_dict(),'model/encoder_weights')
# encoder.load_state_dict(torch.load('model/encoder_weights'))
# x = encoder(data[0])
pass