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

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class IMU_Transformer(nn.Module):
    def __init__(self,tokens) -> None:
        super().__init__()
        self.encoder = IMU_Encoder(window_size,kernel_sizes,channels)
        self.encoder.load_state_dict(torch.load('model/encoder_weights'))
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=channels[-1],nhead=8,norm_first=True,batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer=self.transformer_layer,num_layers=10)
        self.pos_enc = PositionalEncoding(channels[-1])
        self.flatten = nn.Flatten()
        self.relu = nn.GELU()
        self.dropout = nn.Dropout1d()

        self.cls_token = nn.Parameter(torch.zeros((tokens, channels[-1])), requires_grad=True)
        self.lin1 = nn.Linear(18,tokens)
        self.lin2 = nn.Linear(tokens,tokens)
        self.lin3 = nn.Linear(tokens,tokens)
        self.lin4 = nn.Linear(tokens,18)

    def forward(self, x, start):
        with torch.no_grad():
            x = self.encoder(x)
            x = self.relu(x)
        x = x.permute(2,0,1)
        cls_token = self.cls_token.unsqueeze(1).repeat(1,x.shape[1],1)
        x = torch.cat([F.pad(start,(0,tokens-18),'constant',0).unsqueeze(0), x])
        x += self.pos_enc(x)
        x = self.transformer(x)[0][:,:18]
        x = self.dropout(x)
        x = self.relu(x)
        x = self.lin1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.lin2(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.lin3(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.lin4(x)
        
        leftShoulder = start[:,:3]
        rightShoulder = start[:,3:6]

        leftElbowLen = torch.sqrt(torch.sum((start[:,:3] - start[:,6:9])**2,axis=1))
        leftElbow = leftShoulder + F.normalize(x[:,6:9] + start[:,6:9]) * leftElbowLen.repeat(3,1).transpose(0,1)
        
        rightElbowLen = torch.sqrt(torch.sum((start[:,3:6] - start[:,9:12])**2,axis=1))
        rightElbow = rightShoulder + F.normalize(x[:,9:12] + start[:,9:12]) * rightElbowLen.repeat(3,1).transpose(0,1)

        leftWristLen = torch.sqrt(torch.sum((start[:,6:9] - start[:,12:15])**2,axis=1))
        leftWrist = leftElbow + F.normalize(start[:,12:15] + start[:,12:15]) * leftWristLen.repeat(3,1).transpose(0,1)

        rightWristLen = torch.sqrt(torch.sum((start[:,9:12] - start[:,15:18])**2,axis=1))
        rightWrist = rightElbow + F.normalize(x[:,15:18] + start[:,15:18]) * rightWristLen.repeat(3,1).transpose(0,1)

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
weight_decay = 0.01
holdout = 0.2
batch_size = 128
tokens = 64
learning_rate = 1e-4

dtype = torch.float32

encoder = IMU_Encoder(window_size,kernel_sizes,channels)
transformer = IMU_Transformer(tokens).to(device,dtype=dtype)
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

if __name__ == '__main__':
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
            delta_penalty = F.relu(torch.abs(y[:,12:,1] - y_pred[:,12:]) - (0.2))
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

    pass