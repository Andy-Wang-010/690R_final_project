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
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=channels[-1],nhead=8)
        self.transformer = nn.TransformerEncoder(encoder_layer=self.transformer_layer,num_layers=10)
        self.pos_enc = PositionalEncoding(channels[-1])
        self.flatten = nn.Flatten()
        self.relu = nn.GELU()
        self.batchnorm = nn.BatchNorm1d(channels[-1])

        self.cls_token = nn.Parameter(torch.zeros((tokens, channels[-1])), requires_grad=True)
        self.lin1 = nn.Linear(tokens,18)

    def forward(self, x, start):
        x = self.encoder(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        x = x.permute(2,0,1)
        cls_token = self.cls_token.unsqueeze(1).repeat(1,x.shape[1],1)
        x = torch.cat([F.pad(start,(0,tokens-18),'constant',0).unsqueeze(0), x])
        x += self.pos_enc(x)
        x = self.transformer(x)[0]
        x = self.relu(x)
        x = self.lin1(x)
        return x


# Tuneable Parameters

window_size = 50
overlap = 0.5
kernel_sizes = [1,1,1]
channels = [12,16,32,64]
num_epochs = 1e4
weight_decay = 0.01
holdout = 0.2
batch_size = 128
tokens = 64
learning_rate = 1e-4

encoder = IMU_Encoder(window_size,kernel_sizes,channels)
transformer = IMU_Transformer(tokens).to(device)
optim = torch.optim.Adam(transformer.parameters(),weight_decay=weight_decay, lr=learning_rate)
loss_criterion = torch.nn.MSELoss()

vIMU_loader = VirtualIMU_Loader()
dataset, mocap_dataset = vIMU_loader.extract_dataset(window_size,overlap)
    
data = torch.tensor(dataset,device=device,dtype=torch.float32)
mocap_data = torch.tensor(mocap_dataset,device=device,dtype=torch.float32)
X_train = data[int(holdout*data.shape[0]):]
X_test = data[:int(holdout*data.shape[0])]
y_train = mocap_data[int(holdout*data.shape[0]):]
y_test = mocap_data[:int(holdout*data.shape[0])]

torch.cuda.empty_cache()
epochs = tqdm(range(int(num_epochs)))
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
        y_pred = transformer(X_test,y_test[:,:,0])
        loss = loss_criterion(y_test[:,:,1],y_pred)
        epochs.set_description(f'LOSS: {int(loss.cpu().item()*10000)/10000}, ')

torch.save(encoder.state_dict(),'model/encoder_weights')
encoder.load_state_dict(torch.load('model/encoder_weights'))
x = encoder(data[0])
pass