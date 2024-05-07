import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
from tqdm import tqdm

sys.path.append('./')
from utils.virtualIMU_loader import VirtualIMU_Loader

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

print(f'Using Device: {device}')

class IMU_Decoder(nn.Module):
    def __init__(self, window, kernels, channels) -> None:
        super().__init__()

        layers = []
        for i in reversed(range(len(kernels))):
            layers.append(nn.ConvTranspose1d(channels[i+1],channels[i],kernels[i]))
            layers.append(nn.ReLU())
        layers.pop()
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x

class IMU_Encoder(nn.Module):
    def __init__(self, window, kernels, channels) -> None:
        super().__init__()
        layers = []
        for i in range(len(kernels)):
            layers.append(nn.Conv1d(channels[i],channels[i+1],kernels[i]))
            layers.append(nn.ReLU())
        layers.pop()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x

if __name__ == '__main__':
    # Tuneable Parameters

    window_size = 100
    overlap = 0.5
    kernel_sizes = [1,1,1]
    channels = [12,64,64,64]
    num_epochs = 1e4
    weight_decay = 0.01
    holdout = 0.2

    encoder = IMU_Encoder(window_size,kernel_sizes,channels)
    decoder = IMU_Decoder(window_size,kernel_sizes,channels)
    autoencoder = nn.Sequential(encoder,decoder).to(device)
    optim = torch.optim.Adam(autoencoder.parameters(),weight_decay=weight_decay)
    loss_criterion = torch.nn.MSELoss()

    vIMU_loader = VirtualIMU_Loader()
    dataset, _ = vIMU_loader.extract_dataset(window_size,overlap)
        
    data = torch.tensor(dataset,device=device,dtype=torch.float32)
    train_data = data[int(holdout*data.shape[0]):]
    test_data = data[:int(holdout*data.shape[0])]

    epochs = tqdm(range(int(num_epochs)))
    for i in epochs:
        optim.zero_grad()
        d_out = autoencoder(train_data)
        loss = loss_criterion(train_data,d_out)
        loss.backward()
        optim.step()

        with torch.no_grad():
            d_test = autoencoder(test_data)
            loss = loss_criterion(test_data,d_test)
            epochs.set_description(f'LOSS: {int(loss.cpu().item()*100)/100}')

    torch.save(encoder.state_dict(),'model/encoder_weights')
    encoder.load_state_dict(torch.load('model/encoder_weights'))
    x = encoder(data[0])
    pass
