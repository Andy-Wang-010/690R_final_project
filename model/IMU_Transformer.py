import torch
import torch.nn as nn
import sys
sys.path.append('./')
from model.IMU_Encoder import IMU_Encoder
from utils.virtualIMU_loader import VirtualIMU_Loader
from tqdm import tqdm

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

print(f'Using Device: {device}')

class IMU_Transformer(nn.Module):
    def __init__(self, encoder) -> None:
        super().__init__()
        self.encoder = IMU_Encoder(window_size,kernel_sizes,channels)
        self.encoder.load_state_dict(torch.load('model/encoder_weights'))
        self.transformer = nn.Transformer(d_model=channels[-1])
        self.lin1 = nn.Linear(channels[-1],18)

    def forward(self, x):
        x = self.encoder(x)
        x = self.transformer(x)
        x = self.lin1(x)


# Tuneable Parameters

window_size = 100
overlap = 0.5
kernel_sizes = [3,3]
channels = [12,64,32]
num_epochs = 1e4
weight_decay = 0.01
holdout = 0.2

encoder = IMU_Encoder(window_size,kernel_sizes,channels)
transformer = IMU_Transformer(encoder).to(device)
optim = torch.optim.Adam(transformer.parameters(),weight_decay=weight_decay)
loss_criterion = torch.nn.MSELoss()

vIMU_loader = VirtualIMU_Loader()
dataset, mocap_data = vIMU_loader.extract_dataset(window_size,overlap)
    
data = torch.tensor(dataset,device=device,dtype=torch.float32)
X_train = data[int(holdout*data.shape[0]):]
X_test = data[:int(holdout*data.shape[0])]
y_train = mocap_data[int(holdout*data.shape[0]):]
y_test = mocap_data[:int(holdout*data.shape[0])]

epochs = tqdm(range(int(num_epochs)))
for i in epochs:
    optim.zero_grad()
    y_pred = transformer(X_train)
    loss = loss_criterion(y_train,y_pred)
    loss.backward()
    optim.step()

    with torch.no_grad():
        y_pred = transformer(X_test)
        loss = loss_criterion(y_test,y_pred)
        epochs.set_description(f'LOSS: {int(loss.cpu().item()*100)/100}')

torch.save(encoder.state_dict(),'model/encoder_weights')
encoder.load_state_dict(torch.load('model/encoder_weights'))
x = encoder(data[0])
pass