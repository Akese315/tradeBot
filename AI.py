import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import LSTM
from torch.nn import Linear

class tradingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = LSTM(9,20,2,dropout=0.2, batch_first=True)
        self.linear = Linear(9,1)
    
    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        output = self.lstm(x,(h0,c0))
        output = output[:, -1, :]
        output = self.linear(output)
        return output
    

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = tradingModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)