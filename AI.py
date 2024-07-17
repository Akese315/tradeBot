import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import LSTM
from torch.nn import Linear
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset

class TrainingDataset(Dataset):
    def __init__(self, data, target,batch_size):
        self.data = data
        self.target = target
        self.batch_size = batch_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.target[idx]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    

class tradingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_size = 256
        self.num_layers = 2
        self.input_size = 9
        self.lstm = LSTM(self.input_size,self.hidden_size,self.num_layers,dropout=0.2, batch_first=True)
        self.linear = Linear(self.hidden_size,3)
        
    
    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        output, _ = self.lstm(x,(h0,c0))
        #print(output)
        output = self.linear(output[:, -1, :]).squeeze(-1)
        return output
    
def create_batches(data, batch_size, shift):
    batches = []
    total_size = len(data)
    indices = torch.arange(total_size)
    for start in range(0, len(data) - batch_size + 1, shift):
        batch = Subset(data, indices[start:start+batch_size])
        batches.append(batch)
    return batches


def trainModel(dataset:TrainingDataset):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
    print(f"Using {device} device")

    model = tradingModel().to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr=0.001)

    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    # Calculer les tailles des ensembles
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)

    indices = torch.arange(total_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    

    num_epochs = 1500
    model.train(True)
    for epoch in range(num_epochs):
        
        running_loss = 0.
        for i, batch in enumerate(train_loader):
            print(batch[0].shape)
            input("pause")
            inputs, target = batch
            inputs = inputs.unsqueeze(1).to(device)
            #print(inputs.shape)
            target = target.to(device)
            
            output = model(inputs)
            if epoch == 100:
                input("pause")
                print(output) 
                print(target)
            loss = loss_fn(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        print(f"Epoch {epoch + 1}, iter {i + 1}: {running_loss / 100}")
        running_loss = 0.

        print(f"epoch {epoch + 1} done")
    answer = input("do you want to save the model ? (Yes/No)" )
    if answer == "Yes":
        torch.save(model.state_dict(), "./model/version_1.pth")
    else:
        print("Model not saved, fin du programme")



donnee = torch.arange(2000)
donnee = create_batches(donnee, 24, 1)
data = donnee[0]
print(data)
data = donnee[1]
print(data)

