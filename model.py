from torch.nn import LSTM
from torch.utils.data import Dataset
from torch.nn import Linear
import torch
import torch.nn as nn
import numpy as np

class LSTM_Dataset(Dataset):
  def __init__(self,inputs,labels,time_series_len):
    self.inputs = inputs
    self.labels = labels
    self.time_series_len = time_series_len
    shifted_input, shifted_target = self.shift_data()
    self.inputs = torch.tensor(np.array(shifted_input))
    self.labels = torch.tensor(np.array(shifted_target))
    print(self.inputs.shape)
    print(self.labels.shape)

  def shift_data(self):
    shifted_target = []
    shifted_input = []
    inputs = self.inputs
    labels = self.labels
    for start in range(0, len(inputs) - self.time_series_len, 1):
        shifted_input.append(inputs[start:start+self.time_series_len])
        shifted_target.append(labels[start+self.time_series_len])
    return shifted_input,shifted_target

  def __getitem__(self,idx):
      return self.inputs[idx], self.labels[idx]

  def __len__(self):
      return len(self.inputs)


class BasicDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]
    
    def getData(self):
        return self.data
    
    def getTarget(self):
        return self.target

class GeneralDataset(Dataset):
    def __init__(self, data, target,train_ratio, test_ratio):
        self.data = data
        self.target = target
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

    def getTrainDataset(self):
        train_len = int(len(self.data)*self.train_ratio)
        trainData = self.data[:train_len]
        trainTarget = self.target[:train_len]
        return BasicDataset(trainData, trainTarget)
    
    def getTestDataset(self):
        train_len = int(len(self.data)*self.train_ratio)
        test_len = int(len(self.data)*self.test_ratio)

        testData = self.data[train_len:train_len+test_len]
        testTaget = self.target[train_len:train_len+test_len]
        return BasicDataset(testData, testTaget)

    def getValidationDataset(self):
        train_len = int(len(self.data)*self.train_ratio)
        test_len = int(len(self.data)*self.test_ratio)

        valData = self.data[train_len+test_len:]
        valTaget = self.target[train_len+test_len:]

        return BasicDataset(valData, valTaget)

class TradingModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = 3
        self.input_size = input_size
        self.lstm = LSTM(self.input_size,self.hidden_size,self.num_layers,dropout=dropout, batch_first=True)
        self.linear = Linear(self.hidden_size,3)
        
    
    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        output, _ = self.lstm(x,(h0,c0))
        #print(output)
        output = self.linear(output[:, -1, :]).squeeze(-1)
        return output
    
def create_batches(data:BasicDataset, batch_size, shift):
    targetBatches = []
    inputBatches = []
    inputs = data.getData()
    target = data.getTarget()
    for start in range(0, len(data) - batch_size + 1, shift):
        inputBatch = inputs[start:start+batch_size]
        targetBatch=target[start:start+batch_size]
        targetBatches.append(targetBatch)
        inputBatches.append(inputBatch)
    return inputBatches,targetBatches