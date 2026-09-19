import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import LSTM
import matplotlib.pyplot as plt
from torch.nn import Linear
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset, SequentialSampler


    
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

def testModel(testingDataset, validatingDataset,name):
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = tradingModel()
    loss_fn = nn.MSELoss()
    model.load_state_dict(torch.load("./model/version_1.pth"))
    model.eval()

    test_loss =0.0
    val_loss = 0.0

    output_plot = []
    target_plot = []

    for i, data in enumerate(testingDataset):
        inputs, labels = data
        inputs = inputs.to(torch.float32)
        labels = labels.to(torch.float32)
        inputs = inputs.unsqueeze(1).to(device)
        labels = labels.to(device)
        output = model(inputs)
        output_plot.append(output[-1][0].detach().numpy())
        target_plot.append(labels[-1][0].detach().numpy())
        loss = loss_fn(output, labels)
        test_loss += loss.item()

    for i, data in enumerate(validatingDataset):
        inputs, labels = data
        inputs = inputs.to(torch.float32)
        labels = labels.to(torch.float32)
        inputs = inputs.unsqueeze(1).to(device)
        labels = labels.to(device)
        output = model(inputs)
        
        loss = loss_fn(output, labels)
        val_loss += loss.item()
    


    print("loss for test : ", (test_loss/len(testingDataset)))
    print("loss for val : ", (val_loss/len(testingDataset)))

    time = np.arange(0, len(output_plot))

    plt.plot(time,output_plot,'-', color='blue')
    plt.plot(time,target_plot,'-', color='red')
    plt.show()


def trainModel(dataset:GeneralDataset):
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    

    train_dataset = dataset.getTrainDataset()
    test_dataset = dataset.getTestDataset()
    val_dataset = dataset.getValidationDataset()

    train_loader = create_batches(train_dataset,32,1)
    test_loader = create_batches(test_dataset,32,1)
    val_loader = create_batches(val_dataset,32,1)

    '''train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, sampler=SequentialSampler(train_dataset))
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, sampler=SequentialSampler(val_dataset))
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, sampler=SequentialSampler(test_dataset))

'''

    num_epochs = 15
    model.train(True)
    for epoch in range(num_epochs):
        running_loss = 0.
        for i in range(len(train_loader[0])):
            
            inputs = train_loader[0][i]
            labels = train_loader[1][i]

            inputs = inputs.to(torch.float32)
            labels = labels.to(torch.float32)

            inputs = inputs.unsqueeze(1).to(device)
            labels = labels.to(device)
            output = model(inputs)
            if epoch == 14 and (i == 3 or i ==4):
                print(inputs)
                print(output)
                print(labels)
            
            loss = loss_fn(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
        print(f"Epoch {epoch + 1}, iter {i + 1}: {running_loss / 100}")
        scheduler.step(running_loss)
        print(f"epoch {epoch + 1} done")
    answer = input("do you want to save the model ? (Yes/No)" )
    if answer == "Yes":
        torch.save(model.state_dict(), "./model/version_1.pth")
    else:
        print("Model not saved, fin du programme")
    testModel(test_loader, val_loader, "./model/version_1.pth")
