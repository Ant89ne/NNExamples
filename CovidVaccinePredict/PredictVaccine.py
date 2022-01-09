from os import path
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset
import matplotlib.pyplot as plt
import time
import torch.optim as optim
import random
import csv


class myNet(nn.Module):
    def __init__(self, nbInput, nbLayer):
        super(myNet, self).__init__()
        
        hid = 5

        self.nbInput = nbInput

        self.layers = []
        self.layers.append(nn.Linear(nbInput, nbInput*hid))
        self.layers.append(nn.BatchNorm1d(nbInput*hid))
        self.layers.append(nn.ReLU())

        for l in range(nbLayer) :
            self.layers.append(nn.Linear(nbInput*hid, nbInput*hid))
            self.layers.append(nn.BatchNorm1d(nbInput*hid))
            self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(nbInput*hid, nbInput))
        self.layers.append(nn.BatchNorm1d(nbInput))
        self.layers.append(nn.ReLU())

        self.layers.append(nn.Linear(nbInput, 1))

        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        
        ret = self.layers(x)

        return ret


class myDataset(Dataset):
    def __init__(self, myFile, nbinput):         #Function TO BE filled
        #Initialization
        self.file = myFile
        self.nbinput = nbinput
        self.data = []
        with open(self.file) as datafile:
            csvreader = csv.reader(datafile, delimiter = ',')
            for line in csvreader : 
                if len(line) != 0 :
                    self.data.append(line)
        
        self.data = np.array(self.data).astype("int16")
        self.mean = np.mean(self.data)
        self.std = np.std(self.data)
        self.data = (self.data - self.mean)/self.std

    def __len__(self):          #Function TO BE filled
        #Return the length of the dataset
        s = self.data.shape
        return s[0] * (s[1]-self.nbinput)
        
    def __getitem__(self, index): #Function TO BE filled
        s = self.data.shape
        i = index//(s[1]-self.nbinput)
        j = index - i*(s[1]-self.nbinput)
        d = self.data[i][j:j+self.nbinput]
        l = self.data[i][j+self.nbinput]

        return torch.Tensor(d),torch.Tensor(np.array([l]))

        
#Initialization
nbInput = 15
nbLayers = 5
device = 'cuda' if torch.cuda.is_available() else 'cpu'

mydata = myDataset("inj1.csv", nbInput)        #Dataset

model = myNet(nbInput, nbLayers)             #Neural Net
model.to(device)
x,l = mydata[0]

mydataloader = DataLoader(mydata, 64, shuffle = True)    #Dataloader

epoch = 1000      #Nb epoch to run

myLoss = nn.MSELoss()   #Loss function 

print(model)

optimizer = optim.Adam(model.parameters(), lr = 0.0001)    #Optimization method

totLoss = []        #List of losses for final visualization
finLoss = []

for e in range(epoch):
    print("Epoch ", e+1, "/", epoch)
    
    for data, label in mydataloader:
        data = data.to(device)
        label = label.to(device)

        #Initialization of the gradient error
        optimizer.zero_grad()

        #Prediction of the labels from the input data by the net (y^)
        pred = model(data)

        #Calculation of the loss
        loss = myLoss(pred, label)

        #Save the loss for later visualization
        totLoss.append(loss.item())

        #Backpropagation of the error through the net
        loss.backward()

        #Update of the weights and biaises of the net
        optimizer.step()

    finLoss.append(np.mean(totLoss))
    print("Training Loss ", finLoss[-1])
    totLoss = []


plt.figure(0)
plt.plot([i for i in range(epoch)], finLoss)
plt.savefig("loss.png")
plt.show()



