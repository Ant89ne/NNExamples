from os import path
import numpy as np
import torch 
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset
import matplotlib.pyplot as plt
import time
import torch.optim as optim
import random
import csv


class myNet(nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        pass


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
        print(self.data)

    def __len__(self):          #Function TO BE filled
        #Return the length of the dataset
        s = self.data.shape
        print(s)
        return s[0] * (s[1]-self.nbinput)
        
    def __getitem__(self, index): #Function TO BE filled
        s = self.data.shape
        i = index//(s[1]-self.nbinput)
        j = index - i*(s[1]-self.nbinput)
        print(j,i)
        d = self.data[i][j:j+self.nbinput]
        l = self.data[i][j+self.nbinput]

        return d,l

        
#Initialization
mydata = myDataset("inj1.csv", 10)        #Dataset

model = myNet()             #Neural Net

mydataloader = DataLoader(mydata, 8, shuffle = True)    #Dataloader

epoch = 10      #Nb epoch to run

myLoss = nn.MSELoss()   #Loss function 

optimizer = optim.Adam(model.parameters(), lr = 0.01)    #Optimization method

totLoss = []        #List of losses for final visualization

for e in range(epoch):
    for data, label in mydataloader:
        
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





