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

def getDailyData(csvreader):
    dataCases = []
    #Extract the header
    header = next(csvreader)
    #Extract sepcific index of interest
    for i, text in enumerate(header) : 
        if text == "New_cases":         #Number of cases
            indexCases = i
        if text == "Country":           #Country
            indexCountry = i

    line = next(csvreader)
    countryData = [int(line[indexCases])]
    country = line[indexCountry]

    for line in csvreader : 
        c = line[indexCountry]
        if country == c :
            countryData.append(int(line[indexCases]))
        else :
            if np.sum(np.array(countryData)) != 0 :
                dataCases.append(countryData)
            countryData = [int(line[indexCases])]
            country = c

    dataCases.append(countryData)
    return dataCases

def getMonthlyData(csvreader):
    dataCases = []
    #Extract the header
    header = next(csvreader)
    #Extract sepcific index of interest
    for i, text in enumerate(header) : 
        if text == "New_cases":         #Number of cases
            indexCases = i
        if text == "Country":           #Country
            indexCountry = i
        if text == "Date_reported":
            indexDate = i

    line = next(csvreader)
    countryData = [int(line[indexCases])]
    country = line[indexCountry]
    month = line[indexDate].split('-')[1]

    for line in csvreader : 
        c = line[indexCountry]
        m = line[indexDate].split('-')[1]
        if country == c and month != m:
            countryData.append(int(line[indexCases]))
            month = m
        elif country == c and month == m:
            countryData[-1] += int(line[indexCases])
        elif country != c :
            if np.sum(np.array(countryData)) != 0 :
                dataCases.append(countryData)
            countryData = [int(line[indexCases])]
            country = c
            month = m

    dataCases.append(countryData)

    return dataCases



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
    def __init__(self, myFile, nbinput, temp = "daily"):         #Function TO BE filled
        #Initialization
        self.file = myFile
        self.nbinput = nbinput
        self.temp = temp
        self.dataCases = []

        with open(self.file) as datafile:
            csvreader = csv.reader(datafile, delimiter = ',')
            
            if self.temp == "daily" :
                self.dataCases = getDailyData(csvreader)
            elif self.temp == "monthly":
                self.dataCases = getMonthlyData(csvreader)

        self.dataCases = np.array(self.dataCases).astype("int64")
        self.mean = np.mean(self.dataCases, axis = 1)
        self.std = np.std(self.dataCases, axis = 1)

        s = self.dataCases.shape
        for i in range(s[0]):
            self.dataCases[i,:] = (self.dataCases[i,:] - self.mean[i])/self.std[i]

    def __len__(self):          #Function TO BE filled
        #Return the length of the dataset
        s = self.dataCases.shape
        return s[0] * (s[1]-self.nbinput)
        
    def __getitem__(self, index): #Function TO BE filled
        s = self.dataCases.shape
        i = index//(s[1]-self.nbinput)
        j = index - i*(s[1]-self.nbinput)
        d = self.dataCases[i][j:j+self.nbinput]
        l = self.dataCases[i][j+self.nbinput]

        return torch.Tensor(d),torch.Tensor(np.array([l]))

        
#Initialization
nbInput = 15
nbLayers = 5
temporality = "daily"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

mydata = myDataset("world_covid_data.csv", nbInput, temporality)        #Dataset

print(mydata[len(mydata)-1])

model = myNet(nbInput, nbLayers)             #Neural Net
model.to(device)
x,l = mydata[0]

mydataloader = DataLoader(mydata, 64, shuffle = True)    #Dataloader

epoch = 100      #Nb epoch to run

myLoss = nn.MSELoss()   #Loss function 

print(model)

optimizer = optim.Adam(model.parameters(), lr = 0.0001)    #Optimization method

totLoss = []        #List of losses for final visualization
finLoss = []
validLoss = []

for e in range(epoch):
    print("Epoch ", e+1, "/", epoch)
    model.train()
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

    model.eval()
    # validLoss.append(validLoop(validDataloader, model))



plt.figure(0)
plt.plot([i for i in range(epoch)], finLoss)
plt.savefig("loss.png")
plt.show()



