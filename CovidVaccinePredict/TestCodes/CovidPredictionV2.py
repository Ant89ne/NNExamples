from os import path
from turtle import color
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
    countries = []
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
                countries.append(country)
            countryData = [int(line[indexCases])]
            country = c

    dataCases.append(countryData)
    countries.append(country)
    return dataCases, countries

def getWeeklyData(csvreader):
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
    countries = []
    week = 0

    for line in csvreader : 
        c = line[indexCountry]
        if country == c and week >= 7:
            countryData.append(int(line[indexCases]))
            week = 0
        elif country == c and week < 7:
            countryData[-1] += int(line[indexCases])
            week += 1
        elif country != c :
            if np.sum(np.array(countryData)) != 0 :
                dataCases.append(countryData)
                countries.append(country)
            countryData = [int(line[indexCases])]
            country = c
            week = 0

    dataCases.append(countryData)
    countries.append(country)

    return dataCases, countries



def getMonthlyData(csvreader):
    dataCases = []
    countries = []
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
                countries.append(country)
            countryData = [int(line[indexCases])]
            country = c
            month = m

    dataCases.append(countryData)
    countries.append(country)

    return dataCases, countries


def validLoop(dataloader, model, myLoss, device = "cpu"):
    model.eval()
    totLoss = 0
    b = 0
    for data, label in dataloader:
        data = data.to(device)
        label = label.to(device)

        #Prediction of the labels from the input data by the net (y^)
        pred = model(data)

        #Calculation of the loss
        loss = myLoss(pred, label)

        #Save the loss for later visualization
        totLoss += loss.item()
        b += data.shape[0]
    return totLoss/b

def testVis(GT, wPred, fPred, wLoss, fLoss, country, cstd, cmean, skip, nbInput):

    plt.figure(1, figsize=(20,10))
    plt.clf()
    plt.subplot(311)
    plt.plot([i for i in range(len(GT)-skip)], GT[skip:], 'r', [i for i in range(len(wPred))], wPred, 'b', [i for i in range(len(fPred))], fPred, 'g')
    plt.axvline(nbInput, color='k')
    plt.title("Normalized predictions for " + country)
    plt.legend(["Ground Truth", "Prediction, GT input", "Prediction, Pred input"])
    plt.text(3,1/2*np.max(GT), "Input Data")
    
    
    plt.subplot(312)
    plt.plot([i for i in range(len(GT)-skip)], GT[skip:]*cstd + cmean, 'r', [i for i in range(len(wPred))], wPred*cstd + cmean, 'b', [i for i in range(len(fPred))], fPred*cstd + cmean, 'g')
    plt.axvline(nbInput,color='k')
    plt.title("True values predictions for " + country)
    plt.legend(["Ground Truth", "Prediction, GT input", "Prediction, Pred input"])
    plt.text(3,1/2*np.max(GT*cstd+cmean), "Input Data")

    plt.subplot(313)
    plt.plot([i for i in range(len(wLoss))], wLoss, 'b', [i for i in range(len(fLoss))], fLoss, 'g')
    plt.axvline(nbInput, color='k')
    plt.title("MSE Loss for " + country)
    plt.legend(["Prediction, GT input", "Prediction, Pred input"])
    plt.text(3,1/2*np.max(fLoss), "Initialization")

    plt.savefig("TestRes/test"+country+".png")


def testLoop(dataset, model, myLoss, nbInput, device = "cpu"):
    model.eval()
    
    s = dataset.dataCases.shape

    for c in range(s[0]):
        weeklyPred = []
        fullPred = []
        weeklyLoss = []
        fullLoss = []
        data = dataset.dataCases[c]
        mini = np.min(data)
        skip = 0

        for i, k in enumerate(data) :
            if k < mini+0.5 and len(weeklyPred) == 0:
                skip += 1
                continue
            elif len(weeklyPred) >= nbInput :
                ####Predictions TODO
                weeklyPred.append(model(torch.Tensor(data[i-nbInput:i]).reshape((1,nbInput)).to(device)).item())
                fullPred.append(model(torch.Tensor(fullPred[i-skip-nbInput:i-skip]).reshape((1, nbInput)).to(device)).item())
                
                
                weeklyLoss.append(myLoss(torch.Tensor([weeklyPred[-1]]), torch.Tensor([k])).item())
                fullLoss.append(myLoss(torch.Tensor([fullPred[-1]]), torch.Tensor([k])).item())
                
            else : 
                weeklyPred.append(k)
                fullPred.append(k)
                weeklyLoss.append(0)
                fullLoss.append(0)

        weeklyPred, fullPred = np.array(weeklyPred), np.array(fullPred)
        testVis(data, weeklyPred, fullPred, weeklyLoss, fullLoss, dataset.country[c], dataset.std[c], dataset.mean[c], skip, nbInput)

    
    # totLoss = 0
    # b = 0
    # for data, label in dataloader:
    #     data = data.to(device)
    #     label = label.to(device)

    #     #Prediction of the labels from the input data by the net (y^)
    #     pred = model(data)

    #     #Calculation of the loss
    #     loss = myLoss(pred, label)

    #     #Save the loss for later visualization
    #     totLoss += loss.item()
    #     b += data.shape[0]
    # print('E', totLoss, b)
    # return totLoss/b


class myNet(nn.Module):
    def __init__(self, nbInput, nbLayer):
        super(myNet, self).__init__()
        
        hid = 3

        self.nbInput = nbInput

        self.layers = []
        self.layers.append(nn.Linear(nbInput, nbInput*hid))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(nbInput*hid))

        for l in range(nbLayer) :
            self.layers.append(nn.Linear(nbInput*hid, nbInput*hid))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.BatchNorm1d(nbInput*hid))

        self.layers.append(nn.Linear(nbInput*hid, nbInput))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(nbInput))

        self.layers.append(nn.Linear(nbInput, 1))

        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        
        ret = self.layers(x)

        return ret

class myCNNNet(nn.Module):
    def __init__(self, nbInput, nbLayer):
        super(myCNNNet, self).__init__()
        
        hid = 3

        self.nbInput = nbInput

        self.layers = []
        self.layers.append(nn.Conv1d(1,5,7, padding=3))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(5))

        self.layers.append(nn.Conv1d(5,10,7,padding = 3))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(10))

        self.layers.append(nn.Conv1d(10,5,7, padding=3))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.BatchNorm1d(5))


        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(5*(nbInput), 1))#2*(nbInput-2)))
        # self.layers.append(nn.ReLU())
        # self.layers.append(nn.BatchNorm1d(2*(nbInput-2)))

        # self.layers.append(nn.Linear(2*(nbInput-2), 1))


        # for l in range(nbLayer) :
        #     self.layers.append(nn.Linear(nbInput*hid, nbInput*hid))
        #     self.layers.append(nn.ReLU())
        #     self.layers.append(nn.BatchNorm1d(nbInput*hid))

        # self.layers.append(nn.Linear(nbInput*hid, nbInput))
        # self.layers.append(nn.ReLU())
        # self.layers.append(nn.BatchNorm1d(nbInput))

        # self.layers.append(nn.Linear(nbInput, 1))

        self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        s = x.shape
        x = torch.reshape(x, (s[0],1,s[1]))
        ret = self.layers(x)

        return ret

class myDataset(Dataset):
    def __init__(self, myFile, nbinput, temp = "daily", percentages = [0.8,0.1,0.1], typeIndex = 0):         #Function TO BE filled
        #Initialization
        self.file = myFile
        self.nbinput = nbinput
        self.temp = temp
        self.dataCases = []
        self.country = []

        with open(self.file) as datafile:
            csvreader = csv.reader(datafile, delimiter = ',')
            
            if self.temp == "daily" :
                self.dataCases, self.country = getDailyData(csvreader)
            elif self.temp == "monthly":
                self.dataCases, self.country = getMonthlyData(csvreader)
            elif self.temp == "weekly":
                self.dataCases, self.country  = getWeeklyData(csvreader)

        self.dataCases = np.array(self.dataCases).astype("double")
        self.mean = np.mean(self.dataCases, axis = 1)
        self.std = np.std(self.dataCases, axis = 1)

        s = self.dataCases.shape
        for i in range(s[0]):
            self.dataCases[i,:] = (self.dataCases[i,:] - self.mean[i])/self.std[i]

        indexMin = int(np.sum(percentages[:typeIndex])*s[0])
        indexMax = int(np.sum(percentages[:typeIndex+1])*s[0])

        self.dataCases = self.dataCases[indexMin:indexMax,:]
        self.country = self.country[indexMin:indexMax]

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

    def show(self):
        s = self.dataCases.shape
        for k in range(s[0]):
            plt.figure(0)
            plt.clf()
            plt.plot([i for i in range(s[1])], self.dataCases[k,:])
            plt.title(self.country[k])
            plt.show()
            time.sleep(0.01)
            plt.close()

        
#Initialization
nbInput = 15
nbLayers = 3
temporality = "weekly"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
percentages = [0.8,0.1,0.1]


mydataTrain = myDataset("world_covid_data.csv", nbInput, temporality, percentages, 0)        #Dataset
mydataEval = myDataset("world_covid_data.csv", nbInput, temporality, percentages, 1)        #Dataset
mydataTest = myDataset("world_covid_data.csv", nbInput, temporality, percentages, 2)        #Dataset
print(len(mydataTrain), len(mydataTest), len(mydataEval))


# model = myNet(nbInput, nbLayers)             #Neural Net
model = myCNNNet(nbInput, nbLayers)             #Neural Net

model.to(device)

mydataloaderTrain = DataLoader(mydataTrain, 64, shuffle = True)    #Dataloader
mydataloaderEval = DataLoader(mydataEval, 64, shuffle = False)    #Dataloader
mydataloadertest = DataLoader(mydataTest, 64, shuffle = False)    #Dataloader


epoch = 2000      #Nb epoch to run

# myLoss = nn.MSELoss(reduction='sum')   #Loss function 
myLoss = nn.L1Loss(reduction='sum')   #Loss function 

optimizer = optim.Adam(model.parameters(), lr = 0.00005)    #Optimization method

totLoss = 0        #List of losses for final visualization
b = 0
finLoss = []
validLoss = []

for e in range(epoch):
    print("Epoch ", e+1, "/", epoch)
    model.train()
    for data, label in mydataloaderTrain:
        data = data.to(device)
        label = label.to(device)

        #Initialization of the gradient error
        optimizer.zero_grad()
        
        #Prediction of the labels from the input data by the net (y^)
        pred = model(data)

        #Calculation of the loss
        loss = myLoss(pred, label)

        #Save the loss for later visualization
        totLoss += loss.item()
        b+=data.shape[0]
        #Backpropagation of the error through the net
        loss.backward()

        #Update of the weights and biaises of the net
        optimizer.step()

    if epoch == 500 :
        optimizer = optim.Adam(model.parameters(), lr = 0.00001)    #Optimization method
    if epoch == 1000 :
        optimizer = optim.Adam(model.parameters(), lr = 0.000005)    #Optimization method


    finLoss.append(totLoss/b)
    print("Training Loss ", finLoss[-1])
    totLoss = 0
    b = 0

    model.eval()
    validLoss.append(validLoop(mydataloaderEval, model, myLoss, device))
    print("Evaluation Loss ", validLoss[-1])

testLoop(mydataTest, model, myLoss, nbInput, device)

plt.figure(0)
plt.subplot(211)
plt.plot([i for i in range(epoch)], finLoss, 'r')
plt.plot([i for i in range(epoch)], validLoss, 'b')
plt.legend(["Training Loss", "Evaluation Loss"])
plt.subplot(212)
plt.plot([i for i in range(100,epoch)], finLoss[100:], 'r')
plt.plot([i for i in range(100,epoch)], validLoss[100:], 'b')
plt.legend(["Training Loss", "Evaluation Loss"])
plt.savefig("loss.png")
plt.show()

torch.save(model.state_dict(), "myModel.pt")



