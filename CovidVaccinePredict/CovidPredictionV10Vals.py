from audioop import reverse
from os import path
from turtle import color, forward
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset
import matplotlib.pyplot as plt
import time
import torch.optim as optim
import random
import csv

def getMetrics(dataloader, model, device = "cuda"):
    model.eval()
    with torch.no_grad():
        listae = [[] for i in range(5)]
        listse = [[] for i in range(5)]
        for d, l, w in dataloader :
            d = d.to(device)
            l = l.to(device)
            pred = model(d)

            ae = torch.absolute(pred-l).cpu()
            se = torch.square(pred-l).cpu()

            for k in range(d.shape[0]):
                for i in range(5):
                    listae[i].append(ae[k,0,i].item())
                    listse[i].append(se[k,0,i].item())

        listae = np.array(listae)
        listse = np.array(listse)

        meanValae = np.mean(listae, axis = 1)
        stdValae = np.std(listae, axis = 1)
        meanValse = np.mean(listse, axis = 1)
        stdValse = np.std(listse, axis = 1)

        return meanValae, stdValae, np.sqrt(meanValse), np.sqrt(stdValse)

def writeTable(fullDataset, model, metrics, percentages, device = "cuda"):
    model.eval()
    with torch.no_grad():
        f = open("2022predsRNNCumCases.csv", "w+")
        csvwriter = csv.writer(f, delimiter = ',')
        nbOutput = fullDataset.nboutput
        metricsHeader = []
        for i in range(nbOutput):
            metricsHeader = metricsHeader + ["MAE_week_" + str(i+1), "stdAE_week_"+str(i+1), "RMSE_week_"+str(i+1), "stdRMSE_week_"+str(i+1)]
        csvwriter.writerow(["Country"] + ["Pred_week_" + str(i+1) for i in range(nbOutput)] +  metricsHeader)
        s = fullDataset.dataCases.shape
        for c in range(s[0]):

            flag = (c/s[0] > percentages[0]) + (c/s[0] > (percentages[0] + percentages[1]))

            data = fullDataset.dataCases[c,-fullDataset.nbinput:]
            pred = model(torch.reshape(torch.Tensor(np.array(data)).to(device), (1, fullDataset.nbinput)))


            mu = fullDataset.mean[c]
            sig = fullDataset.std[c]
            met = np.array(metrics[flag]) * sig + mu 
            pred = pred.cpu().numpy()[0,0,:] * sig + mu

            metline = []
            for i in range(nbOutput):
                metline = metline + [str(int(m)) for m in met[:,i]]

            line = [fullDataset.country[c]] + [str(int(p)) for p in pred] + metline

            csvwriter.writerow(line)
        

            


def getDailyData(csvreader):
    dataCases = []
    countries = []
    #Extract the header
    header = next(csvreader)
    #Extract sepcific index of interest
    for i, text in enumerate(header) : 
        if text == "Cumulative_casees":         #Number of cases
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
        if text == "Cumulative_cases":         #Number of cases
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
            # countryData[-1] += int(line[indexCases])
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
        if text == "Cumulative_cases":         #Number of cases
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
    for data, label, w in dataloader:
        data = data.to(device)
        label = label.to(device)
        w = w.to(device)

        #Prediction of the labels from the input data by the net (y^)
        pred = model(data)
        pred = torch.reshape(pred, (data.shape[0], 1, nbOutput))
        #Calculation of the loss
        loss = torch.sum(myLoss(pred, label))#*w)

        #Save the loss for later visualization
        totLoss += loss.item()
        b += data.shape[0]
    return totLoss/b


def SampleVisMultiple(x, pred, l, loss):
    with torch.no_grad() :
        x = torch.reshape(x, (1,1,x.shape[1])).cpu()
        pred = pred.cpu()
        l = l.cpu()
        plt.figure(0)
        plt.clf()
        plt.plot([i for i in range(x.shape[2] + pred.shape[2])],np.concatenate((x, pred), axis = 2)[0,0,:], 'b')
        plt.plot([i for i in range(x.shape[2] + l.shape[2])], np.concatenate((x,l), axis = 2)[0,0,:], 'r')
        plt.legend(["Prediction", "Ground Truth"])
        plt.axis([0, x.shape[2]+pred.shape[2], -1,5])
        plt.show()

def testMultipleLoop(dataloader, model, nbInput, device = "cpu"):
    model.eval()
    myLoss = nn.L1Loss(reduction = 'none')
    for x,l,w in dataloader:
        x = x.to(device)
        l = l.to(device)

        pred = model(x)

        loss = myLoss(pred, l)

        SampleVisMultiple(x, pred, l, loss)

def cloudVis(data, preds, meanVals, c, nbInput, cum):
    with torch.no_grad():
        plt.figure(0, figsize=(20,10))
        plt.clf()
        plt.plot([i for i in range(len(data))], data, '-r')
        # plt.plot([i for i in range(nbInput, nbInput+len(meanVals))], meanVals, '-b')
        for i,p in enumerate(preds) :
            p = p[0,0,:].cpu()
            plt.plot([k+i+nbInput for k in range(len(p))], p, '*', alpha = 0.3)
        
        plt.plot([i for i in range(len(data))], data, '-r')
        # plt.plot([i for i in range(nbInput, nbInput+len(meanVals))], meanVals, '-b')
        plt.title("Predictions for 5 weeks for " + c)
        plt.legend(["Ground Truth", "Mean Predictions", "Five weeks predictions"])
        plt.axvline(nbInput,color = 'k', linestyle = '--')
        plt.axvline(0,color='k')
        plt.axvline(len(data)/2, color = 'k')
        plt.axvline(len(data)-1, color = 'k')
        plt.text(2.5,1/2*np.max(data), "Input Data")
        plt.text(len(data)/4-2,np.min(data) , "2020")
        plt.text(3*len(data)/4-2,np.min(data) , "2021")
        plt.text(len(data)+ 2,np.min(data) , "2022")
        plt.xticks([i for i in range(0,len(data)+10,5)], [i for i in range(0,len(data)+10,5)])
        plt.xlabel("Week number starting in 01-2020")
        plt.ylabel("Number of Cases" + cum)
        plt.show()

def fullTestMultipleLoop(dataset, model, nbInput, nbOutput, device = "cpu"):
    model.eval()
    with torch.no_grad():            

        s = dataset.dataCases.shape
        for d in range(s[0]):
            data = dataset.dataCases[d]
            countryPreds = []
            meanPredict = np.zeros((1, len(data)-nbInput+nbOutput-1))
            nb = np.zeros((1, len(data)-nbInput+nbOutput-1))
            for k in range(len(data) - nbInput) :
                inp = data[k:k+nbInput]
                inp = torch.reshape(torch.Tensor(inp), (1, len(inp))).to(device)
                pred = model(inp)
                pred = torch.reshape(pred, (1,1,nbOutput))
                countryPreds.append(pred)   
                meanPredict[0,k:k+nbOutput] += pred.cpu().numpy()[0,0,:]
                nb[0,k:k+nbOutput] += 1

            data = data * dataset.std[d] + dataset.mean[d]
            
            meanPredict /= nb
            cum = "(cumulative)"
            cloudVis(data, np.array(countryPreds) * dataset.std[d] + dataset.mean[d], meanPredict[0,:] * dataset.std[d] + dataset.mean[d], dataset.country[d], nbInput, cum)
            cum = ""
            countryPreds = np.array(countryPreds) * dataset.std[d] + dataset.mean[d]
            print(countryPreds[0].shape)
            countryPreds = countryPreds.tolist()
            for k in range(len(data) - nbInput) :
                for i in range(len(countryPreds[k][0,0,:])-1,0,-1):
                    countryPreds[k][0,0,i] = countryPreds[k][0,0,i] - countryPreds[k][0,0,i-1]
                countryPreds[k][0,0,0] = countryPreds[k][0,0,0] - data[k+nbInput-1]
            print("AFTER", countryPreds)
            dd = [data[i].tolist() - data[i-1].tolist() for i in range(len(data)-1,0,-1)]
            data = np.array([data[0].tolist()] + [dd[i] for i in range(len(dd)-1,-1,-1)])

            cloudVis(data, np.array(countryPreds) , meanPredict[0,:] * dataset.std[d] + dataset.mean[d], dataset.country[d], nbInput, cum)

class myRNN(nn.Module):
    def __init__(self, nbInput, nbOutput):
        super(myRNN, self).__init__()
        
        self.nbInput = nbInput
        self.nbOutput = nbOutput
        self.hidNb = 100
        self.gru = nn.GRU(input_size = 1, hidden_size = self.hidNb, batch_first = True, dropout=0.5, num_layers=2)

        self.lin = nn.Linear(self.hidNb, 1)

        # self.layers = nn.Sequential(*self.layers)
        
    def forward(self, x):
        s = x.shape
        x = torch.reshape(x, (s[0],s[1], 1))
        vals = x    
        for k in range(self.nbOutput):
            ret, h = self.gru(vals)
            ret = self.lin(ret[:,-1,:])
            ret = torch.reshape(ret, (s[0],1,1))
            vals = torch.cat((vals[:,1:,:], ret), dim = 1)


        return vals[:,-nbOutput:,:]


class myDatasetIter(Dataset):
    def __init__(self, myFile, nbinput, nboutput, temp = "daily", percentages = [0.8,0.1,0.1], typeIndex = 0):         #Function TO BE filled
        #Initialization
        self.file = myFile
        self.nbinput = nbinput
        self.nboutput = nboutput
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

        self.minis = np.min(self.dataCases, axis = -1)
        self.maxis = np.max(self.dataCases, axis = -1)

    def __len__(self):          #Function TO BE filled
        #Return the length of the dataset
        s = self.dataCases.shape
        return s[0] * (s[1]-self.nbinput - self.nboutput)
        
    def __getitem__(self, index): #Function TO BE filled
        s = self.dataCases.shape
        i = index//(s[1]-self.nbinput-self.nboutput)
        j = index - i*(s[1]-self.nbinput-self.nboutput)
        d = self.dataCases[i][j:j+self.nbinput]
        l = self.dataCases[i][j+self.nbinput:j+self.nbinput+self.nboutput]

        w = np.sum(((d-self.minis[i])/self.maxis[i]) * (d >= (self.minis[i] + (self.maxis[i] - self.minis[i])*0.3) ), axis = -1)
        
        return torch.Tensor(d),torch.Tensor(np.array([l])), torch.Tensor(np.array([w]))

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
nbInput = 10
nbOutput = 5
nbLayers = 3
temporality = "weekly"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
percentages = [0.8,0.1,0.1]


mydataTrain = myDatasetIter("world_covid_data.csv", nbInput, nbOutput, temporality, percentages, 0)        #Dataset
mydataEval = myDatasetIter("world_covid_data.csv", nbInput, nbOutput, temporality, percentages, 1)        #Dataset
mydataTest = myDatasetIter("world_covid_data.csv", nbInput, nbOutput, temporality, percentages, 2)        #Dataset
brazilSet = myDatasetIter("brazilData.csv", nbInput, nbOutput, temporality, [1], 0)

fullDataSet = myDatasetIter("world_covid_data.csv", nbInput, nbOutput, temporality, [1], 0)
print(len(mydataTrain), len(mydataTest), len(mydataEval))


# model = myNet(nbInput, nbLayers)             #Neural Net
# model = iterCNNNet(nbInput, nbOutput)             #Neural Net
model = myRNN(nbInput, nbOutput)             #Neural Net

model.to(device)

mydataloaderTrain = DataLoader(mydataTrain, 64, shuffle = True)    #Dataloader
mydataloaderEval = DataLoader(mydataEval, 64, shuffle = False)    #Dataloader
mydataloadertest = DataLoader(mydataTest, 1, shuffle = False)    #Dataloader

model = torch.load("myModelRNNCumulIterCases.pt")
# model.load_state_dict(m)
allMetrics = []
allMetrics.append(getMetrics(mydataloaderTrain, model))
allMetrics.append(getMetrics(mydataloaderEval, model))
allMetrics.append(getMetrics(mydataloadertest, model))

writeTable(fullDataSet, model, allMetrics, percentages)

fullTestMultipleLoop(brazilSet, model, nbInput, nbOutput, device)
exit()
epoch = 1000      #Nb epoch to run

# myLoss = nn.MSELoss(reduction='sum')   #Loss function 
myLoss = nn.MSELoss(reduction='none')   #Loss function 

optimizer = optim.Adam(model.parameters(), lr = 0.00001)#, betas = (0.8,0.9))    #Optimization method

totLoss = 0        #List of losses for final visualization
b = 0
finLoss = []
validLoss = []

for e in range(epoch):
    print("Epoch ", e+1, "/", epoch)
    model.train()
    for data, label, w in mydataloaderTrain:
        data = data.to(device)
        label = label.to(device)
        w = w.to(device)

        #Initialization of the gradient error
        optimizer.zero_grad()
        
        #Prediction of the labels from the input data by the net (y^)
        pred = model(data)
        pred = torch.reshape(pred, (data.shape[0], 1, nbOutput))

        #Calculation of the loss
        loss = myLoss(pred, label)#*w
        loss = torch.sum(loss)
        #Save the loss for later visualization
        totLoss += loss.item()
        b+=data.shape[0]
        #Backpropagation of the error through the net
        loss /= data.shape[0]
        loss.backward()

        #Update of the weights and biaises of the net
        optimizer.step()

    if epoch == 500 :
        optimizer = optim.Adam(model.parameters(), lr = 0.00001)    #Optimization method
    # if epoch == 1000 :
    #     optimizer = optim.Adam(model.parameters(), lr = 0.000005)    #Optimization method


    finLoss.append(totLoss/b)
    print("Training Loss ", finLoss[-1])
    totLoss = 0
    b = 0

    model.eval()
    validLoss.append(validLoop(mydataloaderEval, model, myLoss, device))
    print("Evaluation Loss ", validLoss[-1])

# testLoop(mydataTest, model, myLoss, nbInput, device)

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

torch.save(model, "myModelRNNCumulIterCases.pt")



