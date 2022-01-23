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
    """
    Function used to calculate the mean and standard deviation of the Absolute Error and the Sqared error

    @input dataloader : Dataloader used for evaluation
    @input model :      Model to evaluate
    @input device :     Device on which the net can be run

    @output meanValae :     MAE values for each prediction (t+1, t+2, ...)
    @output stdValae :      Standard deviation of Absolute error
    @output meanValse :     RMSE
    @output stdValse :      Root standard deviation of sqared error
    """
    #Set the model on evaluation mode
    model.eval()
    #Disable the gradient calculation of data tensors
    with torch.no_grad():
        #Initialization
        listae = [[] for i in range(5)]
        listse = [[] for i in range(5)]
        
        for d, l, w in dataloader :
            #Send data to the device selected
            d = d.to(device)
            l = l.to(device)
            #Prediction
            pred = model(d)
            #Calculate Absolute Error
            ae = torch.absolute(pred-l).cpu()
            #Calculate Squared error
            se = torch.square(pred-l).cpu()

            #Add values to respective lists
            for k in range(d.shape[0]):
                for i in range(5):
                    listae[i].append(ae[k,0,i].item())
                    listse[i].append(se[k,0,i].item())

        #Switch to arrays (for practical use)
        listae = np.array(listae)
        listse = np.array(listse)

        #Calculate mean and standard deviation
        meanValae = np.mean(listae, axis = 1)
        stdValae = np.std(listae, axis = 1)
        meanValse = np.mean(listse, axis = 1)
        stdValse = np.std(listse, axis = 1)

        return meanValae, stdValae, np.sqrt(meanValse), np.sqrt(stdValse)

def writeTable(fullDataset, model, metrics, percentages, device = "cuda"):
    """
    Write the table with the predictions and the metrics
    @input fullDataset :    Dataset with data to predict and to calculate metrics
    @input model :          Model to evaluate
    @input metrics :        Metrics to save
    @input percentages :    Percentages of train, eval and test dataset (for metrics identification)
    @input device :         Device on which the net can be run    
    """

    #Switch model to evaluation mode
    model.eval()

    #Disable gradient calculation for data tensors and model
    with torch.no_grad():
        #Open a file to save values (csv)
        f = open("2022preds.csv", "w+")
        #Create csv writter
        csvwriter = csv.writer(f, delimiter = ',')

        #Get number of outputs
        nbOutput = fullDataset.nboutput

        #Create metric header for the csv file
        metricsHeader = []
        for i in range(nbOutput):
            metricsHeader = metricsHeader + ["MAE_week_" + str(i+1), "stdAE_week_"+str(i+1), "RMSE_week_"+str(i+1), "stdRMSE_week_"+str(i+1)]
        csvwriter.writerow(["Country"] + ["Pred_week_" + str(i+1) for i in range(nbOutput)] +  metricsHeader)

        #Shape of the dataset
        s = fullDataset.dataCases.shape

        #Write line
        for c in range(s[0]):
            #Flag used to determine which type of data the country belongs to (train, eval or test)
            flag = (c/s[0] > percentages[0]) + (c/s[0] > (percentages[0] + percentages[1]))

            #Get the final sample to predict values for the future
            data = fullDataset.dataCases[c,-fullDataset.nbinput:]

            #Predictions of the model
            pred = model(torch.reshape(torch.Tensor(np.array(data)).to(device), (1, fullDataset.nbinput)))

            #Get mean and standard deviation of the country
            mu = fullDataset.mean[c]
            sig = fullDataset.std[c]
            #Resize the values to fit to true values (sort of "denormalization")
            met = np.array(metrics[flag]) * sig + mu 
            pred = pred.cpu().numpy()[0,0,:] * sig + mu

            #Prepare the metrics data to be written
            metline = []
            for i in range(nbOutput):
                metline = metline + [str(int(m)) for m in met[:,i]]

            #Prepare the line to be written
            line = [fullDataset.country[c]] + [str(int(p)) for p in pred] + metline
            
            #Write the line
            csvwriter.writerow(line)
        

            


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

    #Read first data line
    line = next(csvreader)
    #Get data values
    countryData = [int(line[indexCases])]
    #Get country name
    country = line[indexCountry]

    #Go through each line
    for line in csvreader : 
        #Country name
        c = line[indexCountry]

        #Check if the country has changed
        if country == c :  #If no changes : add values to the country data
            countryData.append(int(line[indexCases]))
        else :  #Else
            #Check if data are relevant
            if np.sum(np.array(countryData)) != 0 :
                #Add data to the dataset
                dataCases.append(countryData)
                #Add country to list of countries of the dataset
                countries.append(country)
            #Reinitialization with new country values
            countryData = [int(line[indexCases])]
            country = c

    #Final add (data to dataset and contry to list of countries)
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

    #Read first data line
    line = next(csvreader)
    #Get data values
    countryData = [int(line[indexCases])]
    #Get country name
    country = line[indexCountry]
    
    #Initialization
    countries = []
    week = 0
    
    #Go through each line
    for line in csvreader : 
        #Country name
        c = line[indexCountry]

        #Check if the country is still the same and if we have go through 7 days (= a week)
        if country == c and week >= 7:
            #Add data value to data list
            countryData.append(int(line[indexCases]))
            #Reinitialization of the week count
            week = 0
        
        #Check if the country is the same and if if have not fo through 7 days
        elif country == c and week < 7:
            #Sum datacases to the current value
            countryData[-1] += int(line[indexCases])
            #Add a day to the week count
            week += 1
        
        #Check if the country has changed
        elif country != c :
            #Check if the data is relevant
            if np.sum(np.array(countryData)) != 0 :
                #Append data to the data list of the dataset
                dataCases.append(countryData)
                #Add the country to country list of the dataset
                countries.append(country)

            #Reinitialization
            countryData = [int(line[indexCases])]
            country = c
            week = 0

    #Final country appended to the dataset
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

    #Read first data line
    line = next(csvreader)
    #Get data values
    countryData = [int(line[indexCases])]
    #Get country name
    country = line[indexCountry]
    
    #Get month number
    month = line[indexDate].split('-')[1]

    #Go through each line
    for line in csvreader : 
        #Country name
        c = line[indexCountry]
        #Month number
        m = line[indexDate].split('-')[1]
        #CHeck if the country has not changed and the month has changed
        if country == c and month != m:
            #Append value to data list
            countryData.append(int(line[indexCases]))
            #Change month memory
            month = m
        
        #Check if the month has not changed neither the country
        elif country == c and month == m:
            #Cumulate value with the previous ones
            countryData[-1] += int(line[indexCases])

        #Check if the country has changed
        elif country != c :
            #Check if the data is relevant
            if np.sum(np.array(countryData)) != 0 :
                #Append data list to dataset
                dataCases.append(countryData)
                #Append country to dataset country list
                countries.append(country)
            #Reinitialization
            countryData = [int(line[indexCases])]
            country = c
            month = m

    #Final country data appended
    dataCases.append(countryData)
    countries.append(country)

    return dataCases, countries


def validLoop(dataloader, model, myLoss, device = "cpu"):
    """
    Function used for validation of the model
    @input dataloader :     Evaluation dataloader
    @input model :          Model to be evaluated
    @input myLoss :         Loss used for evaluation
    @input device :         Device on which the net can be run

    @output totLoss :       Evaluation loss
    """
    #Switch model to evaluation mode
    model.eval()
    #Initialization
    totLoss = 0
    b = 0

    #Evaluation
    for data, label, w in dataloader:
        #Send data to the device selected
        data = data.to(device)
        label = label.to(device)
        w = w.to(device)

        #Prediction of the labels from the input data by the net (y^)
        pred = model(data)
        #pred = torch.reshape(pred, (data.shape[0], 1, nbOutput))
        #Calculation of the loss
        loss = torch.sum(myLoss(pred, label[:,:,0]))#*w)

        #Save the loss for later visualization
        totLoss += loss.item()
        b += data.shape[0]
    return totLoss/b

class myRNN(nn.Module):
    """
    Class for the RNN with 5 predictions !!!
    """
    def __init__(self, nbInput, nbOutput):
        super(myRNN, self).__init__()
        
        self.nbInput = nbInput
        self.nbOutput = nbOutput
        self.hidNb = 100

        #RNN layer, here used GRU network (Cho et al, 2014)
        self.gru = nn.GRU(input_size = 1, hidden_size = self.hidNb, batch_first = True, dropout=0.5, num_layers=2)
        #Add a FCN after to compute the final prediction (single prediction)
        self.lin = nn.Linear(self.hidNb, 1)
        
    def forward(self, x):
        s = x.shape
        x = torch.reshape(x, (s[0],s[1], 1))
        vals = x    
        #Single prediction
        ret, h = self.gru(vals)
        #Extract the final value (as it is the real prediction)
        ret = self.lin(ret[:,-1,:])

        return ret


class myDatasetIter(Dataset):
    def __init__(self, myFile, nbinput, nboutput, temp = "daily", percentages = [0.8,0.1,0.1], typeIndex = 0):         #Function TO BE filled
        #Initialization
        self.file = myFile
        self.nbinput = nbinput
        self.nboutput = nboutput
        self.temp = temp
        self.dataCases = []
        self.country = []

        #Read data values
        with open(self.file) as datafile:
            csvreader = csv.reader(datafile, delimiter = ',')
            
            #Get the data depending on the temporality chosen
            if self.temp == "daily" :
                self.dataCases, self.country = getDailyData(csvreader)
            elif self.temp == "monthly":
                self.dataCases, self.country = getMonthlyData(csvreader)
            elif self.temp == "weekly":
                self.dataCases, self.country  = getWeeklyData(csvreader)


        self.dataCases = np.array(self.dataCases).astype("double")
        #Extract mean values per country
        self.mean = np.mean(self.dataCases, axis = 1)
        #Extract standard deviation for each country
        self.std = np.std(self.dataCases, axis = 1)

        s = self.dataCases.shape
        #Normalization for each country
        for i in range(s[0]):
            self.dataCases[i,:] = (self.dataCases[i,:] - self.mean[i])/self.std[i]

        #Get the percentages of values to keep in the dataset
        indexMin = int(np.sum(percentages[:typeIndex])*s[0])
        indexMax = int(np.sum(percentages[:typeIndex+1])*s[0])

        #Keep only countries of interest
        self.dataCases = self.dataCases[indexMin:indexMax,:]
        self.country = self.country[indexMin:indexMax]

        #Get min, max values
        self.minis = np.min(self.dataCases, axis = -1)
        self.maxis = np.max(self.dataCases, axis = -1)

    def __len__(self):          #Function TO BE filled
        #Return the length of the dataset
        s = self.dataCases.shape
        return s[0] * (s[1]-self.nbinput - self.nboutput)
        
    def __getitem__(self, index): #Function TO BE filled
        s = self.dataCases.shape
        #Get the line index of the country
        i = index//(s[1]-self.nbinput-self.nboutput)
        #Get the column index of the sample
        j = index - i*(s[1]-self.nbinput-self.nboutput)
        #Extract the sample
        d = self.dataCases[i][j:j+self.nbinput]
        #Extract the labels
        l = self.dataCases[i][j+self.nbinput:j+self.nbinput+self.nboutput]

        #NOT IMPORTANT - USED FOR TESTS BUT NOT RELEVANT AT ALL
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

#Creation of datasets
mydataTrain = myDatasetIter("world_covid_data.csv", nbInput, nbOutput, temporality, percentages, 0)        #Dataset Train
mydataEval = myDatasetIter("world_covid_data.csv", nbInput, nbOutput, temporality, percentages, 1)        #Dataset Evaluation
mydataTest = myDatasetIter("world_covid_data.csv", nbInput, nbOutput, temporality, percentages, 2)        #Dataset Test
brazilSet = myDatasetIter("brazilData.csv", nbInput, nbOutput, temporality, [1], 0)                       #Dataset with top 5 contaminated countries
fullDataSet = myDatasetIter("world_covid_data.csv", nbInput, nbOutput, temporality, [1], 0)                 #Dataset with all countries

#Creation of the model
model = myRNN(nbInput, nbOutput)             #Neural Net
#Send the model to the device available
model.to(device)

#Create Dataloaders
mydataloaderTrain = DataLoader(mydataTrain, 64, shuffle = True)    #Dataloader Train
mydataloaderEval = DataLoader(mydataEval, 64, shuffle = False)    #Dataloader Eval
mydataloadertest = DataLoader(mydataTest, 1, shuffle = False)    #Dataloader Test

#Nb epoch to run
epoch = 1000      

#Loss function
myLoss = nn.L1Loss(reduction='none')    

#Optimization method
optimizer = optim.Adam(model.parameters(), lr = 0.00005)#, betas = (0.8,0.9))

#Initialization    
totLoss = 0        
b = 0
finLoss = []
validLoss = []

for e in range(epoch):
    print("Epoch ", e+1, "/", epoch)
    #Set model to training mode
    model.train()

    for data, label, w in mydataloaderTrain:
        #Send data to the device selected
        data = data.to(device)
        label = label.to(device)[:,:,0]
        w = w.to(device)

        #Initialization of the gradient error
        optimizer.zero_grad()
        
        #Prediction of the labels from the input data by the net (y^)
        pred = model(data)

        #Calculation of the loss
        loss = myLoss(pred, label)
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
        optimizer = optim.Adam(model.parameters(), lr = 0.000005)    #Optimization method



    finLoss.append(totLoss/b)
    print("Training Loss ", finLoss[-1])
    totLoss = 0
    b = 0
    
    #Set model to evaluation mode for validation
    model.eval()
    validLoss.append(validLoop(mydataloaderEval, model, myLoss, device))
    print("Evaluation Loss ", validLoss[-1])

#Visualization of the training and evaluation loss
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

#Save the model
torch.save(model.state_dict(), "myModelRNNSingle.pt")






# model = torch.load("myModelMultiple.pt")
# allMetrics = []
# allMetrics.append(getMetrics(mydataloaderTrain, model))
# allMetrics.append(getMetrics(mydataloaderEval, model))
# allMetrics.append(getMetrics(mydataloadertest, model))

# writeTable(fullDataSet, model, allMetrics, percentages)

# # fullTestMultipleLoop(brazilSet, model, nbInput, nbOutput, device)
# exit()
