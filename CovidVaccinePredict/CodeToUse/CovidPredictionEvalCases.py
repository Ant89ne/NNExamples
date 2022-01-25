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
import os


def checkDirs(path):
    """
    Function used to check if a folder exist, if not the function create the folder
    @input path :   Path to check
    """
    if not os.path.isdir(path):
        os.mkdir(path)

def getLabels(listValues, nbSteps):
    """
    Function used to get the y axis labels equally spaced
    @input listValues :     List of the values (numerical) 
    @input nbSteps :        Number of labels to consider on the y axis

    @return labelsyc :      Labels for the y axis with the willing format
    """

    #Get the upper limit of the data
    nbmaxC = int(np.max([nbSteps, np.max(np.max(listValues))]))
    #Get the delta from a label to another
    incrementC = nbmaxC / nbSteps
    #Get the order of magnitude of the increment (for rounding later)
    magC = int(np.floor(np.log10(incrementC)))
    #Round the delta value
    incrementC = int(np.round(incrementC, -magC))

    #Create the list of labels with the correct delta
    labelsyc = [k for k in range(0, np.round(nbmaxC, -magC) + int(incrementC) +1,  int(incrementC))]

    return labelsyc

def toStr(listValues):
    """
    Function used to get the y axis labels with a '.' at each 10^3
    @input listValues :     List of the values (numerical) to be transfered to string with '.' 

    @return strlistValues :      Labels for the y axis with the willing format
    """
    #Initialization
    strlistValues = []

    #Transfer of each value one by one
    for i, l in enumerate(listValues):
        if l != 0 :
            sl = "{:.1e}".format(l)
        else :
            sl = str(0)
        #Save the value to the final list
        strlistValues.append(sl)
    
    return strlistValues

def getPop(country):
    popFile = "population_data.csv"
    f = open(popFile)
    csvFile = csv.reader(f, delimiter = ',')

    header = next(csvFile)

    for line in csvFile:
        if line[0] in country or country in line[0] :
            # print(country, line)
            return float(line[1]) * 1e3
        elif line[0] > country :
            break

    print(country + " Population not found")
    return  1e6

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
        f = open("2022predsRNNCases.csv", "w+")
        #Create csv writter
        csvwriter = csv.writer(f, delimiter = ',')

        #Get number of outputs
        nbOutput = fullDataset.nboutput

        #Create metric header for the csv file
        metricsHeader = []
        for i in range(nbOutput):
            metricsHeader = metricsHeader + ["MAE_week_" + str(i+1), "stdAE_week_"+str(i+1), "RMSE_week_"+str(i+1), "stdRMSE_week_"+str(i+1)]
        csvwriter.writerow(["Country", "Unit"] + ["Pred_week_" + str(i+1) for i in range(nbOutput)] +  metricsHeader)

        #Shape of the dataset
        s = fullDataset.dataCases.shape
        
        #Write line
        for c in range(s[0]):
            nbPop = getPop(fullDataset.country[c])/1e6

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

            if nbPop != 1 :
                unit = "per million inhabitant"
            else : 
                unit = ""

            #Prepare the line to be written
            line = [fullDataset.country[c], unit] + [str(int(p/nbPop)) for p in pred] + metline
            
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
        pred = torch.reshape(pred, (data.shape[0], 1, nbOutput))
        #Calculation of the loss
        loss = torch.sum(myLoss(pred, label))#*w)

        #Save the loss for later visualization
        totLoss += loss.item()
        b += data.shape[0]
    return totLoss/b


def cloudVis(data, preds, meanVals, c, nbInput, tit=""):
    """
    Function used for visualization
    @input data :       Ground truth values
    @input preds :      Prediction for each sample
    @input meanVals :   Mean value of the predictions for the same week
    @input c :          Country name
    @input nbInput :    Number of input of the network
    @input tit :        Whether the data are cumulated or not
    """
    nbPop = getPop(c) / 1e6
    data /= nbPop
    meanVals/= nbPop

    #Disable gradient calculation
    with torch.no_grad():
        plt.figure(0, figsize=(20,10))                                                      #Create new figure
        plt.clf()                                                                           #Clear the previous plots
        plt.plot([i for i in range(len(data))], data, '-r')                                 #Plot ground truth first time (for legend issues)
        plt.plot([i for i in range(nbInput, nbInput+len(meanVals))], meanVals, '-b')        #Plot mean values first time (for legend issues)
        
        for i,p in enumerate(preds) :
            p /= nbPop
            plt.plot([k+i+nbInput for k in range(len(p))], p, '*', alpha = 0.3)             #Plot predictions using stars
        
        plt.plot([i for i in range(len(data))], data, '-r')                                 #Plot ground truth (to be on the foreground)
        plt.plot([i for i in range(nbInput, nbInput+len(meanVals))], meanVals, '-b')        #Plot mean values (to be on the foreground)
        plt.title("Predictions for 5 weeks for " + c + ' ' + tit, fontsize = 20)            #Title of the plot
        plt.legend(["Ground Truth", "Mean Predictions", "Five weeks predictions"], fontsize = 15)   #Legend of the plot
        plt.axvline(nbInput,color = 'k', linestyle = '--')                                  #Create vertical line to show beginning of predictions
        plt.axvline(0,color='k')                                                            #Vertical line of beginning of 2020
        plt.axvline(len(data)/2, color = 'k')                                               #Vertical line separating 2020 and 2021
        plt.axvline(len(data)-1, color = 'k')                                               #Vertical line separating 2021 and 2022
        plt.text(1,1/2*np.max(data), "Input Data", fontsize = 15)                           #Text on the graph
        plt.text(len(data)/4-2,np.min(data) , "2020", fontsize = 15)                        #Text on the graph
        plt.text(3*len(data)/4-2,np.min(data) , "2021", fontsize = 15)                      #Text on the graph
        plt.text(len(data)+ 2,np.min(data) , "2022", fontsize = 15)                         #Text on the graph
        plt.xticks([i for i in range(0,len(data)+10,5)], [i for i in range(0,len(data)+10,5)], fontsize=12)     #x-axis values names
        ylab = getLabels(data, 10)  #Get ylabels
        plt.yticks(ylab, toStr(ylab), fontsize=12)                                          #y-axis values name
        plt.xlabel("Week number starting in 01-2020", fontsize = 15)                        #x-axis name
        if nbPop != 1 :
            plt.ylabel("Number of Cases per million inhabitant", fontsize = 15)                                        #y-axis name
        else : 
            plt.ylabel("Number of Cases", fontsize = 15)                                        #y-axis name
            
        #Check if the saving directory exists
        checkDirs("RNNResCases")
        #Save the figure
        plt.savefig("RNNResCases/" + c + tit[tit.find('('):tit.rfind(')')] + ".png")

def fullTestMultipleLoop(dataset, model, nbInput, nbOutput, device = "cpu"):
    """
    Function used to test the network and create plots
    @input dataset :        Dataset to use for evaluation
    @input model :          Model to be evaluated
    @input nbInput :        Number of input of the network
    @input nbOutput :       Number of predictions
    @input device :         Device of which the net should be run
    """

    #Switch data to evaluation mode
    model.eval()

    #Disable gradient calculation
    with torch.no_grad():            
        #Shape of the dataset
        s = dataset.dataCases.shape

        for d in range(s[0]):
            #Extract country data
            data = dataset.dataCases[d]
            #Initialization
            countryPreds = []
            meanPredict = np.zeros((1, len(data)-nbInput+nbOutput-1))
            nb = np.zeros((1, len(data)-nbInput+nbOutput-1))
            
            #Evaluate each sample of the country
            for k in range(len(data) - nbInput) :
                #Extract sample
                inp = data[k:k+nbInput]
                inp = torch.reshape(torch.Tensor(inp), (1, len(inp))).to(device)
                #Predictions of the model
                pred = model(inp)
                pred = torch.reshape(pred, (1,1,nbOutput))
                #Data values update
                countryPreds.append(pred.cpu().numpy()[0,0,:])
                meanPredict[0,k:k+nbOutput] += pred.cpu().numpy()[0,0,:]
                nb[0,k:k+nbOutput] += 1
            #Divide sumed values by number of sample (create the true mean)
            meanPredict /= nb
            #Visualization and saving of weekly predictions
            cloudVis(data * dataset.std[d] + dataset.mean[d], np.array(countryPreds) * dataset.std[d] + dataset.mean[d], meanPredict[0,:] * dataset.std[d] + dataset.mean[d], dataset.country[d], nbInput)

            #Cumulate ground truth values
            da = np.cumsum(data* dataset.std[d] + dataset.mean[d])
            #Cumulate predictions values
            countryPreds = np.cumsum(np.array(countryPreds)* dataset.std[d] + dataset.mean[d], axis = 1)
            #Cumulate mean values
            CumulMean = np.zeros((1, len(data)-nbInput+nbOutput-1))
            for k in range(len(data) - nbInput):
                countryPreds[k,:] += da[k+nbInput]
                CumulMean[0,k:k+nbOutput] += countryPreds[k,:]
            CumulMean/=nb

            #Visualization and saving of weekly cumulated values
            cloudVis(da, countryPreds, CumulMean[0], dataset.country[d], nbInput, tit = "(Cumulated)")

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
        #Create 5 predictions depending on the input and the predictions made by the network
        for k in range(self.nbOutput):
            #Get the kth prediction
            ret, h = self.gru(vals)
            #Extract the final value (as it is the real prediction)
            ret = self.lin(ret[:,-1,:])
            ret = torch.reshape(ret, (s[0],1,1))
            #Concatenate the prediction to the input so that next prediction take the current prediction into account
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

#Load the parameters of a saved model
m = torch.load("myModelRNNSingle.pt")
#Load the parameters inside the model
model.load_state_dict(m)
#Get metrics for train, eval and test sets
allMetrics = []
allMetrics.append(getMetrics(mydataloaderTrain, model))
allMetrics.append(getMetrics(mydataloaderEval, model))
allMetrics.append(getMetrics(mydataloadertest, model))

#Write the predictions and metrics on a csv file, for all countries
writeTable(fullDataSet, model, allMetrics, percentages)
#Visualize weekly and cumulated values
fullTestMultipleLoop(fullDataSet, model, nbInput, nbOutput, device)



exit()

# epoch = 1000      #Nb epoch to run

# # myLoss = nn.MSELoss(reduction='sum')   #Loss function 
# myLoss = nn.MSELoss(reduction='none')   #Loss function 

# optimizer = optim.Adam(model.parameters(), lr = 0.00005)#, betas = (0.8,0.9))    #Optimization method

# totLoss = 0        #List of losses for final visualization
# b = 0
# finLoss = []
# validLoss = []

# for e in range(epoch):
#     print("Epoch ", e+1, "/", epoch)
#     model.train()
#     for data, label, w in mydataloaderTrain:
#         data = data.to(device)
#         label = label.to(device)
#         w = w.to(device)

#         #Initialization of the gradient error
#         optimizer.zero_grad()
        
#         #Prediction of the labels from the input data by the net (y^)
#         pred = model(data)
#         pred = torch.reshape(pred, (data.shape[0], 1, nbOutput))

#         #Calculation of the loss
#         loss = myLoss(pred, label)#*w
#         loss = torch.sum(loss)
#         #Save the loss for later visualization
#         totLoss += loss.item()
#         b+=data.shape[0]
#         #Backpropagation of the error through the net
#         loss /= data.shape[0]
#         loss.backward()

#         #Update of the weights and biaises of the net
#         optimizer.step()

#     if epoch == 500 :
#         optimizer = optim.Adam(model.parameters(), lr = 0.00001)    #Optimization method
#     # if epoch == 1000 :
#     #     optimizer = optim.Adam(model.parameters(), lr = 0.000005)    #Optimization method


#     finLoss.append(totLoss/b)
#     print("Training Loss ", finLoss[-1])
#     totLoss = 0
#     b = 0

#     model.eval()
#     validLoss.append(validLoop(mydataloaderEval, model, myLoss, device))
#     print("Evaluation Loss ", validLoss[-1])

# # testLoop(mydataTest, model, myLoss, nbInput, device)

# plt.figure(0)
# plt.subplot(211)
# plt.plot([i for i in range(epoch)], finLoss, 'r')
# plt.plot([i for i in range(epoch)], validLoss, 'b')
# plt.legend(["Training Loss", "Evaluation Loss"])
# plt.subplot(212)
# plt.plot([i for i in range(100,epoch)], finLoss[100:], 'r')
# plt.plot([i for i in range(100,epoch)], validLoss[100:], 'b')
# plt.legend(["Training Loss", "Evaluation Loss"])
# plt.savefig("loss.png")
# plt.show()

# torch.save(model, "myModelRNN.pt")



