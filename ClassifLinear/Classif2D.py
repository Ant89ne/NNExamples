from os import path
import numpy as np
import torch 
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset, ConcatDataset
import matplotlib.pyplot as plt
import time
import torch.optim as optim
import random


class myNet(nn.Module):
    def __init__(self):
        pass
    def forward(self, x):
        pass


class myDataset(Dataset):
    def __init__(self):         #Function TO BE filled
        #Initialization
        self.length = 1000  #Number of points in the dataset  
        self.a = 0.5        #Direction of the line
        self.b = 0.2        #Biais of the line
        self.seed = 25      #Seed (for random generation)
        random.seed(self.seed)

    def __len__(self):          #Function TO BE filled
        #Return the length of the dataset
        return self.length
        
    def __getitem__(self, index): #Function TO BE filled
        #Return the data at the given index
        x1 = random.random()    #Creation of the coordinates
        x2 = random.random()

        #Creation of the label (0 if under the line, 1 if above)
        if x2 < (self.a*x1+self.b) :
            l = torch.Tensor(np.array([1,0]))
        else : 
            l = torch.Tensor(np.array([0,1]))
        
        #Convert from array to Tensor (for training purpose)
        X = torch.Tensor(np.array([x1,x2]))
        return X,l

        
#Initialization
mydata = myDataset()        #Dataset

#Dataset visualization
plt.figure(0)
#Get data
X = np.array([np.array(next(iter(mydata))) for i in range(len(mydata)) ])
#Extract labels (y)
L = np.array([np.array(x) for x in X[:,1]])
#Extracts coordinates (x)
X = np.array([np.array(x) for x in X[:,0]])
#Group data by labels
X1 = []
X0 = []
for i,k in enumerate(L) :
    if k[0] == 1 :
        X0.append(X[i,:])
    elif k[1] == 1 :
        X1.append(X[i,:])
X0 = np.array(X0)
X1 = np.array(X1)
#Visualize and save
plt.plot(X0[:,0], X0[:,1], 'r*')
plt.plot(X1[:,0], X1[:,1], 'b*')
plt.savefig("pilou")




model = myNet()             #Neural Net

mydataloader = DataLoader(mydata, 8, shuffle = True)    #Dataloader

epoch = 10      #Nb epoch to run

myLoss = nn.BCELoss()   #Loss function 

optimizer = optim.SGD(model.parameters(), lr = 0.01)    #Optimization method

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



plt.figure(0)
plt.plot([i for i in range(epoch)], totLoss)



