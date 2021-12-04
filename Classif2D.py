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
    def __init__(self):
        self.length = 1000
        self.a = 0.5
        self.b = 0.2
        self.seed = 25
        random.seed(self.seed)

    def __len__(self):
        return self.length
        
    def __getitem__(self, index):
        x1 = random.random()
        x2 = random.random()

        if x2 < (self.a*x1+self.b) :
            l = torch.Tensor(np.array([1,0]))
        else : 
            l = torch.Tensor(np.array([0,1]))
        
        X = torch.Tensor(np.array([x1,x2]))
        return X,l

        
#Initialization
mydata = myDataset()        #Dataset

plt.figure(0)
X = np.array([np.array(next(iter(mydata))[0]) for i in range(len(mydata)) ])
L = np.array([np.array(next(iter(mydata))[1]) for i in range(len(mydata)) ])
verif = X[0:10,:]
verif[:,0] = verif[:,0] * 0.5 + 0.2
print(verif, L[0:10,:])

X1 = []
X0 = []
for i,k in enumerate(L) :
    if k[0] == 1 :
        X0.append(X[i,:])
    elif k[1] == 1 :
        X1.append(X[i,:])



X0 = np.array(X0)
X1 = np.array(X1)

verif = X0[0:10,:]
verif[:,0] = verif[:,0] * 0.5 + 0.2
print(verif)
plt.plot(X0[:,0], X0[:,1], 'r*')
plt.plot(X1[:,0], X1[:,1], 'b*')
plt.savefig("pilou")

exit()
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

        #Prediction of the labels from the input data by the net
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



