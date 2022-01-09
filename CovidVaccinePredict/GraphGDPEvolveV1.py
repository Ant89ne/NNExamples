import numpy as np
import matplotlib.pyplot as plt
import csv
import os

def checkDirs(path):
    """
    Function used to check if a folder exist, if not the function create the folder
    @input path :   Path to check
    """
    if not os.path.isdir(path):
        os.mkdir(path)

def visualizeData(listValues, first, last, name, xlabels):
    """
    Function used to visualize the GDP
    @input listValues :     List of the GDP to be displayed
    @input first :          First index of the effective data
    @input last :           Last index of the effective data
    @input xlabels :        Labels to display on the x-axis (temporality)
    """

    #Effective data to be plotted
    xvals = [i for i in range(first, last+1)]
    yvals = data[first:last+1]

    #Visualization
    plt.figure(0, figsize=(20,10))
    plt.clf()                                   #Clear the display
    plt.plot(xvals, yvals, '*-')
    plt.xticks([k for k in range(len(xlabels))], xlabels, rotation = 45)
    plt.title("Quarterly GDP for " + name[name.rfind('/')+1:])

    checkDirs("GDP/")
    checkDirs("GDP/" + name[:name.rfind('/')+1])
    plt.savefig("GDP/" + name + "Graph.png")
    # plt.show()



####################################################################################
#                           Main routine                                           #
####################################################################################


#Open the data file
f = open("GDP.csv")
#Read the file as a CSV
csvreader = csv.reader(f)
#Extract the header
header = next(csvreader)

#Beginning of the routine : read each line of the csv
for line in csvreader :
    countryName = line[0]           #Extract the country name (visualizing purposes)
    data = line[2:]                 #Extract the data
    first, last = -1,-1             #First and Last not None values
    #Look for effective data
    for i,d in enumerate(data) : 
        try :                       #Try to convert the data to int value
            d = d.replace(',', '.') #Switch ',' to '.' to be able to consider data as float
            data[i] = float(d)      #Convertion
            if first == -1 :        #If it is the first effective data : set the index
                first = i
            last = i                #Set the index of the last effective data
        except ValueError :         #If an error has been raised, set value to None
            data[i] = None
    
    #Visualization step
    visualizeData(data, first, last, countryName, header[2:])