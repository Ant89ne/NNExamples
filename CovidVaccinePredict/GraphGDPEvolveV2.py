import numpy as np
import matplotlib.pyplot as plt
import csv
import os

from numpy.core.numeric import NaN

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
    yvals = listValues[first:last+1]

    #Visualization
    plt.figure(0, figsize=(20,10))
    plt.clf()                                   #Clear the display
    plt.plot(xvals, yvals, '*-')
    plt.xticks([k for k in range(len(xlabels))], xlabels, rotation = 45)
    plt.title("Quarterly GDP for " + name[name.rfind('/')+1:])

    checkDirs("GDP2/")
    checkDirs("GDP2/" + name[:name.rfind('/')+1])
    plt.savefig("GDP2/" + name + "Graph.png")
    # plt.show()



####################################################################################
#                           Main routine                                           #
####################################################################################


#Open the data file
f = open("GDP.csv")
fbis = open("world_covid_data.csv")
#Read the file as a CSV
csvreader = csv.reader(f, delimiter = ',')
csvreaderbis = csv.reader(fbis, delimiter = ',')
#Extract the header
header = next(csvreader)
headerbis = next(csvreaderbis)

#Extract sepcific index of interest
for i, text in enumerate(headerbis) : 
    if text == "Country":           #Country
        indexCountry = i
    if text == "WHO_region":        #Name of the region
        indexRegion = i 

# Extract all regions names
L = []
for line in csvreaderbis :
    if line[indexRegion] not in L :
        L.append(line[indexRegion])
fbis.close()

#Reopen the file as a csv and extract the header
fbis = open("world_covid_data.csv")
csvreaderbis = csv.reader(fbis, delimiter = ',')
headerbis = next(csvreaderbis)

compareLine = next(csvreaderbis)

#Initialization
GDPRegion = np.zeros((len(L)-1, len(header[2:])))
nbToDivide = np.zeros((len(L)-1, len(header[2:])))

#Beginning of the routine : read each line of the csv
for line in csvreader :
    countryName = line[0]           #Extract the country name (visualizing purposes)
    checkName = countryName[:countryName.rfind(',')]    #Extract part of the name in common with fbis

    #Looking for region
    while checkName not in compareLine[indexCountry] and checkName > compareLine[indexCountry]:
        compareLine = next(csvreaderbis)

    #Look for the region of the country
    if checkName in compareLine[indexCountry]:
        region = compareLine[indexRegion]
    else :
        region = "Other"
    print(region, checkName)
    data = line[2:]                 #Extract the data
    first, last = -1,-1             #First and Last not None values
    #Look for effective data
    for i,d in enumerate(data) : 
        try :                       #Try to convert the data to int value
            d = d.replace(',', '.') #Switch ',' to '.' to be able to consider data as float
            data[i] = float(d)*10e6 #Convertion
            if first == -1 :        #If it is the first effective data : set the index
                first = i
            last = i                #Set the index of the last effective data
        except ValueError :         #If an error has been raised, set value to None
            data[i] = np.NaN
    
    #Convert data to array
    data = np.array(data)

    #Look for a specific index where the region is located in L
    if region != "Other":
        for i,k in enumerate(L) : 
            if k == region:
                index = i
                break
        
        nonNanIndex = np.argwhere(1-np.isnan(data))
        GDPRegion[index,nonNanIndex] += data[nonNanIndex] # * (1-np.isnan(data))
        nbToDivide[index,:] += (1-np.isnan(data))

    #Visualization step
    visualizeData(data, first, last, "Country/"+ countryName, header[2:])

#Visualize the data for each region
for i, k in enumerate(GDPRegion) :
    visualizeData(k, 0, len(k)-1, "Region/" + L[i], header[2:])
