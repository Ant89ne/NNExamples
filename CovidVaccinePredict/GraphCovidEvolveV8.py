import numpy as np
import matplotlib.pyplot as plt
import csv


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
    incrementC = np.round(incrementC, -magC)
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
        sl = str(l)             #Transfer to str
        s = len(sl)             #Size of the string
        nbpts = (s-1) // 3      #Number of '.' to add
        if nbpts > 0 :
            #Add each '.' one by one beginning by the end
            for j in range(1,nbpts+1): 
                sl = sl[:s-3*j] + '.' + sl[s-3*j:]
        #Save the value to the final list
        strlistValues.append(sl)
    
    return strlistValues

def visualization(listCases, listDeaths, countryName) :
    """
    Function used to visualize data with 4 plots : cases, deaths and their cumulative values
    2020 and 2021 data are plotted on the same graph with a different color

    @input listCases :      List of the cases values to be plotted
    @input listDeaths :     List of the deaths values to be plotted
    @input countryName :    Name of the country (or region) for titling and saving aspects
    """

    #Number of y axis labels
    nbSteps = 10
    #x-axis labels
    labels = ["Jan.", "Feb.", "Mar.", "Apr.", "May.", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."]

    #Get cumulative data
    cumulCases = [np.sum(listCases[:k+1]) for k in range(len(labels))] + [np.sum(listCases[len(labels):len(labels)+k+1]) for k in range(len(labels))]
    cumulDeaths = [np.sum(listDeaths[:k+1]) for k in range(len(labels))] + [np.sum(listDeaths[len(labels):len(labels)+k+1]) for k in range(len(labels))]

    #Get labels for each list of data
    labelsyc = getLabels(listCases, nbSteps)
    labelsyd = getLabels(listDeaths, nbSteps)
    labelscumc = getLabels(cumulCases, nbSteps)
    labelscumd = getLabels(cumulDeaths, nbSteps)

    #Get string labels for each list of labels
    strlabelsyc = toStr(labelsyc)
    strlabelsyd = toStr(labelsyd)
    strlabelscumc = toStr(labelscumc)
    strlabelscumd = toStr(labelscumd)

    #Get the true name (not considering the subfolders)
    cName = countryName[countryName.rfind('/')+1:]
    
    #Visualization
    plt.figure(0, figsize=(20,10))
    plt.clf()                                                                   #Clear the display to avoid accumulation of curves
    plt.subplot(221)                                                            #Plot on the top left graph
    plt.plot([k for k in range(len(labels))], listCases[:len(labels)], 'b*-')   #Cases for 2020
    plt.plot([k for k in range(len(labels))], listCases[len(labels):], 'r*-')   #Cases for 2021
    plt.xticks([k for k in range(len(labels))], labels, rotation=45)            #Display specific x-axis labels
    plt.yticks(labelsyc, strlabelsyc)                                           #Display specific y-axis labels
    plt.legend(["2020", "2021"])                                                #Legend
    plt.title("Cases per months in " + cName)                                   #Title

    plt.subplot(222)                                                            #Plot on the top right graph
    plt.plot([k for k in range(len(labels))], listDeaths[:len(labels)], 'b*-')  #Deaths in 2020
    plt.plot([k for k in range(len(labels))], listDeaths[len(labels):], 'r*-')  #Deaths in 2021
    plt.xticks([k for k in range(len(labels))], labels, rotation=45)            #Display specific x-axis labels
    plt.yticks(labelsyd, strlabelsyd)                                           #Display specific y-axis labels
    plt.legend(["2020", "2021"])                                                #Legend
    plt.title("Deaths per months in " + cName)                                  #Title
    
    plt.subplot(223)                                                            #Plot on the bottom left graph
    plt.plot([k for k in range(len(labels))], cumulCases[:len(labels)], 'b*-')  #Cumulative cases in 2020
    plt.plot([k for k in range(len(labels))], cumulCases[len(labels):], 'r*-')  #Cumulative cases in 2021
    plt.xticks([k for k in range(len(labels))], labels, rotation=45)            #Display specific x-axis labels
    plt.yticks(labelscumc, strlabelscumc)                                       #Display specific y-axis labels
    plt.legend(["2020", "2021"])                                                #Legend
    plt.title("Cumulative Cases per months in " + cName)                        #Title

    plt.subplot(224)                                                            #Plot on the bottom right graph
    plt.plot([k for k in range(len(labels))], cumulDeaths[:len(labels)], 'b*-') #Cumulative deaths in 2020
    plt.plot([k for k in range(len(labels))], cumulDeaths[len(labels):], 'r*-') #Cumulative deaths in 2021
    plt.xticks([k for k in range(len(labels))], labels, rotation=45)            #Display specific x-axis labels
    plt.yticks(labelscumd, strlabelscumd)                                       #Display specific y-axis labels
    plt.legend(["2020", "2021"])                                                #Legend
    plt.title("Cumulative Deaths per months in " + cName)                       #Title

    plt.savefig("VisFold8/" + countryName + "Graph.png")                        #Save the figure
    # plt.show()                                                                  #Show the figure to the user (/!\ should be commented for faster processing)


def visualizationAll(listCases, listDeaths, regionName) :
    """
    Function used to visualize data with 2 plots : cases and deaths in 2020 and 2021 for different regions

    @input listCases :      List of the cases values to be plotted
    @input listDeaths :     List of the deaths values to be plotted
    @input regionName :     List of names of the regions for titling and saving aspects
    """
    #Number of y axis labels
    nbSteps = 10

    #x-axis labels creation
    labels1 = ["Jan.", "Feb.", "Mar.", "Apr.", "May.", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."]
    labels = []
    for k in labels1 : 
        labels.append(k+"20")
    for k in labels1 :
        labels.append(k+"21")

    #Get labels for each list of data
    labelsyc = getLabels(listCases, nbSteps)
    labelsyd = getLabels(listDeaths, nbSteps)
    
    #Get string labels for each list of labels
    strlabelsyc = toStr(labelsyc)
    strlabelsyd = toStr(labelsyd)
    
    #Visualization
    plt.figure(0, figsize=(20,10))
    plt.clf()                                                           #Clear the display to avoid accumulation of curves
    plt.subplot(211)                                                    #Diplay on the top graph
    for i in range(len(regionName)) :                                   #Plot cases for each region
        plt.plot([k for k in range(len(labels))], listCases[i,:], '*-')
    plt.xticks([k for k in range(len(labels))], labels, rotation=45)    #Display specific x-axis labels
    plt.yticks(labelsyc, strlabelsyc)                                   #Display specific y-axis labels
    plt.legend(regionName)                                              #Legend
    plt.title("Cases per months")                                       #Title


    plt.subplot(212)                                                    #Diplay on the bottom graph
    for i in range(len(regionName)) :                                   #Plot deaths for each region
        plt.plot([k for k in range(len(labels))], listDeaths[i,:], '*-')
    plt.xticks([k for k in range(len(labels))], labels, rotation=45)    #Display specific x-axis labels
    plt.yticks(labelsyd, strlabelsyd)                                   #Display specific y-axis labels
    plt.legend(regionName)                                              #Legend
    plt.title("Deaths per months")                                      #Title

    plt.savefig("VisFold8/AllGraph.png")                                #Save the figure
    # plt.show()


##########################################################################################
#                                       Main Code                                        #
##########################################################################################

#Open data file
f = open("world_covid_data.csv")
#Read as a CSV file
csvreader = csv.reader(f, delimiter = ',')
#Extract the header
header = next(csvreader)
#Extract sepcific index of interest
for i, text in enumerate(header) : 
    if text == "New_cases":         #Number of cases
        indexCases = i
    if text == "New_deaths" :       #Number of deaths
        indexDeaths = i
    if text == "Date_reported":     #Date
        indexDate = i
    if text == "Country":           #Country
        indexCountry = i
    if text == "WHO_region":        #Name of the region
        indexRegion = i 

# Extract all regions names
L = []
for line in csvreader :
    if line[indexRegion] not in L :
        L.append(line[indexRegion])
f.close()

#Initialization
memDate = '01'
memCountry = "Afghanistan"
monthCases = 0
monthDeaths = 0

listCases = []
listDeaths = []

CasesRegions = np.zeros((6,24))
DeathsRegions = np.zeros((6,24))


#Reopen the file as a csv and extract the header
f = open("world_covid_data.csv")
csvreader = csv.reader(f, delimiter = ',')
header = next(csvreader)

#Beginning of the routine
for line in csvreader :                             #Read each line from the csv file
    date = (line[indexDate].split('-'))             #Extract the date

    #Check if the country is still the same
    if memCountry == line[indexCountry]:       
        #Check if the month is still the same
        if memDate == date[1]:                      
            monthCases += int(line[indexCases])     #Sum the cases
            monthDeaths += int(line[indexDeaths])   #Sum the deaths
        #If the month has changed
        else :                                      
            listCases.append(monthCases)            #Add values to the list of cases of the country
            listDeaths.append(monthDeaths)          #Add values to the list of deaths of the country
            #Reinitialization
            monthCases = 0      
            monthDeaths = 0
            memDate = date[1]                       #Change month
    
    #If the country has changed
    else :                                          
        listCases.append(monthCases)    #Add values to the list of cases of the previous country
        listDeaths.append(monthDeaths)  #Add values to the list of deaths of the previous country
        
        #Visualize the data for the previous country
        visualization(listCases, listDeaths, "Countries/" + memCountry)

        #Check if the country is in a region and get the region index in the list
        if line[indexRegion] != "Other":
            for i,k in enumerate(L) : 
                if k == line[indexRegion]:
                    index = i
                    break
            #Sum list values to the region cases 
            CasesRegions[i,:] += np.array(listCases)
            #Sum list values to the region deaths
            DeathsRegions[i,:] += np.array(listDeaths)

        #Reinitialization
        monthCases = 0
        monthDeaths = 0
        memCountry = line[indexCountry]
        memDate = '01'
        listCases = []
        listDeaths = []


#Visualization of each region independently
for i,k in enumerate(L[:-1]) :
    visualization(CasesRegions[i,:], DeathsRegions[i,:], "Regions/" +  k)

#Visualize the regions data on the same plot for compare
visualizationAll(CasesRegions, DeathsRegions, L[:-1])

