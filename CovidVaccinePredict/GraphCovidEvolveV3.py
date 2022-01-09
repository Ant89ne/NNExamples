import numpy as np
import matplotlib.pyplot as plt
import csv


def visualization(listCases, listDeaths, countryName) :

    nbSteps = 10
    nbmaxC = np.max([nbSteps, np.max(listCases)])
    nbmaxD = np.max([nbSteps, np.max(listDeaths)])

    incrementC = nbmaxC / nbSteps
    incrementD = nbmaxD / nbSteps
    
    magC = int(np.floor(np.log10(incrementC)))
    magD = int(np.floor(np.log10(incrementD)))

    incrementC = np.round(incrementC, -magC)
    incrementD = np.round(incrementD, -magD)
    

    labels = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    labelsy = [k for k in range(0, np.round(nbmaxC, -magC) + int(incrementC) +1,  int(incrementC))] #np.round(np.max(listCases), -4)+1,
    labelsyd = [k for k in range(0, np.round(nbmaxD, -magD) + int(incrementD) +1, int(incrementD))] #np.round(np.max(listDeaths), -3)+1,


    plt.figure(0, figsize=(20,10))
    plt.clf()
    plt.subplot(121)
    plt.plot([k for k in range(len(labels))], listCases[:len(labels)], 'b*-')
    plt.plot([k for k in range(len(labels))], listCases[len(labels):], 'r*-')
    plt.xticks([k for k in range(len(labels))], labels, rotation=45)
    plt.yticks(labelsy, labelsy)

    plt.legend(["2020", "2021"])
    plt.title("Cases per months in " + countryName)

    plt.subplot(122)
    plt.plot([k for k in range(len(labels))], listDeaths[:len(labels)], 'b*-')
    plt.plot([k for k in range(len(labels))], listDeaths[len(labels):], 'r*-')
    plt.xticks([k for k in range(len(labels))], labels, rotation=45)
    plt.yticks(labelsyd, labelsyd)
    plt.legend(["2020", "2021"])
    plt.title("Deaths per months in " + countryName)
    plt.savefig("VisFold/" + countryName + "Graph.png")
    




f = open("world_covid_data.csv")

csvreader = csv.reader(f, delimiter = ',')

header = next(csvreader)
print(header)

for i, text in enumerate(header) : 
    if text == "New_cases":
        indexCases = i
    if text == "New_deaths" :
        indexDeaths = i
    if text == "Date_reported":
        indexDate = i
    if text == "Country":
        indexCountry = i

memDate = '01'
memCountry = "Afghanistan"
monthCases = 0
monthDeaths = 0

listCases = []
listDeaths = []

for line in csvreader : 
    date = (line[indexDate].split('-'))
    
    if memCountry == line[indexCountry]:
        if memDate == date[1]:
            monthCases += int(line[indexCases])
            monthDeaths += int(line[indexDeaths])
        else :
            listCases.append(monthCases)
            listDeaths.append(monthDeaths)
            monthCases = 0
            monthDeaths = 0
            memDate = date[1]
    else :
        listCases.append(monthCases)
        listDeaths.append(monthDeaths)

        visualization(listCases, listDeaths, memCountry)

        monthCases = 0
        monthDeaths = 0
        memCountry = line[indexCountry]
        memDate = '01'
        listCases = []
        listDeaths = []
        

# labels = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
# labelsy = [k for k in range(0, np.round(np.max(allCases), -4)+1, 5000)]
# labelsyd = [k for k in range(0, np.round(np.max(allDeaths), -3)+1, 100)]


# plt.figure(0, figsize=(30,30))
# plt.subplot(121)
# plt.plot([k for k in range(len(labels))], listCases[:len(labels)], 'b*-')
# plt.plot([k for k in range(len(labels))], listCases[len(labels):], 'r*-')
# plt.xticks([k for k in range(len(labels))], labels, rotation=45)
# plt.yticks(labelsy, labelsy)

# plt.legend(["2020", "2021"])
# plt.title("Cases per months")

# plt.subplot(122)
# plt.plot([k for k in range(len(labels))], listDeaths[:len(labels)], 'b*-')
# plt.plot([k for k in range(len(labels))], listDeaths[len(labels):], 'r*-')
# plt.xticks([k for k in range(len(labels))], labels, rotation=45)
# plt.yticks(labelsyd, labelsyd)
# plt.legend(["2020", "2021"])
# plt.title("Deaths per months")
# plt.savefig("AfghData.png")
# plt.show()
