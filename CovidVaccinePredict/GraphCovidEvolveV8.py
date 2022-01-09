import numpy as np
import matplotlib.pyplot as plt
import csv


def getLabels(listValues, nbSteps):
    nbmaxC = int(np.max([nbSteps, np.max(np.max(listValues))]))
    incrementC = nbmaxC / nbSteps
    magC = int(np.floor(np.log10(incrementC)))
    incrementC = np.round(incrementC, -magC)
    labelsyc = [k for k in range(0, np.round(nbmaxC, -magC) + int(incrementC) +1,  int(incrementC))] #np.round(np.max(listCases), -4)+1,
    return labelsyc

def toStr(listValues):
    strlistValues = []
    for i, l in enumerate(listValues):
        sl = str(l)
        s = len(sl)
        nbpts = (s-1) // 3
        if nbpts > 0 :
            for j in range(1,nbpts+1): 
                sl = sl[:s-3*j] + '.' + sl[s-3*j:]
        strlistValues.append(sl)
    
    return strlistValues

def visualization(listCases, listDeaths, countryName) :

    nbSteps = 10
    labels = ["Jan.", "Feb.", "Mar.", "Apr.", "May.", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."]

    cumulCases = [np.sum(listCases[:k+1]) for k in range(len(labels))] + [np.sum(listCases[len(labels):len(labels)+k+1]) for k in range(len(labels))]
    cumulDeaths = [np.sum(listDeaths[:k+1]) for k in range(len(labels))] + [np.sum(listDeaths[len(labels):len(labels)+k+1]) for k in range(len(labels))]

    labelsyc = getLabels(listCases, nbSteps)
    labelsyd = getLabels(listDeaths, nbSteps)
    labelscumc = getLabels(cumulCases, nbSteps)
    labelscumd = getLabels(cumulDeaths, nbSteps)


    strlabelsyc = toStr(labelsyc)
    strlabelsyd = toStr(labelsyd)
    strlabelscumc = toStr(labelscumc)
    strlabelscumd = toStr(labelscumd)
    # print(labelscumc, labelscumd)
    
    cName = countryName[countryName.rfind('/')+1:]
    
    plt.figure(0, figsize=(20,10))
    plt.clf()
    plt.subplot(221)
    plt.plot([k for k in range(len(labels))], listCases[:len(labels)], 'b*-')
    plt.plot([k for k in range(len(labels))], listCases[len(labels):], 'r*-')
    plt.xticks([k for k in range(len(labels))], labels, rotation=45)
    plt.yticks(labelsyc, strlabelsyc)

    plt.legend(["2020", "2021"])
    plt.title("Cases per months in " + cName)

    plt.subplot(222)
    plt.plot([k for k in range(len(labels))], listDeaths[:len(labels)], 'b*-')
    plt.plot([k for k in range(len(labels))], listDeaths[len(labels):], 'r*-')
    plt.xticks([k for k in range(len(labels))], labels, rotation=45)
    plt.yticks(labelsyd, strlabelsyd)
    plt.legend(["2020", "2021"])
    plt.title("Deaths per months in " + cName)
    
    plt.subplot(223)
    plt.plot([k for k in range(len(labels))], cumulCases[:len(labels)], 'b*-')
    plt.plot([k for k in range(len(labels))], cumulCases[len(labels):], 'r*-')
    plt.xticks([k for k in range(len(labels))], labels, rotation=45)
    plt.yticks(labelscumc, strlabelscumc)
    plt.legend(["2020", "2021"])
    plt.title("Cumulative Cases per months in " + cName)

    plt.subplot(224)
    plt.plot([k for k in range(len(labels))], cumulDeaths[:len(labels)], 'b*-')
    plt.plot([k for k in range(len(labels))], cumulDeaths[len(labels):], 'r*-')
    plt.xticks([k for k in range(len(labels))], labels, rotation=45)
    plt.yticks(labelscumd, strlabelscumd)
    plt.legend(["2020", "2021"])
    plt.title("Cumulative Deaths per months in " + cName)

    plt.savefig("VisFold8/" + countryName + "Graph.png")
    # plt.show()


def visualizationAll(listCases, listDeaths, regionName) :

    nbSteps = 10
    labels1 = ["Jan.", "Feb.", "Mar.", "Apr.", "May.", "Jun.", "Jul.", "Aug.", "Sep.", "Oct.", "Nov.", "Dec."]
    
    labels = []
    for k in labels1 : 
        labels.append(k+"20")
    for k in labels1 :
        labels.append(k+"21")


    # cumulCases = [np.sum(listCases[:k+1]) for k in range(len(labels))] + [np.sum(listCases[len(labels):len(labels)+k+1]) for k in range(len(labels))]
    # cumulDeaths = [np.sum(listDeaths[:k+1]) for k in range(len(labels))] + [np.sum(listDeaths[len(labels):len(labels)+k+1]) for k in range(len(labels))]

    labelsyc = getLabels(listCases, nbSteps)
    labelsyd = getLabels(listDeaths, nbSteps)
    # labelscumc = getLabels(cumulCases, nbSteps)
    # labelscumd = getLabels(cumulDeaths, nbSteps)


    strlabelsyc = toStr(labelsyc)
    strlabelsyd = toStr(labelsyd)
    # strlabelscumc = toStr(labelscumc)
    # strlabelscumd = toStr(labelscumd)
    # print(labelscumc, labelscumd)
    
    # cName = countryName[countryName.rfind('/')+1:]
    
    plt.figure(0, figsize=(20,10))
    plt.clf()
    plt.subplot(211)
    for i in range(len(regionName)) :
        plt.plot([k for k in range(len(labels))], listCases[i,:], '*-')
    plt.xticks([k for k in range(len(labels))], labels, rotation=45)
    plt.yticks(labelsyc, strlabelsyc)
    plt.legend(regionName)
    plt.title("Cases per months")

    # plt.subplot(222)
    # for i in range(len(regionName)) :
    #     plt.plot([k for k in range(len(labels))], listCases[i,len(labels):], '*-')
    # plt.xticks([k for k in range(len(labels))], labels, rotation=45)
    # plt.yticks(labelsyc, strlabelsyc)
    # plt.legend(regionName)
    # plt.title("Cases per months in 2021")

    plt.subplot(212)
    for i in range(len(regionName)) :
        plt.plot([k for k in range(len(labels))], listDeaths[i,:], '*-')
    plt.xticks([k for k in range(len(labels))], labels, rotation=45)
    plt.yticks(labelsyd, strlabelsyd)
    plt.legend(regionName)
    plt.title("Deaths per months")

    # plt.subplot(224)
    # for i in range(len(regionName)) :
    #     plt.plot([k for k in range(len(labels))], listDeaths[i,len(labels):], '*-')
    # plt.xticks([k for k in range(len(labels))], labels, rotation=45)
    # plt.yticks(labelsyd, strlabelsyd)
    # plt.legend(regionName)
    # plt.title("Deaths per months in 2021")

    # plt.xticks([k for k in range(len(labels))], labels, rotation=45)
    # plt.yticks(labelsyd, strlabelsyd)
    # plt.legend(["2020", "2021"])
    # plt.title("Deaths per months in " + cName)
    # plt.plot([k for k in range(len(labels))], listDeaths[len(labels):], 'r*-')



    # plt.subplot(223)
    # plt.plot([k for k in range(len(labels))], cumulCases[:len(labels)], 'b*-')
    # plt.plot([k for k in range(len(labels))], cumulCases[len(labels):], 'r*-')
    # plt.xticks([k for k in range(len(labels))], labels, rotation=45)
    # plt.yticks(labelscumc, strlabelscumc)
    # plt.legend(["2020", "2021"])
    # plt.title("Cumulative Cases per months in " + cName)

    # plt.subplot(224)
    # plt.plot([k for k in range(len(labels))], cumulDeaths[:len(labels)], 'b*-')
    # plt.plot([k for k in range(len(labels))], cumulDeaths[len(labels):], 'r*-')
    # plt.xticks([k for k in range(len(labels))], labels, rotation=45)
    # plt.yticks(labelscumd, strlabelscumd)
    # plt.legend(["2020", "2021"])
    # plt.title("Cumulative Deaths per months in " + cName)

    plt.savefig("VisFold8/AllGraph.png")
    # plt.show()





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
    if text == "WHO_region":
        indexRegion = i 

memDate = '01'
memCountry = "Afghanistan"
monthCases = 0
monthDeaths = 0

listCases = []
listDeaths = []


# Lines to show the different regions
L = []

for line in csvreader :
    if line[indexRegion] not in L :
        L.append(line[indexRegion])


f.close()
f = open("world_covid_data.csv")
csvreader = csv.reader(f, delimiter = ',')
header = next(csvreader)


CasesRegions = np.zeros((6,24))
DeathsRegions = np.zeros((6,24))


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

        # visualization(listCases, listDeaths, "Countries/" + memCountry)

        if line[indexRegion] != "Other":
            for i,k in enumerate(L) : 
                if k == line[indexRegion]:
                    index = i
                    break
            CasesRegions[i,:] += np.array(listCases)
            DeathsRegions[i,:] += np.array(listDeaths)

        monthCases = 0
        monthDeaths = 0
        memCountry = line[indexCountry]
        memDate = '01'
        listCases = []
        listDeaths = []


# for i,k in enumerate(L[:-1]) :
#     visualization(CasesRegions[i,:], DeathsRegions[i,:], "Regions/" +  k)

visualizationAll(CasesRegions, DeathsRegions, L[:-1])


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
