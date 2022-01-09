import numpy as np
import matplotlib.pyplot as plt
import csv

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
        monthCases = 0
        monthDeaths = 0
        break

labels = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
labelsy = [k for k in range(0, np.round(np.max(listCases), -4)+1, 5000)]
labelsyd = [k for k in range(0, np.round(np.max(listDeaths), -2)+1, 100)]

print(np.round(np.max(listCases), decimals = -5))

print(len(listCases))

plt.figure(0, figsize=(30,30))
plt.subplot(121)
plt.plot([k for k in range(len(labels))], listCases[:len(labels)], 'b*-')
plt.plot([k for k in range(len(labels))], listCases[len(labels):], 'r*-')
plt.xticks([k for k in range(len(labels))], labels, rotation=45)
plt.yticks(labelsy, labelsy)

plt.legend(["2020", "2021"])
plt.title("Cases per months")

plt.subplot(122)
plt.plot([k for k in range(len(labels))], listDeaths[:len(labels)], 'b*-')
plt.plot([k for k in range(len(labels))], listDeaths[len(labels):], 'r*-')
plt.xticks([k for k in range(len(labels))], labels, rotation=45)
plt.yticks(labelsyd, labelsyd)
plt.legend(["2020", "2021"])
plt.title("Deaths per months")
plt.savefig("AfghData.png")
plt.show()
