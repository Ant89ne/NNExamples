import csv
import numpy as np 

initFile = "initData.csv"
outputFileinj1 = "inj1.csv"
outputFileterm = "term.csv"

f = open(initFile)
csvreader = csv.reader(f, delimiter = ',')

finj1 = open(outputFileinj1, 'w+')
csvwriterinj1 = csv.writer(finj1, delimiter = ',')

fterm = open(outputFileterm, 'w+')
csvwriterterm = csv.writer(fterm, delimiter = ',')


header = next(csvreader)
for i,k in enumerate(header) : 
    if k == "departement_residence":
        dep_index = i
    if k == "type_vaccin":
        vac_index = i
    if k == "classe_age":
        age_index = i
    if k == "date":
        date_index = i
    if k == "effectif_1_inj":
        inj1_index = i
    if k == "effectif_termine":
        term_index = i

AllInj1 = []
AllTerm = []
l = next(csvreader)
# dmem = l[date_index]
for l in csvreader : 
    
    # d = l[date_index]

    # if d != dmem and l[dep_index] != "Tout département": 
    #     break
    if l[dep_index] in ["Tout département", "2A", "2B"] or int(l[dep_index]) > 100 :
        continue
    if l[vac_index] != "Tout vaccin" or l[age_index] != "TOUT_AGE":
        continue

    try:
        inj1_data = int(l[inj1_index])
    except ValueError :
        inj1_data = 0

    try:
        term_data = int(l[term_index])
    except ValueError :
        term_data = 0

    dep = int(l[dep_index])

    while len(AllInj1) < dep:
        AllInj1.append([])
        AllTerm.append([])

    AllInj1[dep-1].append(inj1_data) 
    AllTerm[dep-1].append(term_data)

    # dmem = d

print(AllInj1)
print(AllTerm)
print(len(AllInj1)*len(AllInj1[0]))
print(len(AllInj1[0]))

for line in AllInj1 :
    csvwriterinj1.writerow(line)

for line in AllTerm :
    csvwriterterm.writerow(line)

f.close()
finj1.close()
fterm.close()













