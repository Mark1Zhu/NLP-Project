import csv

def loadCombinedNews():
    with open('data/Combined_News_DJIA.csv', "rt", newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    for row in data[1:]: #all except first row
        for colNum in range(2,len(row)):
            if row[colNum][0] == "b":
                #some sentences start with b' or b" and end with apostrophe/quotation
                row[colNum] = row[colNum][2:-1]
            row[colNum] = row[colNum].lower().split()
    return data[1:]

def splitData(data): #given array, splits into 0 and 1 arrays
    trueArray = []
    falseArray = []
    for el in data:
        if el[1] == '0':
            perDate = [sentence for sentence in el[2:]]
            falseArray.append(perDate)
        elif el[1] == '1':
            perDate = [sentence for sentence in el[2:]]
            trueArray.append(perDate)
        else:
            print("?")
    return trueArray,falseArray

if __name__ == "__main__":
    a = loadCombinedNews()
    t,f = splitData(loadCombinedNews())