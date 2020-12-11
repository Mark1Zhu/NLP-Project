import csv

def loadCombinedNews():
    with open('data/Combined_News_DJIA.csv', "rt", newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
    for row in data[1:]: #all except first row
        colNum = 0
        for col in row:
            if row[colNum][0] == "b" and colNum > 1:
                #some sentences start with b' or b" and end with apostrophe/quotation
                row[colNum] = row[colNum][2:-1]
            colNum+=1
    return data[1:]

def splitData(data): #given array, splits into 0 and 1 arrays
    trueArray = []
    falseArray = []
    for el in data:
        if el[1] == '0':
            [falseArray.append(sentence.split()) for sentence in el[2:]]
        elif el[1] == '1':
            [trueArray.append(sentence.split()) for sentence in el[2:]]
        else:
            print("?")
    return trueArray,falseArray

if __name__ == "__main__":
    t,f = splitData(loadCombinedNews())