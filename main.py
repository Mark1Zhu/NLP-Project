import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import vstack

from dataLoader import loadCombinedNews, splitData


def testHeadline(model, daysBack, arrFlag): #test per headline, daysBack is how many days to look backwards
    def findMostCommonPrediction(x,y,good,bad):
        for dayNum in range(x+daysBack, y): #per day
            posSentence = 0
            negSentence = 0
            for days in range(daysBack+1):
                for sentence in dataVecByDay[dayNum-days]: #for each sentence in that day
                    if arrFlag == 1:
                        sentencePredict = model.predict(sentence.toarray())[0]
                    else:
                        sentencePredict = model.predict(sentence)[0]
                    if sentencePredict == '1':
                        posSentence += 1
                    elif sentencePredict == '0':
                        negSentence += 1
                    else:
                        print(sentencePredict)
            if posSentence > negSentence: #generate prediction per sentence, and choose most common prediction per day
                dayPredict = '1'
            elif posSentence < negSentence:
                dayPredict = '0'
            else:
                bad += 1
                continue

            if dayPredict == labels[dayNum]: #compare our most common prediction per day with day's actual label
                good += 1
            else:
                bad += 1
        return good,bad
        #end of helper subfunction

    if arrFlag == 1:
        model.fit(train.toarray(), trainLabel)
    else:
        model.fit(train, trainLabel)
    good,bad = findMostCommonPrediction(0, start, 0, 0)
    good,bad = findMostCommonPrediction(end, len(dataVecByDay)-400, good, bad)
    tgood,tbad = findMostCommonPrediction(len(dataVecByDay)-400,len(dataVecByDay),0,0)

    print("{}: {} GOOD AND {} BAD FOR {}% ACCURACY".format(model, good, bad, good / (good + bad)))
    print("TESTING: {} GOOD AND {} BAD FOR {}% ACCURACY".format(tgood,tbad, tgood/(tgood+tbad)))

def testDay(model): #test per day
    #train = dataVecByDay[start:end]
    train = [vstack(day) for day in dataVecByDay[start:end]]
    trainLabel = labels[start:end]
    test = dataVecByDay[0:start] + dataVecByDay[end:len(dataVecByDay)-1]
    testLabel = labels[0:start] + labels[end:len(labels)-1]

    model.fit(train, trainLabel)
    good = 0
    bad = 0

    for i in range(len(test)):
        if model.predict(test[i]) == testLabel[i]:
            good += 1
        else:
            bad += 1

    print("{} DAY: {} GOOD AND {} BAD FOR {}% ACCURACY".format(model, good, bad, good / (good + bad)))

if __name__ == "__main__":
    #at this stage, data is arrays of arrays of arrays of words
    #innermost arrays are words in sentence
    #then grouped into sentences in dates
    #then dates grouped up into whether day was 0 or 1 - indices 0 and 1 are for date and label
    data = loadCombinedNews()


    #remove stopwords & stem
    stopwords = stopwords.words('english')
    stemmer = SnowballStemmer("english")
    for day in data:
        for sentenceNum in range(2, len(day)): #don't bother with the date/label
            day[sentenceNum] = [stemmer.stem(word) for word in day[sentenceNum] if word not in stopwords]

    #training
    vectorizer = CountVectorizer()
    gnb = GaussianNB()
    logit = LogisticRegression(max_iter=1000)
    svm = SVC(kernel='rbf')
    knn = KNeighborsClassifier(n_neighbors=5)

    dataVec = []
    labels = []
    indexer = [] #this will hold the count of headlines per day
    for day in data:
        counter = 0
        labels.append(day[1])
        for sentenceNum in range(2, len(day)): #don't bother with the date/label
            dataVec.append(' '.join(day[sentenceNum]))
            counter += 1
        indexer.append(counter)
    dataVec = vectorizer.fit_transform(dataVec)

    counter = 0
    dataVecByDay = []
    # combine each group of vectors back into the day, using the headline count from indexer
    for numSentence in indexer:
        dataVecByDay.append([dataVec[i+counter] for i in range(numSentence)])
        counter += numSentence

    #testing - kinda hard coded
    #start/end are the day interval for training (think of it as sliding window); everything else is testing
    start = 0
    end = 100 + start
    train = []
    trainLabel = []
    for i in range(start,end): #load training examples
        for sentenceVec in dataVecByDay[i]:
            train.append(sentenceVec)
            trainLabel.append(labels[i]) #training heuristic: all sentences in training are the day's label
    train = vstack(train)