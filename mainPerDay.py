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


def testDay(model): #test per headline
    model.fit(train.toarray(), trainLabel)
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
        labels.append(day[1])
        dayBOW = ""
        for sentenceNum in range(2, len(day)): #don't bother with the date/label
            dayBOW += ' '.join(day[sentenceNum])
        dataVec.append(dayBOW)
    dataVec = vectorizer.fit_transform(dataVec)


    #testing - kinda hard coded
    #start/end are the day interval for training (think of it as sliding window); everything else is testing
    start = 0
    end = 100 + start
    train = []
    trainLabel = []
    test = []
    testLabel = []
    for i in range(start,end): #load training examples
        train.append(dataVec[i])
        trainLabel.append(labels[i]) #training heuristic: all sentences in training are the day's label
    train = vstack(train)
    for j in range(0,start):
        test.append(dataVec[j])
        testLabel.append(labels[j])
    for k in range(end,len(labels)):
        test.append(dataVec[k])
        testLabel.append(labels[k])