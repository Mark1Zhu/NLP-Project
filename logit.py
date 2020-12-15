import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from scipy.sparse import vstack

from dataLoader import loadCombinedNews, splitData


def gnbTest():
    # Gaussian Naive Bayes
    gnb.fit(train.toarray(), trainLabel)
    good = 0
    bad = 0
    for el in range(len(test)):
        if gnb.predict(test[el].toarray()) == testLabel[el]:
            good += 1
        else:
            bad += 1
    print("GNB: {} GOOD AND {} BAD FOR {}% ACCURACY".format(good, bad, good / (good + bad)))

def logitTest():
    # LOGIT
    logit.fit(train, trainLabel)
    good = 0
    bad = 0
    for el in range(len(test)):
        if logit.predict(test[el]) == testLabel[el]:
            good += 1
        else:
            bad += 1
    print("LOGIT: {} GOOD AND {} BAD FOR {}% ACCURACY".format(good, bad, good / (good + bad)))

def svmTest():
    # SVC
    svm.fit(train, trainLabel)
    good = 0
    bad = 0
    for el in range(len(test)):
        if svm.predict(test[el]) == testLabel[el]:
            good += 1
        else:
            bad += 1
    print("SVM: {} GOOD AND {} BAD FOR {}% ACCURACY".format(good, bad, good / (good + bad)))

def NNTest():
    # NearestNeighbors
    neigh.fit(train, trainLabel)
    good = 0
    bad = 0
    for el in range(len(test)):
        if neigh.predict(test[el]) == testLabel[el]:
            good += 1
        else:
            bad += 1
    print("NN: {} GOOD AND {} BAD FOR {}% ACCURACY".format(good, bad, good / (good + bad)))


if __name__ == "__main__":
    #at this stagae, t and f are arrays of arrays of words (inner arrays are sentences)
    t,f = splitData(loadCombinedNews())


    #remove stopwords
    stopwords = stopwords.words('english')
    t = [[word for word in sentence if word not in stopwords] for sentence in t]
    f = [[word for word in sentence if word not in stopwords] for sentence in f]


    #training
    vectorizer = CountVectorizer()
    gnb = GaussianNB()
    logit = LogisticRegression(max_iter=1000)
    svm = SVC(kernel='rbf')
    neigh = KNeighborsClassifier(n_neighbors=5)

    allData = []
    allLabels = []
    for el in t:
        allData.append(' '.join(el))
        allLabels.append(1)
    for el in f:
        allData.append(' '.join(el))
        allLabels.append(0)
    allData = vectorizer.fit_transform(allData)
    #we now have allData as a matrix of vectors - indices 0-26617 are the positive examples and
    #indices 26618-496718 are negative (aka 0-23100 + len(t) offset)


    #testing - kinda hard coded
    #start and end are the interval for training (think of it as sliding window); everything else is testing
    start = 1000
    end = 5000 + start
    train = []
    trainLabel = []
    test = []
    testLabel = []
    for i in range(start,end): #load training examples
        train.append(allData[i])
        trainLabel.append(allLabels[i])
        train.append(allData[i+len(t)])
        trainLabel.append(allLabels[i+len(t)])
    for j in range(0,start): #load first testing examples
        test.append(allData[j])
        testLabel.append(allLabels[j])
        test.append(allData[j+len(t)])
        testLabel.append(allLabels[j+len(t)])
    for k in range(end,len(t)): #load last true testing examples
        test.append(allData[k])
        testLabel.append(allLabels[k])
    for k in range(end, len(f)): #load last false testing examples
        test.append(allData[k+len(t)])
        testLabel.append(allLabels[k+len(t)])
    train = vstack(train)