import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from scipy.sparse import vstack

from dataLoader import loadCombinedNews, splitData


t,f = splitData(loadCombinedNews())

#remove stopwords
stopwords = stopwords.words('english')
t = [[word for word in sentence if word not in stopwords] for sentence in t]
f = [[word for word in sentence if word not in stopwords] for sentence in f]

#training
vectorizer = CountVectorizer()
logit = LogisticRegression(max_iter=1000)

allData = []
allLabels = []
for el in t:
    allData.append(' '.join(el))
    allLabels.append(1)
for el in f:
    allData.append(' '.join(el))
    allLabels.append(0)
allData = vectorizer.fit_transform(allData)

#everything after here needs to be fixed/generalized lul
train = []
trainLabel = []
test = []
testLabel = []
for i in range(20000):
    train.append(allData[i])
    trainLabel.append(allLabels[i])
    train.append(allData[i+len(t)])
    trainLabel.append(allLabels[i+len(t)])
for j in range(6618):
    test.append(allData[j+20000])
    testLabel.append(allLabels[j+20000])
for k in range(3100):
    test.append(allData[k+46618])
    testLabel.append(allLabels[k+46618])

train = vstack(train)
logit.fit(train,trainLabel)
good = 0
bad = 0
for m in range(6618):
    if logit.predict(allData[m+20000]) == testLabel[m]:
        good += 1
    else:
        bad += 1
for n in range(3100):
    if logit.predict(allData[n+46618]) == testLabel[n+6618]:
        good += 1
    else:
        bad += 1