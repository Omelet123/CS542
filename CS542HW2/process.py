#!/usr/bin/env python
import pandas as pd
import numpy as np
from math import floor
from sys import argv


def preprocess(data):
    # replace all the data with NaN so that we can use fillna()
    data = data.replace('?', np.NaN)
    # for the first column we replace '?' with 'b'
    data[0] = data[0].fillna('b')

    tomedian = [3, 4, 5, 6, 8, 9, 11, 12]
    tomean = [1, 2, 7, 10, 13, 14]
    # replacing missing data by its median
    for i in tomedian:
        data.sort_values(by=[i])
        n =floor(data[i].count()/2)
        m = data.loc[n, i]
        data[i] = data[i].fillna(m)

    # replacing missing data by its mean
    labels = ['+', '-']
    for i in tomean:
        data[i] = data[i].apply(float)
        for c in labels:
            data.loc[(data[i].isnull()) & (data[15] == c), i] = data[i][data[15] == c].mean()
    # normalize with z scaling
    for i in tomean:
        data[i] = (data[i] - data[i].mean()) / data[i].std()
    return data



script, training, testing = argv
trainingData = pd.read_csv(training, header=None)
trainingData = preprocess(trainingData)
#print(trainingData)
trainingData.to_csv('crx.training.processed', header=False, index=False)

testingData = pd.read_csv(testing, header=None)
testingData = preprocess(testingData)
#print(testingData)
testingData.to_csv('crx.testing.processed', header=False, index=False)
