#!/usr/bin/env python
import pandas as pd
import math
from sys import argv


# caculate the L2 distance of training data and testing data, only works on real number
def distance(a, b):
    sum = 0.0
    list = []
    # walk through the whole dataset
    for i in range(lr):
        for j in range(lc - 1):
            # check if the data is real value
            # when it is not real value, just check whether two element are the same
            if(isinstance(a.iloc[i][j], str ) == True):
                if(a.iloc[i][j]!= b[j]):
                    sum = sum + 1.0
                else:
                    sum = sum + 0.0
            else:
                # when it is a real value caculate the square difference
                diff = math.pow((a.iloc[i][j] - b[j]), 2)
                sum = sum + diff
        list.append(math.sqrt(sum))
        sum = 0
    # list contains all the row L2 distance and each row is a data point
    return list


# generate the predict label
def predict(listIndex, training_data):
    for i in range(len(listIndex)):
        label = training_data.iloc[listIndex[i]][lc-1]
        labels.append(label)
    return max(set(labels), key=labels.count)


script, K, training, testing = argv

training_data = pd.read_csv(training, header=None)
testing_data = pd.read_csv(testing, header=None)
# get the length of index and the length of column
lr, lc = training_data.shape
lrt, lct = testing_data.shape

neighborsIndex = []
labels = []
# remove the label column of testing data
testing_data[lct] = testing_data[lct-1]
a = 0

for index, row in testing_data.iterrows():
    neighborsDistance = distance(training_data,row)
    sortedIndex = sorted(range(len(neighborsDistance)), key=lambda k: neighborsDistance[k])
    for i in range(int(K)):
        neighborsIndex.append(sortedIndex[i])

    p = predict(neighborsIndex, training_data)
    testing_data.loc[a, lct] = p
    a = a + 1
print(testing_data)
testing_data.to_csv('testing', header=False, index=False)
