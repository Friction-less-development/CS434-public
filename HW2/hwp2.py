import numpy as np
import matplotlib as mpl
import math as math

mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



X = np.zeros(shape=(1,31)) # Matrix of data
y = np.zeros(shape=(284,1)) # Matrix of data

# get matrix from knn_train
firstLine = True
with open('knn_train.csv','r') as f:
    for line in f:
        featureNum = 0
        lineWords = []
        averageValue = []
        for word in line.split(','):
           if featureNum >= 0:
               lineWords.append(float(word))
               featureNum += 1
        if firstLine:
            X[0] = lineWords
            firstLine = False
        else:
            X = np.vstack((X, lineWords))

# make a new vector of y values
yTrain = X[:,0]

#cut out the y values
XTrain = np.delete(X, 0, 1)        
            
#normalize data between 0 and 1
XTrain = (XTrain - XTrain.min(0)) / XTrain.ptp(0)

print("x shape: " , XTrain.shape)
print("y shape: " , yTrain.shape)

# tree class.
class Tree(object):
    def __init__(self):
        self.xLeft = None
        self.xRight = None
        self.yLeft = None
        self.yRight = None        
        # index of the feature for this node
        self.feature = None        
        # threshold for the feature
        self.threshold = None
        self.informationGain = None
        self.entropy = None 

# data is our dataset (X)
# index is the index of the feature we're splitting on
# threshold is the threshold value
# returns XleftSplit, XrightSplit, yLeftSplit, yLeftSplit 
def split(Xset, yset, index, threshold):
    XleftSplit, XrightSplit, yLeftSplit, yLeftSplit = list(), list(), list(), list()
    i = 0
    for row in Xset:
        if row[index] <= threshold:
            XleftSplit.append(i)
            yLeftSplit.append(i)
        if row[index] > threshold:
            XrightSplit.append(i)
            yLeftSplit.append(i)
        i = i + 1
    return XleftSplit, XrightSplit, yLeftSplit, yLeftSplit

def findSplit(Xset, yset):
    
    PSEUDO CODE
    
    xlength = length of Xset (num of samples)
    ylength = length of yset (num of features)
    
    maxIgain = 0
    
    for j in range(ylength):
        for i in range(xlength):
            current = Xset[i]
            threshold = current[j]
            
            XleftSplit, XrightSplit = split(X, y, feature_idx, threshold)
            cost = cost(XleftSplit, XrightSplit)
            if cost > maxIgain:
                maxIgain = cost
                bestSplit = XleftSplit, XrightSplit
                featureIndex = j
                finalThreshold = threshold
    return . . .

    
XleftSplit, XrightSplit, yLeftSplit, yLeftSplit = split(XTrain, 10, .1)
print(len(XleftSplit))
print(len(XrightSplit))
            