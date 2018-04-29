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
#         self.yLeft = None
#         self.yRight = None        
        # index of the feature for this node
        self.feature = None        
        # threshold for the feature
        self.threshold = None
        self.informationGain = None
        self.entropy = None 
        
        
def infoGain(XleftSplit, XrightSplit): # need to store what you find in a list or something, once it become "permeanent/chosen" branch/node
    hS = -1.0 # will calculate below
    hSList = []
    temp = -1
    posNegativeRatio = 0
    negPositiveRatio = 0
    numLeftPositives = 0
    numLeftNegatives = 0
    if np.size(XleftSplit) > 0:
        for i in range(0, np.size(XleftSplit)):
            if yTrain[XleftSplit[i]] == 1:
                numLeftPositives += 1
            elif yTrain[XleftSplit[i]] == -1:
                numLeftNegatives += 1
            else:
                print "Shouldn't get here left side"
        if np.size(XleftSplit) > 0:
            posNegativeRatio = numLeftPositives/(float(numLeftPositives + numLeftNegatives))
            negPositiveRatio = numLeftNegatives/(float(numLeftPositives + numLeftNegatives))
    if posNegativeRatio == 0 or negPositiveRatio == 0:
        temp = 0
    else:
        temp = -posNegativeRatio*np.log2(posNegativeRatio)-negPositiveRatio*np.log2(negPositiveRatio)
    hSList.append(temp)
    temp = -1
    numRightPositives = 0
    numRightNegatives = 0
    if np.size(XrightSplit) > 0:
        
        for i in range(0, np.size(XrightSplit)):
            if yTrain[XrightSplit[i]] == 1:
                numRightPositives += 1
            elif yTrain[XrightSplit[i]] == -1:
                numRightNegatives += 1
            else:
                print "Shouldn't get here right side"
        if np.size(XrightSplit) > 0:
            posNegativeRatio = numRightPositives/(float(numRightPositives + numRightNegatives))
            negPositiveRatio = numRightNegatives/(float(numRightPositives + numRightNegatives))
    if posNegativeRatio == 0 or negPositiveRatio == 0:
        temp = 0
    else:
        temp = -posNegativeRatio*np.log2(posNegativeRatio)-negPositiveRatio*np.log2(negPositiveRatio)
    hSList.append(temp)
    hS = -(numLeftPositives+numRightPositives)/(float(np.size(XleftSplit)+np.size(XrightSplit)))*np.log2((numLeftPositives+numRightPositives)/(float(np.size(XleftSplit)+np.size(XrightSplit))))-(numLeftNegatives+numRightNegatives)/(float(np.size(XleftSplit)+np.size(XrightSplit)))*np.log2((numLeftNegatives+numRightNegatives)/(float(np.size(XleftSplit)+np.size(XrightSplit))))
    return hS - (numLeftPositives+numLeftNegatives)/(float(np.size(XleftSplit)+np.size(XrightSplit)))-(numRightPositives+numRightNegatives)/(float(np.size(XleftSplit)+np.size(XrightSplit)))

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
    return XleftSplit, XrightSplit

def findSplit(Xset, yset):

    
    numSamples, numFeatures = Xset.shape 
    print(Xset.shape)
    print("numfeatures: ", numFeatures)
    
    maxIgain = 0
    bestXleftSplit = -1 
    bestXrightSplit = -1
    featureIndex = -1
    finalThreshold = -1
    
    for feature in range(numFeatures):
        for sample in range(numSamples):
            current = Xset[sample]
            threshold = current[feature]
            XleftSplit, XrightSplit = split(Xset, yset, feature, threshold)
            igain = infoGain(XleftSplit, XrightSplit)
#             print("igain: ", igain)
#             print("feature: ", feature)
#             print("sample: ", sample)
            if igain > maxIgain:
                maxIgain = igain
                bestXleftSplit = XleftSplit 
                bestXrightSplit = XrightSplit
                featureIndex = feature
                finalThreshold = threshold
    
    node = Tree()

    node.xLeft = bestXleftSplit
    node.xRight = bestXRightSplit
    node.feature = featureIndex
    node.informationGain = maxIgain
    node.threshold = finalThreshold
    
    
    return node

def createTree():
    
    return test


    
rootNode = Tree()
rootNode = findSplit(XTrain, yTrain)

print(rootNode.feature)
print(rootNode.threshold)
print(rootNode.informationGain)
    
# XleftSplit, XrightSplit, yLeftSplit, yLeftSplit = split(XTrain, 10, .1)
# print(len(XleftSplit))
# print(len(XrightSplit))
            
    