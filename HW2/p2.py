#Authors: Rex Henzie, Benjamin Richards, and Michael Giovannoni

#HOW TO RUN: python2.7 p2.py
#Folder must contain usps-4-9-train.csv and usps-4-9-test.csv data files

#This program was modelled off of Jason Brownlee's tutorial found at https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/

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

# tree class.
class Tree(object):
    def __init__(self):
        self.xLeft = None
        self.xRight = None       
        # index of the feature for this node
        self.feature = None        
        # threshold for the feature
        self.threshold = None
        self.informationGain = None
        self.entropy = None
        self.rightChild = None
        self.leftChild = None
        self.classification = None
        
        
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

    if (numLeftPositives == 0 and numRightPositives == 0):
    	hS = -(numLeftNegatives+numRightNegatives)/(float(np.size(XleftSplit)+np.size(XrightSplit)))*np.log2((numLeftNegatives+numRightNegatives)/(float(np.size(XleftSplit)+np.size(XrightSplit))))
    elif (numLeftNegatives == 0 and numRightNegatives == 0):
    	hS = -(numLeftPositives+numRightPositives)/(float(np.size(XleftSplit)+np.size(XrightSplit)))*np.log2((numLeftPositives+numRightPositives)/(float(np.size(XleftSplit)+np.size(XrightSplit))))
    else:
    	hS = -(numLeftPositives+numRightPositives)/(float(np.size(XleftSplit)+np.size(XrightSplit)))*np.log2((numLeftPositives+numRightPositives)/(float(np.size(XleftSplit)+np.size(XrightSplit))))-(numLeftNegatives+numRightNegatives)/(float(np.size(XleftSplit)+np.size(XrightSplit)))*np.log2((numLeftNegatives+numRightNegatives)/(float(np.size(XleftSplit)+np.size(XrightSplit))))


    return (hS - (numLeftPositives+numLeftNegatives)/(float(np.size(XleftSplit)+np.size(XrightSplit)))*hSList[0]-(numRightPositives+numRightNegatives)/(float(np.size(XleftSplit)+np.size(XrightSplit)))*hSList[1])

# split a given set based of the provided threshold and feature
# Xset is our dataset
# index is the index of the feature we're splitting on
# threshold is the threshold value
# returns XleftSplit, XrightSplit
def split(Xset, index, threshold):
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

# find the optimum split 
def findSplit(Xset):    
    numSamples, numFeatures = XTrain.shape 
    numSamples = len(Xset)
    
    Xset = XTrain[Xset]
    
    maxIgain = 0.0
    featureIndex = -1
    finalThreshold = -1
    bestXleftSplit = []
    bestXrightSplit = []
    for feature in range(numFeatures):
        for sample in range(numSamples):
            current = Xset[sample]
            threshold = current[feature]
            XleftSplit, XrightSplit = split(Xset, feature, threshold)
            igain = infoGain(XleftSplit, XrightSplit)
            if igain > maxIgain:
                maxIgain = igain
                bestXleftSplit = XleftSplit 
                bestXrightSplit = XrightSplit
                featureIndex = feature
                finalThreshold = threshold
    
    node = Tree()
    
    node.xLeft = bestXleftSplit
    node.xRight = bestXrightSplit   
    node.feature = featureIndex
    node.informationGain = maxIgain
    node.threshold = finalThreshold
    
    
    return node

# recursively create the decision tree
def createTree(currentDepth, maxDepth, currentNode):
    xLeft = currentNode.xLeft
    xRight = currentNode.xRight
    
    # class values
    yLeft = yTrain[xLeft]
    yRight = yTrain[xRight]
    
    # left and right child subtrees
    leftChildNode = Tree()
    rightChildNode = Tree()
    
    currentNode.leftChild = leftChildNode
    currentNode.rightChild = rightChildNode 

    
    # if there are 0 samples in one of our splits then its a leaf node and we set the classification
    if len(xLeft) == 0 or len(xRight) == 0:
        yBoth = np.append(yLeft, yRight)        
        
        # most common classification
        numNegativeOnes = (yBoth == -1).sum()
        numOnes = (yBoth == 1).sum()
        if(numNegativeOnes >= numOnes):
            currentNode.leftChild.classification = -1
            currentNode.rightChild.classification = -1
        else:
            currentNode.leftChild.classification = 1
            currentNode.rightChild.classification = 1
        
        return None
    
    
    # if we've reached our max depth create left and right leaf nodes
    if currentDepth >= maxDepth:
        # set classification to most frequent
        numNegativeOnes = (yLeft == -1).sum()
        numOnes = (yLeft == 1).sum()
        
        if(numNegativeOnes >= numOnes):
            currentNode.leftChild.classification = -1
        else:
            currentNode.leftChild.classification = 1
        
        numNegativeOnes = (yRight == -1).sum()
        numOnes = (yRight == 1).sum()
        
        if(numNegativeOnes >= numOnes):
            currentNode.rightChild.classification = -1
        else:
            currentNode.rightChild.classification = 1

        return None
    
    
    # if only a single sample left don't split node
    if len(xLeft) <= 1:
        # set classification to most frequent
        numNegativeOnes = (yLeft == -1).sum()
        numOnes = (yLeft == 1).sum()
        
        if(numNegativeOnes >= numOnes):
            currentNode.leftChild.classification = -1
        else:
            currentNode.leftChild.classification = 1        
        
    # else find split and recursively call createTree
    else:
        currentNode.leftChild = findSplit(xLeft)
        createTree(1+currentDepth, maxDepth, currentNode.leftChild)
    
    
    
    # if only a single sample left don't split node
    if len(xRight) <= 1:
        numNegativeOnes = (yRight == -1).sum()
        numOnes = (yRight == 1).sum()
        
        if(numNegativeOnes >= numOnes):
            currentNode.rightChild.classification = -1
        else:
            currentNode.rightChild.classification = 1
        
    # else find split and recursively call createTree
    else:
        currentNode.rightChild = findSplit(xRight)
        createTree(1+currentDepth, maxDepth, currentNode.rightChild)

    return currentNode

def calcAccuracyTrain(treeNode): # need threshold and feature for node
    firstLine = True
    XT = np.zeros(shape=(1,31)) # Matrix of data
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
                XT[0] = lineWords
                firstLine = False
            else:
                XT = np.vstack((XT, lineWords))
    leftBranchRatio = 0.0 # left branch for ratio of correct/totalRightBranch with regard to branch label
    rightBranchRatio = 0.0 # right branch for ratio of correct/totalRightBranch with regard to branch label
    rightBranch = []
    leftBranch = []
    rightBranchNumPositive = 0
    rightBranchNumNegative = 0
    leftBranchNumWNegative = 0
    leftBranchNumPositive = 0
    for i in range(0, np.size(XT, 0)):
        if XT[i][treeNode.feature] > treeNode.threshold:
            rightBranch.append(XT[i][treeNode.feature])
            if XT[i][0] == 1:
                rightBranchNumPositive += 1
            else:
                rightBranchNumNegative += 1
        else:
            leftBranch.append(XT[i][treeNode.feature])
            if XT[i][0] == 1:
                leftBranchNumPositive += 1
            else:
                leftBranchNumWNegative += 1
    if np.size(leftBranch) == 0:
        leftBranchRatio = 0
    elif leftBranchNumWNegative > leftBranchNumPositive:
        leftBranchRatio = leftBranchNumWNegative/float(np.size(leftBranch))*100
    else:
        leftBranchRatio = leftBranchNumPositive/float(np.size(leftBranch))*100
    if np.size(rightBranch) == 0:
        rightBranchRatio = 0
    elif rightBranchNumNegative > rightBranchNumPositive:
        rightBranchRatio = rightBranchNumNegative/float(np.size(rightBranch))*100
    else:
        rightBranchRatio = rightBranchNumPositive/float(np.size(rightBranch))*100
    totalRatio = 0.0
    totalRatio = rightBranchRatio/100.0*(np.size(rightBranch)/float(np.size(XT, 0)))+leftBranchRatio/100.0*(np.size(leftBranch)/float(np.size(XT, 0)))
    totalRatio = totalRatio * 100
    return totalRatio

def numNodes(treeNode, totalNodes):
	if treeNode.classification != 1 and treeNode.classification != 1 and treeNode.feature != None:
		totalNodes += 1
		totalNodes = numNodes(treeNode.leftChild, totalNodes)
		totalNodes = numNodes(treeNode.rightChild, totalNodes)
		return totalNodes
	else:
		return totalNodes

def totalTreeCalcTrain(treeNode): # start with root node
	totalSumRatio = 0.0
	if treeNode.classification != 1 and treeNode.classification != 1 and treeNode.feature != None:
		totalSumRatio += calcAccuracyTrain(treeNode)
		totalSumRatio += totalTreeCalcTrain(treeNode.leftChild)
		totalSumRatio += totalTreeCalcTrain(treeNode.rightChild)
	return totalSumRatio


def calcAccuracyTest(treeNode): # need threshold and feature for node
    firstLine = True
    XT = np.zeros(shape=(1,31)) # Matrix of data
    with open('knn_test.csv','r') as f:
        for line in f:
            featureNum = 0
            lineWords = []
            averageValue = []
            for word in line.split(','):
               if featureNum >= 0:
                   lineWords.append(float(word))
                   featureNum += 1
            if firstLine:
                XT[0] = lineWords
                firstLine = False
            else:
                XT = np.vstack((XT, lineWords))
    leftBranchRatio = 0.0 # left branch for ratio of correct/totalRightBranch with regard to branch label
    rightBranchRatio = 0.0 # right branch for ratio of correct/totalRightBranch with regard to branch label
    rightBranch = []
    leftBranch = []
    rightBranchNumPositive = 0
    rightBranchNumNegative = 0
    leftBranchNumWNegative = 0
    leftBranchNumPositive = 0
    for i in range(0, np.size(XT, 0)):
        if XT[i][treeNode.feature] > treeNode.threshold:
            rightBranch.append(XT[i][treeNode.feature])
            if XT[i][0] == 1:
                rightBranchNumPositive += 1
            else:
                rightBranchNumNegative += 1
        else:
            leftBranch.append(XT[i][treeNode.feature])
            if XT[i][0] == 1:
                leftBranchNumPositive += 1
            else:
                leftBranchNumWNegative += 1
    if np.size(leftBranch) == 0:
        leftBranchRatio = 0
    elif leftBranchNumWNegative > leftBranchNumPositive:
        leftBranchRatio = leftBranchNumWNegative/float(np.size(leftBranch))*100
    else:
        leftBranchRatio = leftBranchNumPositive/float(np.size(leftBranch))*100
    if np.size(rightBranch) == 0:
        rightBranchRatio = 0
    elif rightBranchNumNegative > rightBranchNumPositive:
        rightBranchRatio = rightBranchNumNegative/float(np.size(rightBranch))*100
    else:
        rightBranchRatio = rightBranchNumPositive/float(np.size(rightBranch))*100
    totalRatio = 0.0
    totalRatio = rightBranchRatio/100.0*(np.size(rightBranch)/float(np.size(XT, 0)))+leftBranchRatio/100.0*(np.size(leftBranch)/float(np.size(XT, 0)))
    totalRatio = totalRatio * 100
    return totalRatio

def totalTreeCalcTest(treeNode): # start with root node
	totalSumRatio = 0.0
	if treeNode.classification != 1 and treeNode.classification != 1 and treeNode.feature != None:
		totalSumRatio += calcAccuracyTest(treeNode)
		totalSumRatio += totalTreeCalcTest(treeNode.leftChild)
		totalSumRatio += totalTreeCalcTest(treeNode.rightChild)
	return totalSumRatio

    
rootNode = Tree()

xNumberOfSamples, xNumberOfFeatures = XTrain.shape
# to find root call findSplit with a list that contains the indices of every sample in the training set
xStartingSet = range(xNumberOfSamples)

rootNode = findSplit(xStartingSet)


print("Root node information: ")
print "\n"
print(rootNode.feature)
print(rootNode.threshold)
print(rootNode.informationGain)

print "d = 1"
print "Training: "
print calcAccuracyTrain(rootNode)
print "Test: "
print calcAccuracyTest(rootNode)
print "\n"

totalTrainList = []
totalTestList = []
totalTrainList.append(calcAccuracyTrain(rootNode))
totalTestList.append(calcAccuracyTest(rootNode))

for i in range(2, 7):
	decisionTree = Tree()
	decisionTree = createTree(1, i, rootNode)
	print "d = ", i
	# print numNodes(decisionTree, 0)
	print "Training: "
	totalTrainList.append(totalTreeCalcTrain(decisionTree)/(float(numNodes(decisionTree, 0))))
	totalTestList.append(totalTreeCalcTest(decisionTree)/(float(numNodes(decisionTree, 0))))
	print totalTreeCalcTrain(decisionTree)/(float(numNodes(decisionTree, 0)))
	print "Test: "
	print totalTreeCalcTest(decisionTree)/(float(numNodes(decisionTree, 0)))

	print "\n"

plt.figure(1)
plt.ylabel('Error Rates')
plt.xlabel('D Features')
xAxis = [1, 2, 3, 4, 5, 6]
plt.title('Number of Features VS Error Rate')
plt.plot(xAxis, totalTrainList, 'ro', label='Training Data')
plt.plot(xAxis, totalTestList, 'bo', label='Test Data')

plt.axis([0,8, 50, 85])

plt.legend()
plt.savefig('p2p2.png')