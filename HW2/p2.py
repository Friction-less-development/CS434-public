import numpy as np
import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#Authors: Rex Henzie, Benjamin Richards, and Michael Giovannoni
#Sources: https://matplotlib.org/users/pyplot_tutorial.html
#	https://stackoverflow.com/questions/13336823/matplotlib-python-error
#	https://matplotlib.org/tutorials/intermediate/legend_guide.html#sphx-glr-tutorials-intermediate-legend-guide-py
#	https://matplotlib.org/users/pyplot_tutorial.html
#	https://stackoverflow.com/questions/2828059/sorting-arrays-in-numpy-by-column
#	https://docs.scipy.org/doc/numpy/reference/generated/numpy.ma.size.html
#	https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.log2.html
#	https://stackoverflow.com/questions/5326112/how-to-round-each-item-in-a-list-of-floats-to-2-decimal-places

#HOW TO RUN: python p2.py

#Question 2 Decision Tree
#Part 1 Decision Stump

# avgTrainList = [] # hold the values of avg ASE for training
# avgTestList = [] # hold the values of avg ASE for test

# get matrix from knn_train
np.set_printoptions(suppress=True)
X = np.zeros(shape=(1,31)) # Matrix of data
XT = np.zeros(shape=(1,31)) # Matrix of test


print "Problem 1 of Part 2"
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

#normalize data
findMax = -1
findMin = -1
for i in range(1, 31):
	for j in range(0, np.size(X, 0)):
		if findMax == -1:
			findMax = X[j][i]
		elif findMax < X[j][i]:
			findMax = X[j][i]
		if findMin == -1:
			findMin = X[j][i]
		elif findMin > X[j][i]:
			findMin = X[j][i]

for i in range(1, 31):
	for j in range(0, np.size(X, 0)):
		X[j][i] = (X[j][i]-findMin)/float(findMax-findMin)

#global
bestStump = -1 # global, will be used to find the best overall decision stump
hS = 0.0 # H(S) value, will be the same for no matter which column it is, because the number of positives/negatives overall will not change
hSList = [] # list of h(s1), h(s2)... will be used later

# note that this is a per column basis
bestGreaterThan = 0.0
bestColumn = -1
greaterThan = 0.0 # will be used to where the cutoff is, it will go anything greater than will become "+" while evertyhing less than or equal to will become "-", as shown in slide 19
positiveList = [] # overall column positive list
negativesList = [] # overall column negative list
lBranch = [] # used to store if something is negative/positive in left branch
rBranch = [] # used to store if something is negative/positive in right branch0

rightBranch = [] # will be used to hold everything greater than greaterThan value
leftBranch = []  # will be used to hold everything less than or equal to greaterThan value
# print X
# print Y
# print "\n"
XSortTest = X[X[:,1].argsort()]
Y = XSortTest[:, 1] # getting a single column, which might be useful for doing calculations on
# print XSortTest
# print "\n"
# print Y

for i in range(0, np.size(Y)):
	if X[i][0] == 1:
		positiveList.append(float(Y[i]))
	else:
		negativesList.append(float(Y[i]))

print "Total number of positives: ", np.size(positiveList)
print "Total number of negatives: ", np.size(negativesList)
tPosONegatives = np.size(positiveList)/float(np.size(negativesList)+np.size(positiveList)) # positives/negatives
tNegOPositives = np.size(negativesList)/float(np.size(positiveList)+np.size(negativesList)) # negatives/positives
hS = -tPosONegatives*np.log2(tPosONegatives)-tNegOPositives*np.log2(tNegOPositives)
print "H(S): ", hS
#column d = 1, aka decision stump
for k in range(1, 31):
	rightBranchLabel = 0
	leftBranchLabel = 0
	del hSList[:]
	XSortTest = X[X[:,k].argsort()]
	Y = XSortTest[:, k] # getting a single column, which might be useful for doing calculations on
	for j in range(0, np.size(Y)-1):
		greaterThan = Y[j]
		for i in range(0, np.size(Y)):
			if Y[i] > greaterThan:
				rightBranch.append(Y[i])
				if X[i][0] == 1:
					rBranch.append(1)
				else:
					rBranch.append(-1)
			else:
				leftBranch.append(Y[i])
				if X[i][0] == 1:
					lBranch.append(1)
				else:
					lBranch.append(-1)
			# if X[i][0] == 1:
			# 	positiveList.append(float(Y[i]))
			# else:
			# 	negativesList.append(float(Y[i]))
		rBranchLabel = 0 # start as neutral
		lBranchLabel = 0 # start as neutral
		for i in range(0, np.size(rBranch)):
			if rBranch[i] == 1:
				rBranchLabel += 1
			else:
				rBranchLabel -= 1
		for i in range(0, np.size(lBranch)):
			if lBranch[i] == 1:
				lBranchLabel += 1
			else:
				lBranchLabel -= 1
		if rBranchLabel > 0:
			rBranchLabel = 1
		elif rBranchLabel < 0:
			rBranchLabel = -1
		if lBranchLabel > 0:
			lBranchLabel = 1
		elif lBranchLabel < 0:
			lBranchLabel = -1
		# formattedPositiveList = [ '%.2f' % elem for elem in positiveList]
		# formattedNegativeList = [ '%.2f' % elem for elem in negativesList]
		# print formattedPositiveList
		# print "Number of positives: ", np.size(positiveList)
		# print "\n", formattedNegativeList
		# print "Number of negatives: ", np.size(negativesList)
		# print "\n", greaterThan
		tempLabelBranch = -1
		if np.size(positiveList) > np.size(negativesList):
			tempLabelBranch = 1
		else:
			tempLabelBranch = 0
		if np.size(hSList) == 0:
			tempLBranchPositives = 0
			tempLBranchNegatives = 0
			for i in range(0, np.size(lBranch)):
				if lBranch[i] == 1:
					tempLBranchPositives += 1
				else:
					tempLBranchNegatives += 1
			posONegatives = tempLBranchPositives/float(tempLBranchPositives+tempLBranchNegatives) # positives/total for this branch
			negOPositives = tempLBranchNegatives/float(tempLBranchPositives+tempLBranchNegatives) # negatives/total for this branch
			# print tempBranchPositives
			# print tempBranchNegatives
			temp = -1
			if posONegatives == 0 or negOPositives == 0:
				temp = 0
			else:
				temp = -posONegatives*np.log2(posONegatives)-negOPositives*np.log2(negOPositives)
			hSList.append(temp)
			tempRBranchPositives = 0
			tempRBranchNegatives = 0
			for i in range(0, np.size(rBranch)):
				if rBranch[i] == 1:
					tempRBranchPositives += 1
				else:
					tempRBranchNegatives += 1
			posONegatives = tempRBranchPositives/float(tempRBranchPositives+tempRBranchNegatives) # positives/total for this branch
			negOPositives = tempRBranchNegatives/float(tempRBranchPositives+tempRBranchNegatives) # negatives/total for this branch
			temp = -1
			if posONegatives == 0 or negOPositives == 0:
				temp = 0
			else:
				temp = -posONegatives*np.log2(posONegatives)-negOPositives*np.log2(negOPositives)
			rightBranchLabel = rBranchLabel
			leftBranchLabel = lBranchLabel
			hSList.append(temp)
			# greaterThanPrev = greaterThan
			# print temp
			# print greaterThanPrev
			# print tempBranchPositives
			# print tempBranchNegatives
			tempNewHS = hS - (tempLBranchPositives + tempLBranchNegatives)/float(np.size(Y))*hSList[0]-(tempRBranchPositives + tempRBranchNegatives)/float(np.size(Y))*hSList[1]
			if bestStump == -1:
				bestStump = tempNewHS
				bestGreaterThan = greaterThan
				bestColumn = k
			elif tempNewHS < bestStump:
				bestStump = tempNewHS
				bestGreaterThan = greaterThan
				bestColumn = k
		elif rBranchLabel != rightBranchLabel or lBranchLabel != leftBranchLabel:
			tempLBranchPositives = 0
			tempLBranchNegatives = 0
			for i in range(0, np.size(lBranch)):
				if lBranch[i] == 1:
					tempLBranchPositives += 1
				else:
					tempLBranchNegatives += 1
			posONegatives = tempLBranchPositives/float(tempLBranchPositives+tempLBranchNegatives) # positives/total for this branch
			negOPositives = tempLBranchNegatives/float(tempLBranchPositives+tempLBranchNegatives) # negatives/total for this branch
			# print posONegatives
			# print negOPositives
			temp = -1
			if posONegatives == 0 or negOPositives == 0:
				temp = 0
			else:
				temp = -posONegatives*np.log2(posONegatives)-negOPositives*np.log2(negOPositives)
			# if temp < hSList[0]: # Will be i*2
			hSList[0] = temp
				# greaterThanPrev = greaterThan
				# print "\n"
				# print greaterThanPrev
				# print tempBranchPositives
				# print tempBranchNegatives
			tempRBranchPositives = 0
			tempRBranchNegatives = 0
			for i in range(0, np.size(rBranch)):
				if rBranch[i] == 1:
					tempRBranchPositives += 1
				else:
					tempRBranchNegatives += 1
			posONegatives = tempRBranchPositives/float(tempRBranchPositives+tempRBranchNegatives) # positives/total for this branch
			negOPositives = tempRBranchNegatives/float(tempRBranchPositives+tempRBranchNegatives) # negatives/total for this branch
			temp = -1
			if posONegatives == 0 or negOPositives == 0:
				temp = 0
			else:
				temp = -posONegatives*np.log2(posONegatives)-negOPositives*np.log2(negOPositives)
			# if temp < hSList[1]: # will be i*2+1
			hSList[1] = temp
				# greaterThanPrev = greaterThan
				# print "\n"
				# print greaterThanPrev
				# print tempBranchPositives
				# print tempBranchNegatives
			tempNewHS = hS - (tempLBranchPositives + tempLBranchNegatives)/float(np.size(Y))*hSList[0]-(tempRBranchPositives + tempRBranchNegatives)/float(np.size(Y))*hSList[1]
			if tempNewHS < bestStump:
				bestStump = tempNewHS
				bestGreaterThan = greaterThan
				bestColumn = k
				rightBranchLabel = rBranchLabel
				leftBranchLabel = lBranchLabel
				# print "changed"
				# print "newStump: " + "%0.10f" % bestStump
				# print "newGreaterThan: ", bestGreaterThan
				# print "tempLBranchPositives: ", tempLBranchPositives
				# print "tempLBranchNegatives: ", tempLBranchNegatives
				# print "tempRBranchPositives: ", tempRBranchPositives
				# print "tempRBranchNegatives: ", tempRBranchNegatives
		# print hSList
		# del hSList[:] # used to empty a list
		del rBranch[:]
		del lBranch[:]
		del rightBranch[:]
		del leftBranch[:]

print "\n"
# print hSList
print "Decision Stump: " + "%0.16f" % bestStump
print "Value used in greater than operator: ", bestGreaterThan
print "Column/feature used: ", bestColumn # note that if column 1, would be first feature column
print "Right Branch Label: ", rightBranchLabel
print "Left Branch Label: ", leftBranchLabel

leftBranchRatio = 0.0 # left branch for ratio of correct/totalRightBranch with regard to branch label
rightBranchRatio = 0.0 # right branch for ratio of correct/totalRightBranch with regard to branch label
rightBranchNumPositive = 0
rightBranchNumNegative = 0
leftBranchNumWNegative = 0
leftBranchNumPositive = 0
leftBranch = []
rightBranch = []
for i in range(0, np.size(Y)):
	if X[i][bestColumn] > bestGreaterThan:
		rightBranch.append(X[i][bestColumn])
		if X[i][0] == 1:
			rightBranchNumPositive += 1
		else:
			rightBranchNumNegative += 1
	else:
		leftBranch.append(X[i][bestColumn])
		if X[i][0] == 1:
			leftBranchNumPositive += 1
		else:
			leftBranchNumWNegative += 1

if leftBranchNumWNegative > leftBranchNumPositive:
	leftBranchRatio = leftBranchNumWNegative/float(np.size(leftBranch))*100
else:
	leftBranchRatio = leftBranchNumPositive/float(np.size(leftBranch))*100
if rightBranchNumNegative > rightBranchNumPositive:
	rightBranchRatio = rightBranchNumNegative/float(np.size(rightBranch))*100
else:
	rightBranchRatio = rightBranchNumPositive/float(np.size(rightBranch))*100

print "Left branch Training Error: ", leftBranchRatio
print "Right branch Training Error: ", rightBranchRatio
totalRatio = 0.0
totalRatio = rightBranchRatio/100.0*(np.size(rightBranch)/float(np.size(Y)))+leftBranchRatio/100.0*(np.size(leftBranch)/float(np.size(Y)))
totalRatio = totalRatio * 100
print "Decision Stump Training Error: ", totalRatio
print "\n"
# test data
del leftBranch[:]
del rightBranch[:]
leftBranchRatio = 0.0 # left branch for ratio of correct/totalRightBranch with regard to branch label
rightBranchRatio = 0.0 # right branch for ratio of correct/totalRightBranch with regard to branch label
rightBranchNumPositive = 0
rightBranchNumNegative = 0
leftBranchNumWNegative = 0
leftBranchNumPositive = 0
firstLine = True
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

#normalize data
findMax = -1
findMin = -1
for i in range(1, 31):
	for j in range(0, np.size(XT, 0)):
		if findMax == -1:
			findMax = XT[j][i]
		elif findMax < XT[j][i]:
			findMax = XT[j][i]
		if findMin == -1:
			findMin = XT[j][i]
		elif findMin > XT[j][i]:
			findMin = XT[j][i]

for i in range(1, 31):
	for j in range(0, np.size(XT, 0)):
		XT[j][i] = (XT[j][i]-findMin)/float(findMax-findMin)

for i in range(0, np.size(Y)):
	if XT[i][bestColumn] > bestGreaterThan:
		rightBranch.append(XT[i][bestColumn])
		if XT[i][0] == 1:
			rightBranchNumPositive += 1
		else:
			rightBranchNumNegative += 1
	else:
		leftBranch.append(XT[i][bestColumn])
		if XT[i][0] == 1:
			leftBranchNumPositive += 1
		else:
			leftBranchNumWNegative += 1

if leftBranchNumWNegative > leftBranchNumPositive:
	leftBranchRatio = leftBranchNumWNegative/float(np.size(leftBranch))*100
else:
	leftBranchRatio = leftBranchNumPositive/float(np.size(leftBranch))*100
if rightBranchNumNegative > rightBranchNumPositive:
	rightBranchRatio = rightBranchNumNegative/float(np.size(rightBranch))*100
else:
	rightBranchRatio = rightBranchNumPositive/float(np.size(rightBranch))*100

print "Left branch Test Error: ", leftBranchRatio
print "Right branch Test Error: ", rightBranchRatio
totalRatio = 0.0
totalRatio = rightBranchRatio/100.0*(np.size(rightBranch)/float(np.size(Y)))+leftBranchRatio/100.0*(np.size(leftBranch)/float(np.size(Y)))
totalRatio = totalRatio * 100
print "Decision Stump Test Error: ", totalRatio