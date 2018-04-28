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

print "Part 1"
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

#global
bestStump = 0.0 # global, will be used to find the best overall decision stump
hS = 0.0 # H(S) value, will be the same for no matter which column it is, because the number of positives/negatives overall will not change
hSList = [] # list of h(s1), h(s2)... will be used later

# note that this is a per column basis
greaterThan = 0.0 # will be used to where the cutoff is, it will go anything greater than will become "+" while evertyhing less than or equal to will become "-", as shown in slide 19
positiveList = [] # overall column positive list
negativesList = [] # overall column negative list
lBranch = [] # used to store if something is negative/positive in left branch
rBranch = [] # used to store if something is negative/positive in right branch
rightBranch = [] # will be used to hold everything greater than greaterThan value
leftBranch = []  # will be used to hold everything less than or equal to greaterThan value
posNumCount = -1
negNumCount = -1
labelBranch = -1
print X
# print Y
print "\n"
XSortTest = X[X[:,1].argsort()]
Y = XSortTest[:, 1] # getting a single column, which might be useful for doing calculations on
print XSortTest
print "\n"
print Y

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
#column 1
# for j in range(0, np.size(Y)-1):
# 	greaterThan = Y[j]
# 	for i in range(0, np.size(Y)):
# 		if Y[i] > greaterThan:
# 			rightBranch.append(Y[i])
# 			if X[i][0] == 1:
# 				rBranch.append(1)
# 			else:
# 				rBranch.append(-1)
# 		else:
# 			leftBranch.append(Y[i])
# 			if X[i][0] == 1:
# 				lBranch.append(1)
# 			else:
# 				lBranch.append(-1)
# 		# if X[i][0] == 1:
# 		# 	positiveList.append(float(Y[i]))
# 		# else:
# 		# 	negativesList.append(float(Y[i]))

# 	# formattedPositiveList = [ '%.2f' % elem for elem in positiveList]
# 	# formattedNegativeList = [ '%.2f' % elem for elem in negativesList]
# 	# print formattedPositiveList
# 	# print "Number of positives: ", np.size(positiveList)
# 	# print "\n", formattedNegativeList
# 	# print "Number of negatives: ", np.size(negativesList)
# 	# print "\n", greaterThan
# 	tempLabelBranch = -1
# 	if np.size(positiveList) > np.size(negativesList):
# 		tempLabelBranch = 1
# 	else:
# 		tempLabelBranch = 0
# 	if np.size(hSList) == 0:
# 		posONegatives = np.size(positiveList)/float(np.size(negativesList)+np.size(positiveList)) # positives/negatives
# 		negOPositives = np.size(negativesList)/float(np.size(positiveList)+np.size(negativesList)) # negatives/positives
# 		temp = -posONegatives*np.log2(posONegatives)-negOPositives*np.log2(negOPositives)
# 		posNumCount = np.size(positiveList)
# 		negNumCount = np.size(negativesList)
# 		if(posNumCount > negNumCount):
# 			labelBranch = 1
# 		else:
# 			labelBranch = 0
# 		hSList.append(temp)
# 	elif tempLabelBranch != labelBranch:
# 		posONegatives = np.size(positiveList)/float(np.size(negativesList)+np.size(positiveList)) # positives/negatives
# 		negOPositives = np.size(negativesList)/float(np.size(positiveList)+np.size(negativesList)) # negatives/positives
# 		temp = -posONegatives*np.log2(posONegatives)-negOPositives*np.log2(negOPositives)
# 		if temp < hSList[0]: # 0 would i, in some loop for each column/feature
# 			hSList[0] = temp

# 	# print hSList
# 	# del hSList[:] # used to empty a list
# 	# del negativesList[:]
# 	# del positiveList[:]

# print hSList

# greaterThan = Y[1]
# for i in range(0, np.size(Y)):
# 	if Y[i] > greaterThan:
# 		positiveList.append(Y[i])
# 	else:
# 		negativesList.append(Y[i])

# # formattedPositiveList = [ '%.2f' % elem for elem in positiveList]
# # formattedNegativeList = [ '%.2f' % elem for elem in negativesList]
# # print formattedPositiveList
# # print "Number of positives: ", np.size(positiveList)
# # print "\n", formattedNegativeList
# # print "Number of negatives: ", np.size(negativesList)
# # print "\n", greaterThan
# if np.size(hSList) == 0:
# 	posONegatives = np.size(positiveList)/float(np.size(negativesList)+np.size(positiveList)) # positives/negatives
# 	negOPositives = np.size(negativesList)/float(np.size(positiveList)+np.size(negativesList)) # negatives/positives
# 	temp = -posONegatives*np.log2(posONegatives)-negOPositives*np.log2(negOPositives)
# 	posNumCount = np.size(positiveList)
# 	negNumCount = np.size(negativesList)
# 	hSList.append(temp)
# elif np.size(positiveList) != posNumCount and np.size(negativesList) != negNumCount:
# 	posONegatives = np.size(positiveList)/float(np.size(negativesList)+np.size(positiveList)) # positives/negatives
# 	negOPositives = np.size(negativesList)/float(np.size(positiveList)+np.size(negativesList)) # negatives/positives
# 	temp = -posONegatives*np.log2(posONegatives)-negOPositives*np.log2(negOPositives)
# 	if temp < hSList[0]: # 0 would i, in some loop for each column/feature
# 		hSList[0] = temp

# print hSList
# # del hSList[:] # used to empty a list
# del negativesList[:]
# del positiveList[:]
