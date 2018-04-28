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

print X
# print Y
print "\n"
XSortTest = X[X[:,1].argsort()]
Y = XSortTest[:, 1] # getting a single column, which might be useful for doing calculations on
print XSortTest
print "\n"
print Y

#global
bestStump = 0.0 # global, will be used to find the best overall decision stump h(s)
hSList = [] # list of h(s1), h(s2)... will be used later

# note that this is a per column basis
greaterThan = 0.0 # will be used to where the cutoff is, it will go anything greater than will become "+" while evertyhing less than or equal to will become "-", as shown in slide 19
positiveList = [] # will be used to hold everything greater than greaterThan value
negativesList = []  # will be used to hold everything less than or equal to greaterThan value

negativesList.append(Y[0])
greaterThan = Y[0]
for i in range(1, np.size(Y)):
	if Y[i] > greaterThan:
		positiveList.append(Y[i])
	else:
		negativesList.append(Y[i])

formattedPositiveList = [ '%.2f' % elem for elem in positiveList]
formattedNegativeList = [ '%.2f' % elem for elem in negativesList]
print formattedPositiveList
print "Number of positives: ", np.size(positiveList)
print "\n", formattedNegativeList
print "Number of negatives: ", np.size(negativesList)
print "\n", greaterThan