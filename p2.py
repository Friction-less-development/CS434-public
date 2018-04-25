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

#HOW TO RUN: python p2.py

#Question 2 Decision Tree
#Part 1 Decision Stump

avgTrainList = [] # hold the values of avg ASE for training
avgTestList = [] # hold the values of avg ASE for test

# part 1-2 with dummy variable
np.set_printoptions(suppress=True)
X = np.zeros(shape=(1,30)) # 14 with dummy variable
Y = np.zeros(shape=(1,1))
print "Parts 1-2"
firstLine = True
with open('knn_train.csv','r') as f:
    for line in f:
    	featureNum = 0
    	lineWords = [] # used to add a 1s to the first column (dummy variable)
    	averageValue = []
        for word in line.split(','):
           if featureNum > 0:
           	lineWords.append(float(word))
           	featureNum += 1
           else:
           	averageValue.append(float(word))
           	featureNum += 1
        if firstLine:
        	X[0] = lineWords
        	Y[0] = averageValue
        	firstLine = False
        else:
        	X = np.vstack((X, lineWords))
        	Y = np.vstack((Y, averageValue))

print X
# print Y