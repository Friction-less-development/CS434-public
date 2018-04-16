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

#HOW TO RUN: python2.7 p1.py
#	For parts 1-3 it will say which part it is outputting, to help make it clear
#	it will output the data required for part 1, 2 and 3, saying it is with or without a dummy variable, and if it is training or test data
#	lastly it will output p1part4.png which is the graph for part 4

#Question 1 Linear Regression

avgTrainList = [] # hold the values of avg ASE for training
avgTestList = [] # hold the values of avg ASE for test

# part 1-2 with dummy variable
np.set_printoptions(suppress=True)
X = np.zeros(shape=(1,14)) # 14 with dummy variable
Y = np.zeros(shape=(1,1))
print "Parts 1-2"
firstLine = True
with open('housing_train.txt','r') as f:
    for line in f:
    	featureNum = 0
    	lineWords = [1.] # used to add a 1s to the first column (dummy variable)
    	averageValue = []
        for word in line.split():
           if featureNum < 13:
           	lineWords.append(float(word))
           	featureNum += 1
           else:
           	averageValue.append(float(word))
        if firstLine:
        	X[0] = lineWords
        	Y[0] = averageValue
        	firstLine = False
        else:
        	X = np.vstack((X, lineWords))
        	Y = np.vstack((Y, averageValue))

XT = np.transpose(X)
XTX = np.dot(XT, X)
XInv = np.linalg.inv(XTX)
XInvXT = np.dot(XInv, XT)
w = np.dot(XInvXT, Y)
print "Training with dummy variable: W: "
print w

for i in range(0, 14):
	X[:,i] *= w[i]
avgSum = 0.0
columnSum = 0.0
# get average SSE
for i in range(0, 433):
	columnSum = 0.0
	for j in range(0, 14):
		columnSum += X[i][j]
	avgSum += (Y[i][0]-columnSum)*(Y[i][0]-columnSum)

avgSum = avgSum / 433
print "Training with dummy variable: Average sum of squares: ", avgSum

#test data with dummy variable
X = np.zeros(shape=(1,14)) # 14 with dummy variable
Y = np.zeros(shape=(1,1))

firstLine = True
with open('housing_test.txt','r') as f:
    for line in f:
    	featureNum = 0
    	lineWords = [1.] # used to add a 1s to the first column (dummy variable)
    	averageValue = []
        for word in line.split():
           if featureNum < 13:
           	lineWords.append(float(word))
           	featureNum += 1
           else:
           	averageValue.append(float(word))
        if firstLine:
        	X[0] = lineWords
        	Y[0] = averageValue
        	firstLine = False
        else:
        	X = np.vstack((X, lineWords))
        	Y = np.vstack((Y, averageValue))

XT = np.transpose(X)
XTX = np.dot(XT, X)
XInv = np.linalg.inv(XTX)
XInvXT = np.dot(XInv, XT)
w = np.dot(XInvXT, Y)
print "Test with dummy variable: W: "
print w

for i in range(0, 14):
	X[:,i] *= w[i]
avgSum = 0.0
columnSum = 0.0
# get average SSE
for i in range(0, 74):
	columnSum = 0.0
	for j in range(0, 14):
		columnSum += X[i][j]
	avgSum += (Y[i][0]-columnSum)*(Y[i][0]-columnSum)

avgSum = avgSum / 74
print "Test with dummy variable: Average sum of squares: ", avgSum
print "\nPart 3"

# # part 3 without dummy variable (aka part 1-2 but without dummy variable)
X = np.zeros(shape=(1,13)) # 13 without dummy variable
Y = np.zeros(shape=(1,1))

firstLine = True
with open('housing_train.txt','r') as f:
    for line in f:
    	featureNum = 0
    	lineWords = []
    	averageValue = []
        for word in line.split():
           if featureNum < 13:
           	lineWords.append(float(word))
           	featureNum += 1
           else:
           	averageValue.append(float(word))
        if firstLine:
        	X[0] = lineWords
        	Y[0] = averageValue
        	firstLine = False
        else:
        	X = np.vstack((X, lineWords))
        	Y = np.vstack((Y, averageValue))

XT = np.transpose(X)
XTX = np.dot(XT, X)
XInv = np.linalg.inv(XTX)
XInvXT = np.dot(XInv, XT)
w = np.dot(XInvXT, Y)
print "Training without dummy variable: W: "
print w

for i in range(0, 13):
	X[:,i] *= w[i]
avgSum = 0.0
columnSum = 0.0
# get average SSE
for i in range(0, 433):
	columnSum = 0.0
	for j in range(0, 13):
		columnSum += X[i][j]
	avgSum += (Y[i][0]-columnSum)*(Y[i][0]-columnSum)

avgSum = avgSum / 433
print "Training without dummy variable: Average sum of squares: ", avgSum
avgTrainList.append(float(avgSum))

#test data without dummy variable
X = np.zeros(shape=(1,13)) # 14 with dummy variable
Y = np.zeros(shape=(1,1))

firstLine = True
with open('housing_test.txt','r') as f:
    for line in f:
    	featureNum = 0
    	lineWords = []
    	averageValue = []
        for word in line.split():
           if featureNum < 13:
           	lineWords.append(float(word))
           	featureNum += 1
           else:
           	averageValue.append(float(word))
        if firstLine:
        	X[0] = lineWords
        	Y[0] = averageValue
        	firstLine = False
        else:
        	X = np.vstack((X, lineWords))
        	Y = np.vstack((Y, averageValue))

XT = np.transpose(X)
XTX = np.dot(XT, X)
XInv = np.linalg.inv(XTX)

XInvXT = np.dot(XInv, XT)
w = np.dot(XInvXT, Y)
print "Test without dummy variable: W: "
print w
# print X

for i in range(0, 13):
	X[:,i] *= w[i]
avgSum = 0.0
columnSum = 0.0
# get average SSE
for i in range(0, 74):
	columnSum = 0.0
	for j in range(0, 13):
		columnSum += X[i][j]
	avgSum += (Y[i][0]-columnSum)*(Y[i][0]-columnSum)

avgSum = avgSum / 74
print "Test without dummy variable: Average sum of squares: ", avgSum
avgTestList.append(float(avgSum))

# part 4
# print "\nPart 4"
mu, sigma = 0, 0.1 # mean and standard deviation
s = np.random.normal(mu, sigma, 433) # training
s2 = np.random.normal(mu, sigma, 433) # training
s3 = np.random.normal(mu, sigma, 74) # test
s4 = np.random.normal(mu, sigma, 74) # test
X = np.zeros(shape=(1,13)) # 14 with dummy variable
Y = np.zeros(shape=(1,1))
firstLine = True
with open('housing_train.txt','r') as f:
    for line in f:
    	featureNum = 0
    	lineWords = []
    	averageValue = []
        for word in line.split():
           if featureNum < 13:
           	lineWords.append(float(word))
           	featureNum += 1
           else:
           	averageValue.append(float(word))
        if firstLine:
        	X[0] = lineWords
        	Y[0] = averageValue
        	firstLine = False
        else:
        	X = np.vstack((X, lineWords))
        	Y = np.vstack((Y, averageValue))

s.shape = (433, 1)
s2.shape = (433, 1)
X = np.append(X, s, 1)
X = np.append(X, s2, 1)
XT = np.transpose(X)
XTX = np.dot(XT, X)
XInv = np.linalg.inv(XTX)
XInvXT = np.dot(XInv, XT)
w = np.dot(XInvXT, Y)

for i in range(0, 15):
	X[:,i] *= w[i]
avgSum = 0.0
columnSum = 0.0
# get average SSE
for i in range(0, 433):
	columnSum = 0.0
	for j in range(0, 15):
		columnSum += X[i][j]
	avgSum += (Y[i][0]-columnSum)*(Y[i][0]-columnSum)

avgSum = avgSum / 433
# print "Training with d=2:"
# print "Average sum of squares: ", avgSum

avgTrainList.append(float(avgSum))

# test with d=2
X = np.zeros(shape=(1,13))
Y = np.zeros(shape=(1,1))
firstLine = True
with open('housing_test.txt','r') as f:
    for line in f:
    	featureNum = 0
    	lineWords = []
    	averageValue = []
        for word in line.split():
           if featureNum < 13:
           	lineWords.append(float(word))
           	featureNum += 1
           else:
           	averageValue.append(float(word))
        if firstLine:
        	X[0] = lineWords
        	Y[0] = averageValue
        	firstLine = False
        else:
        	X = np.vstack((X, lineWords))
        	Y = np.vstack((Y, averageValue))

s3.shape = (74, 1)
s4.shape = (74, 1)
X = np.append(X, s3, 1)
X = np.append(X, s4, 1)
XT = np.transpose(X)
XTX = np.dot(XT, X)
XInv = np.linalg.inv(XTX)
XInvXT = np.dot(XInv, XT)
w = np.dot(XInvXT, Y)

for i in range(0, 15):
	X[:,i] *= w[i]
avgSum = 0.0
columnSum = 0.0
# get average SSE
for i in range(0, 74):
	columnSum = 0.0
	for j in range(0, 15):
		columnSum += X[i][j]
	avgSum += (Y[i][0]-columnSum)*(Y[i][0]-columnSum)

avgSum = avgSum / 74
# print "Test with d=2:"
# print "Average sum of squares: ", avgSum
avgTestList.append(float(avgSum))

for k in range(4, 12, 2):
	colRange = k+13
	sTrainList = []
	sTestList = []
	for j in range(0, k):
		sTrainList.append(np.random.normal(mu, sigma, 433))
		sTestList.append(np.random.normal(mu, sigma, 74))
	# s = np.random.normal(mu, sigma, 433) # training
	# s2 = np.random.normal(mu, sigma, 433) # training
	# s3 = np.random.normal(mu, sigma, 74) # test
	# s4 = np.random.normal(mu, sigma, 74) # test
	X = np.zeros(shape=(1,13)) # 14 with dummy variable
	Y = np.zeros(shape=(1,1))
	firstLine = True
	with open('housing_train.txt','r') as f:
	    for line in f:
	    	featureNum = 0
	    	lineWords = []
	    	averageValue = []
	        for word in line.split():
	           if featureNum < 13:
	           	lineWords.append(float(word))
	           	featureNum += 1
	           else:
	           	averageValue.append(float(word))
	        if firstLine:
	        	X[0] = lineWords
	        	Y[0] = averageValue
	        	firstLine = False
	        else:
	        	X = np.vstack((X, lineWords))
	        	Y = np.vstack((Y, averageValue))
	for j in range(0, k):
		sTrainList[j].shape = (433, 1)
	# s.shape = (433, 1)
	# s2.shape = (433, 1)
	for j in range(0, k):
		X = np.append(X, sTrainList[j], 1)
	# X = np.append(X, s, 1)
	# X = np.append(X, s2, 1)
	XT = np.transpose(X)
	XTX = np.dot(XT, X)
	XInv = np.linalg.inv(XTX)
	XInvXT = np.dot(XInv, XT)
	w = np.dot(XInvXT, Y)

	for i in range(0, colRange):
		X[:,i] *= w[i]
	avgSum = 0.0
	columnSum = 0.0
	# get average SSE
	for i in range(0, 433):
		columnSum = 0.0
		for j in range(0, colRange):
			columnSum += X[i][j]
		avgSum += (Y[i][0]-columnSum)*(Y[i][0]-columnSum)

	avgSum = avgSum / 433
	# print "Training with d=", k
	# print "Average sum of squares: ", avgSum

	avgTrainList.append(float(avgSum))

	X = np.zeros(shape=(1,13))
	Y = np.zeros(shape=(1,1))
	firstLine = True
	with open('housing_test.txt','r') as f:
	    for line in f:
	    	featureNum = 0
	    	lineWords = []
	    	averageValue = []
	        for word in line.split():
	           if featureNum < 13:
	           	lineWords.append(float(word))
	           	featureNum += 1
	           else:
	           	averageValue.append(float(word))
	        if firstLine:
	        	X[0] = lineWords
	        	Y[0] = averageValue
	        	firstLine = False
	        else:
	        	X = np.vstack((X, lineWords))
	        	Y = np.vstack((Y, averageValue))
	for j in range(0, k):
		sTestList[j].shape = (74, 1)
	# s3.shape = (74, 1)
	# s4.shape = (74, 1)
	for j in range(0, k):
		X = np.append(X, sTestList[j], 1)
	# X = np.append(X, s3, 1)
	# X = np.append(X, s4, 1)
	XT = np.transpose(X)
	XTX = np.dot(XT, X)
	XInv = np.linalg.inv(XTX)
	XInvXT = np.dot(XInv, XT)
	w = np.dot(XInvXT, Y)

	for i in range(0, colRange):
		X[:,i] *= w[i]
	avgSum = 0.0
	columnSum = 0.0
	# get average SSE
	for i in range(0, 74):
		columnSum = 0.0
		for j in range(0, colRange):
			columnSum += X[i][j]
		avgSum += (Y[i][0]-columnSum)*(Y[i][0]-columnSum)

	avgSum = avgSum / 74
	# print "Test with d=", k
	# print "Average sum of squares: ", avgSum
	avgTestList.append(float(avgSum))

# print "avgTrainList: ", avgTrainList
# print "avgTestList: ", avgTestList

plt.figure(1)
plt.ylabel('ASE')
plt.xlabel('Additional d features')
xAxis = [0, 2, 4, 6, 8, 10]
plt.title('ASE VS Number of Additional Features')
plt.plot(xAxis, avgTrainList, 'ro', label='Training Data')
plt.plot(xAxis, avgTestList, 'bo', label='Test Data')
# plt.plot(0, avgTrainList[0], 'ro', 2, avgTrainList[1], 'ro', 4, avgTrainList[2], 'ro', 6, avgTrainList[3], 'ro', 8, avgTrainList[4], 'ro', 10, avgTrainList[5], 'ro')
# plt.plot(0, avgTestList[0], 'bo', 2, avgTestList[1], 'bo', 4, avgTestList[2], 'bo', 6, avgTestList[3], 'bo', 8, avgTestList[4], 'bo', 10, avgTestList[5], 'bo')

plt.axis([0,12, 0, 30])
plt.legend()
plt.savefig('p1part4.png')