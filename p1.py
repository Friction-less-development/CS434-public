import numpy as np

#Authors: Rex Henzie, Benjamin Richards, and Michael Giovannoni
#Question 1 Linear Regression

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
# print X #for testing purposes
# print "---"
# print Y[-1] # for testing purposes
# print "---"
XT = np.transpose(X)
# print XT
# print "---"
XTX = np.dot(XT, X)
XInv = np.linalg.inv(XTX)
# print XInv
# XTY = np.dot(XT, Y)
# print "---"
# print XTY
# print "---"
# print XT.shape
# print Y.shape
XInvXT = np.dot(XInv, XT)
w = np.dot(XInvXT, Y)
print "Training with dummy variable: W: "
print w
# print X

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
# print X #for testing purposes
# print "---"
# print Y[-1] # for testing purposes
# print "---"
XT = np.transpose(X)
# print XT
# print "---"
XTX = np.dot(XT, X)
XInv = np.linalg.inv(XTX)
# print XInv
# XTY = np.dot(XT, Y)
# print "---"
# print XTY
# print "---"
# print XT.shape
# print Y.shape
XInvXT = np.dot(XInv, XT)
w = np.dot(XInvXT, Y)
print "Test with dummy variable: W: "
print w
# print X

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
# print X #for testing purposes
# print "---"
# print Y[-1] # for testing purposes
# print "---"
XT = np.transpose(X)
# print XT
# print "---"
XTX = np.dot(XT, X)
XInv = np.linalg.inv(XTX)
# print XInv
# XTY = np.dot(XT, Y)
# print "---"
# print XTY
# print "---"
# print XT.shape
# print Y.shape
XInvXT = np.dot(XInv, XT)
w = np.dot(XInvXT, Y)
print "Training without dummy variable: W: "
print w
# print X

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
# print X #for testing purposes
# print "---"
# print Y[-1] # for testing purposes
# print "---"
XT = np.transpose(X)
# print XT
# print "---"
XTX = np.dot(XT, X)
XInv = np.linalg.inv(XTX)
# print XInv
# XTY = np.dot(XT, Y)
# print "---"
# print XTY
# print "---"
# print XT.shape
# print Y.shape
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


# part 4
print "Part 4"
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
# print X
# print X.shape
# print s.shape
s.shape = (433, 1)
s2.shape = (433, 1)
# print s.shape
X = np.append(X, s, 1)
X = np.append(X, s2, 1)
# print X #for testing purposes
# print "---"
# print Y[-1] # for testing purposes
# print "---"
XT = np.transpose(X)
# print XT
# print "---"
XTX = np.dot(XT, X)
XInv = np.linalg.inv(XTX)
# print XInv
# XTY = np.dot(XT, Y)
# print "---"
# print XTY
# print "---"
# print XT.shape
# print Y.shape
XInvXT = np.dot(XInv, XT)
w = np.dot(XInvXT, Y)
# print "Training with d=2 variables: W: "
# print w
# print X

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
print "Training with d=2 variables: Average sum of squares: ", avgSum

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
# print X
# print X.shape
# print s.shape
s3.shape = (74, 1)
s4.shape = (74, 1)
# print s3.shape
X = np.append(X, s3, 1)
X = np.append(X, s4, 1)
# print X #for testing purposes
# print "---"
# print Y[-1] # for testing purposes
# print "---"
XT = np.transpose(X)
# print XT
# print "---"
XTX = np.dot(XT, X)
XInv = np.linalg.inv(XTX)
# print XInv
# XTY = np.dot(XT, Y)
# print "---"
# print XTY
# print "---"
# print XT.shape
# print Y.shape
XInvXT = np.dot(XInv, XT)
w = np.dot(XInvXT, Y)
# print "Test with d=2 variable: W: "
# print w
# print X

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
print "Test with d=2 variables: Average sum of squares: ", avgSum