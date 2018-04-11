import numpy as np

np.set_printoptions(suppress=True)
X = np.zeros(shape=(1,14)) # 13 if without dummy column, else 14.
Y = np.zeros(shape=(1,1))

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
print "W: "
print w