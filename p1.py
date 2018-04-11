import numpy as np

X = np.zeros(shape=(1,13)) # 13 if without dummy column, else 14.
Y = np.zeros(shape=(1,1))

firstLine = True
with open('housing_train.txt','r') as f:
    for line in f:
    	featureNum = 0
    	lineWords = []
    	averageValue = []
    	# lineWords = [1.] # used to add a 1s to the first column (dummy variable)
        for word in line.split():
           if featureNum < 13:
           	lineWords.append(word)
           	featureNum += 1
           else:
           	averageValue.append(word)
        if firstLine:
        	X[0] = lineWords
        	Y[0] = averageValue
        	firstLine = False
        else:
        	X = np.vstack((X, lineWords))
        	Y = np.vstack((Y, averageValue))
print X #for testing purposes
print Y[-1] # for testing purposes