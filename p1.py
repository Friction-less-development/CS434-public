import numpy as np

X = np.zeros(shape=(1,14))
firstLine = True
with open('housing_train.txt','r') as f:
    for line in f:
    	featureNum = 0
    	# lineWords = [1.] # used to add a 1s to the first column (dummy variable)
        for word in line.split():

           if featureNum < 13:
           	lineWords.append(word)
           	featureNum += 1
        if firstLine:
        	X[0] = lineWords
        	firstLine = False
        else:
        	X = np.vstack((X, lineWords))
print X