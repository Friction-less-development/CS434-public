import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

X = np.zeros(shape=(1, 8))
Y = np.zeros(shape=(1,1))

firstLine = True
lines = 1
with open('Subject_1.csv','r') as f:
    for line in f:
        featureNum = 0
        lineWords = []
        averageValue = []
        for word in line.split(','):
            if featureNum > 0 and featureNum < 9:
           	  lineWords.append(float(word))
           	  featureNum += 1
            elif featureNum == 9:
           	  averageValue.append(float(word))
            else:
              featureNum += 1
            # print lineWords

        if firstLine:
        	X[0] = lineWords
        	Y[0] = averageValue
        	firstLine = False
        else:
        	X = np.vstack((X, lineWords))
        	Y = np.vstack((Y, averageValue))


print np.shape(X)
print np.shape(Y)