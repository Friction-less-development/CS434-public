import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SUB1 = np.zeros(shape=(1, 8))
SUB1HYPO = np.zeros(shape=(1,1)) # this is whether there will be a hypo in 30 minutes or not (what we're trying to predict)
SUB1INDICES = np.zeros(shape=(1,1)) # this will hold the indices for subject 1

firstLine = True
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
        	SUB1[0] = lineWords
        	SUB1HYPO[0] = averageValue
        	firstLine = False
        else:
        	SUB1 = np.vstack((SUB1, lineWords))
        	SUB1HYPO = np.vstack((SUB1HYPO, averageValue))

firstLine = True
with open('list_1.csv','r') as f:
    for line in f:
    	lineWords = []
        for word in line.split(','):
        	lineWords.append(int(word))

        if firstLine:
        	SUB1INDICES[0] = lineWords
        	firstLine = False
        else:
        	SUB1INDICES = np.vstack((SUB1INDICES, lineWords))

print "Subject_1"
print np.shape(SUB1)
print np.shape(SUB1HYPO)
print np.shape(SUB1INDICES)

SUB4 = np.zeros(shape=(1, 8))
SUB4HYPO = np.zeros(shape=(1,1)) # this is whether there will be a hypo in 30 minutes or not (what we're trying to predict)
SUB4INDICES = np.zeros(shape=(1,1)) # this will hold the indices for subject 4



firstLine = True
with open('Subject_4.csv','r') as f:
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
        	SUB4[0] = lineWords
        	SUB4HYPO[0] = averageValue
        	firstLine = False
        else:
        	SUB4 = np.vstack((SUB4, lineWords))
        	SUB4HYPO = np.vstack((SUB4HYPO, averageValue))

firstLine = True
with open('list_4.csv','r') as f:
    for line in f:
    	lineWords = []
        for word in line.split(','):
        	lineWords.append(int(word))

        if firstLine:
        	SUB4INDICES[0] = lineWords
        	firstLine = False
        else:
        	SUB4INDICES = np.vstack((SUB4INDICES, lineWords))

print "Subject_4"
print np.shape(SUB4)
print np.shape(SUB4HYPO)
print np.shape(SUB4INDICES)

SUB6 = np.zeros(shape=(1, 8))
SUB6HYPO = np.zeros(shape=(1,1)) # this is whether there will be a hypo in 30 minutes or not (what we're trying to predict)
SUB6INDICES = np.zeros(shape=(1,1)) # this will hold the indices for subject 6



firstLine = True
with open('Subject_6.csv','r') as f:
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
        	SUB6[0] = lineWords
        	SUB6HYPO[0] = averageValue
        	firstLine = False
        else:
        	SUB6 = np.vstack((SUB6, lineWords))
        	SUB6HYPO = np.vstack((SUB6HYPO, averageValue))

firstLine = True
with open('list_6.csv','r') as f:
    for line in f:
    	lineWords = []
        for word in line.split(','):
        	lineWords.append(int(word))

        if firstLine:
        	SUB6INDICES[0] = lineWords
        	firstLine = False
        else:
        	SUB6INDICES = np.vstack((SUB6INDICES, lineWords))

print "Subject_6"
print np.shape(SUB6)
print np.shape(SUB6HYPO)
print np.shape(SUB6INDICES)

SUB9 = np.zeros(shape=(1, 8))
SUB9HYPO = np.zeros(shape=(1,1)) # this is whether there will be a hypo in 30 minutes or not (what we're trying to predict)
SUB9INDICES = np.zeros(shape=(1,1)) # this will hold the indices for subject 9



firstLine = True
with open('Subject_9.csv','r') as f:
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
        	SUB9[0] = lineWords
        	SUB9HYPO[0] = averageValue
        	firstLine = False
        else:
        	SUB9 = np.vstack((SUB9, lineWords))
        	SUB9HYPO = np.vstack((SUB9HYPO, averageValue))

firstLine = True
with open('list_9.csv','r') as f:
    for line in f:
    	lineWords = []
        for word in line.split(','):
        	lineWords.append(int(word))

        if firstLine:
        	SUB9INDICES[0] = lineWords
        	firstLine = False
        else:
        	SUB9INDICES = np.vstack((SUB9INDICES, lineWords))

print "Subject_9"
print np.shape(SUB9)
print np.shape(SUB9HYPO)
print np.shape(SUB9INDICES)