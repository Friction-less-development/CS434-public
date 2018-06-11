import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.neural_network import MLPClassifier

numColumns = 8
instance = np.zeros(shape=(7, numColumns))
chunks = np.zeros(shape=(1, numColumns))
chunks1List = []
chunks2List = []
chunks4List = []
chunks6List = []
chunks9List = []
sub1HypoChunkList = []
sub4HypoChunkList = []
sub6HypoChunkList = []
sub9HypoChunkList = []
sub2HypoChunkList = [] # for verification subject
SUB1 = np.zeros(shape=(1, numColumns))
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
           	  averageValue.append(int(word))
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
#classWeights = compute_class_weight("balanced", [0., 1.], np.ravel(SUB1HYPO))
#print classWeights
#classWeights[0] = classWeights[0]/126. # can only put class weight of 0 in, so divide it by the wiehgt of class weight of 1, more or less.
counter = 0 # must get to at least equaling 6, which is a minimum of 7 instances for a 30 minute period
#subForest = RandomForestClassifier(criterion="entropy", max_depth=7, max_features=6, random_state=0, warm_start=True, bootstrap=False, class_weight={0.:classWeights[0]}) # class_weight={0.:classWeights[0]}

##normalizing the data. This isn't relevant elsewhere
maxtestvals = np.zeros(shape=(1,8))
mintestvals = np.zeros(shape=(1,8))
maxtestvals[0] = SUB1[0]
mintestvals[0] = SUB1[0]
for i in range(1,len(SUB1)):      #This loop finds the man/mix of each feature
 for j in range(0,8):
  if SUB1[i][j] > maxtestvals[0][j]:
   maxtestvals[0][j] = SUB1[i][j]
  if SUB1[i][j] < mintestvals[0][j]:
   mintestvals[0][j] = SUB1[i][j]
for i in range(0,len(SUB1)):      #This loop applies the normalization function
 for j in range(0,8):
  SUB1[i][j] = (SUB1[i][j] - mintestvals[0][j])/(maxtestvals[0][j] - mintestvals[0][j])
#data should be normalized now.



sub1HypoChunk = np.zeros(shape=(1,1))
for i in range (0, np.size(SUB1,0)):
    if counter == 0:
        sub1HypoChunk[0] = SUB1HYPO[i][:]
        chunks[0] = SUB1[i][:]
        counter += 1
    else:
        chunks = np.vstack((chunks, SUB1[i][:]))
        sub1HypoChunk = np.vstack((sub1HypoChunk, SUB1HYPO[i][:]))
        counter += 1
    if SUB1INDICES[i][0] - 1 != SUB1INDICES[i-1][0] and counter < 6:
        temp = np.zeros(shape=(1, numColumns))
        chunks = temp
        counter = 0
        temp2 = np.zeros(shape=(1,1))
        sub1HypoChunk = temp2
    elif SUB1INDICES[i][0] - 1 != SUB1INDICES[i-1][0] and counter >= 6:
        sub1HypoChunkList.append(sub1HypoChunk)
        chunks1List.append(chunks)
        temp = np.zeros(shape=(1, numColumns))
        chunks = temp
        temp2 = np.zeros(shape=(1,1))
        sub1HypoChunk = temp2
        counter = 0

counter = 0
pos1Instances = np.zeros(shape=(1,7,numColumns))
firstPos = 0
firstNeg = 0
neg1Instances = np.zeros(shape=(1,7,numColumns))
for i in range(0, len(chunks1List)):
    temp2 = np.zeros(shape=(7,1))
    sub1HypoChunk = temp2
    temp = np.zeros(shape=(7, numColumns))
    instance = temp
    for j in range(0, len(chunks1List[i])):
        if j<7:
            instance[j][:] = chunks1List[i][j][:]
            sub1HypoChunk[j][0] = int(sub1HypoChunkList[i][j][0])
        else:
            if sub1HypoChunk[len(sub1HypoChunk)-1] == float(1):
                if firstPos == 0:
                    pos1Instances[0] = instance
                else:
                    temp = np.zeros(shape=(1,7,8))
                    temp[0] = instance
                    pos1Instances = np.vstack((pos1Instances, temp))
                firstPos = firstPos + 1
            else:
                if firstNeg == 0:
                    neg1Instances[0] = instance
                else:
                    temp = np.zeros(shape=(1,7,8))
                    temp[0] = instance
                    neg1Instances = np.vstack((neg1Instances, temp))
                firstNeg = firstNeg +1
            tempNum = j%7-1
            instance = np.delete(instance, tempNum, 0)
            sub1HypoChunk = np.delete(sub1HypoChunk, tempNum, 0)
            instance = np.vstack((instance, chunks1List[i][j][:]))
            sub1HypoChunk = np.vstack((sub1HypoChunk, int(sub1HypoChunkList[i][j][0])))

print "\n"
print "Subject_1"
print np.shape(SUB1)
print np.shape(SUB1HYPO)
print np.shape(SUB1INDICES)
#-------------------------------------------------------------------------------
SUB4 = np.zeros(shape=(1, numColumns))
SUB4HYPO = np.zeros(shape=(1,1)) # this is whether there will be a hypo in 30 minutes or not (what we're trying to predict)
SUB4INDICES = np.zeros(shape=(1,1)) # this will hold the indices for subject 1


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
           	  averageValue.append(int(word))
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
#classWeights = compute_class_weight("balanced", [0., 1.], np.ravel(SUB1HYPO))
#print classWeights
#classWeights[0] = classWeights[0]/126. # can only put class weight of 0 in, so divide it by the wiehgt of class weight of 1, more or less.
counter = 0 # must get to at least equaling 6, which is a minimum of 7 instances for a 30 minute period
#subForest = RandomForestClassifier(criterion="entropy", max_depth=7, max_features=6, random_state=0, warm_start=True, bootstrap=False, class_weight={0.:classWeights[0]}) # class_weight={0.:classWeights[0]}

##normalizing the data. This isn't relevant elsewhere
maxtestvals = np.zeros(shape=(1,8))
mintestvals = np.zeros(shape=(1,8))
maxtestvals[0] = SUB4[0]
mintestvals[0] = SUB4[0]
for i in range(1,len(SUB4)):      #This loop finds the man/mix of each feature
 for j in range(0,8):
  if SUB4[i][j] > maxtestvals[0][j]:
   maxtestvals[0][j] = SUB4[i][j]
  if SUB4[i][j] < mintestvals[0][j]:
   mintestvals[0][j] = SUB4[i][j]
for i in range(0,len(SUB4)):      #This loop applies the normalization function
 for j in range(0,8):
  SUB4[i][j] = (SUB4[i][j] - mintestvals[0][j])/(maxtestvals[0][j] - mintestvals[0][j])
#data should be normalized now.



sub4HypoChunk = np.zeros(shape=(1,1))
for i in range (0, np.size(SUB4,0)):
    if counter == 0:
        sub4HypoChunk[0] = SUB4HYPO[i][:]
        chunks[0] = SUB4[i][:]
        counter += 1
    else:
        chunks = np.vstack((chunks, SUB4[i][:]))
        sub4HypoChunk = np.vstack((sub4HypoChunk, SUB4HYPO[i][:]))
        counter += 1
    if SUB4INDICES[i][0] - 1 != SUB4INDICES[i-1][0] and counter < 6:
        temp = np.zeros(shape=(1, numColumns))
        chunks = temp
        counter = 0
        temp2 = np.zeros(shape=(1,1))
        sub4HypoChunk = temp2
    elif SUB4INDICES[i][0] - 1 != SUB4INDICES[i-1][0] and counter >= 6:
        sub4HypoChunkList.append(sub4HypoChunk)
        chunks4List.append(chunks)
        temp = np.zeros(shape=(1, numColumns))
        chunks = temp
        temp2 = np.zeros(shape=(1,1))
        sub4HypoChunk = temp2
        counter = 0

counter = 0
pos4Instances = np.zeros(shape=(1,7,numColumns))
firstPos = 0
firstNeg = 0
neg4Instances = np.zeros(shape=(1,7,numColumns))
for i in range(0, len(chunks4List)):
    temp2 = np.zeros(shape=(7,1))
    sub4HypoChunk = temp2
    temp = np.zeros(shape=(7, numColumns))
    instance = temp
    for j in range(0, len(chunks4List[i])):
        if j<7:
            instance[j][:] = chunks4List[i][j][:]
            sub4HypoChunk[j][0] = int(sub4HypoChunkList[i][j][0])
        else:
            if sub4HypoChunk[len(sub4HypoChunk)-1] == float(1):
                if firstPos == 0:
                    pos4Instances[0] = instance
                else:
                    temp = np.zeros(shape=(1,7,8))
                    temp[0] = instance
                    pos4Instances = np.vstack((pos4Instances, temp))
                firstPos = firstPos + 1
            else:
                if firstNeg == 0:
                    neg4Instances[0] = instance
                else:
                    temp = np.zeros(shape=(1,7,8))
                    temp[0] = instance
                    neg4Instances = np.vstack((neg4Instances, temp))
                firstNeg = firstNeg +1
            tempNum = j%7-1
            instance = np.delete(instance, tempNum, 0)
            sub4HypoChunk = np.delete(sub4HypoChunk, tempNum, 0)
            instance = np.vstack((instance, chunks4List[i][j][:]))
            sub4HypoChunk = np.vstack((sub4HypoChunk, int(sub4HypoChunkList[i][j][0])))

print "\n"
print "Subject_4"
print np.shape(SUB4)
print np.shape(SUB4HYPO)
print np.shape(SUB4INDICES)
#-------------------------------------------------------------------------------
SUB6 = np.zeros(shape=(1, numColumns))
SUB6HYPO = np.zeros(shape=(1,1)) # this is whether there will be a hypo in 30 minutes or not (what we're trying to predict)
SUB6INDICES = np.zeros(shape=(1,1)) # this will hold the indices for subject 1


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
           	  averageValue.append(int(word))
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
#classWeights = compute_class_weight("balanced", [0., 1.], np.ravel(SUB1HYPO))
#print classWeights
#classWeights[0] = classWeights[0]/126. # can only put class weight of 0 in, so divide it by the wiehgt of class weight of 1, more or less.
counter = 0 # must get to at least equaling 6, which is a minimum of 7 instances for a 30 minute period
#subForest = RandomForestClassifier(criterion="entropy", max_depth=7, max_features=6, random_state=0, warm_start=True, bootstrap=False, class_weight={0.:classWeights[0]}) # class_weight={0.:classWeights[0]}

##normalizing the data. This isn't relevant elsewhere
maxtestvals = np.zeros(shape=(1,8))
mintestvals = np.zeros(shape=(1,8))
maxtestvals[0] = SUB6[0]
mintestvals[0] = SUB6[0]
for i in range(1,len(SUB6)):      #This loop finds the man/mix of each feature
 for j in range(0,8):
  if SUB6[i][j] > maxtestvals[0][j]:
   maxtestvals[0][j] = SUB6[i][j]
  if SUB6[i][j] < mintestvals[0][j]:
   mintestvals[0][j] = SUB6[i][j]
for i in range(0,len(SUB6)):      #This loop applies the normalization function
 for j in range(0,8):
  SUB6[i][j] = (SUB6[i][j] - mintestvals[0][j])/(maxtestvals[0][j] - mintestvals[0][j])
#data should be normalized now.



sub6HypoChunk = np.zeros(shape=(1,1))
for i in range (0, np.size(SUB6,0)):
    if counter == 0:
        sub6HypoChunk[0] = SUB6HYPO[i][:]
        chunks[0] = SUB6[i][:]
        counter += 1
    else:
        chunks = np.vstack((chunks, SUB6[i][:]))
        sub6HypoChunk = np.vstack((sub6HypoChunk, SUB6HYPO[i][:]))
        counter += 1
    if SUB6INDICES[i][0] - 1 != SUB6INDICES[i-1][0] and counter < 6:
        temp = np.zeros(shape=(1, numColumns))
        chunks = temp
        counter = 0
        temp2 = np.zeros(shape=(1,1))
        sub6HypoChunk = temp2
    elif SUB6INDICES[i][0] - 1 != SUB6INDICES[i-1][0] and counter >= 6:
        sub6HypoChunkList.append(sub6HypoChunk)
        chunks6List.append(chunks)
        temp = np.zeros(shape=(1, numColumns))
        chunks = temp
        temp2 = np.zeros(shape=(1,1))
        sub6HypoChunk = temp2
        counter = 0

counter = 0
pos6Instances = np.zeros(shape=(1,7,numColumns))
firstPos = 0
firstNeg = 0
neg6Instances = np.zeros(shape=(1,7,numColumns))
for i in range(0, len(chunks6List)):
    temp2 = np.zeros(shape=(7,1))
    sub6HypoChunk = temp2
    temp = np.zeros(shape=(7, numColumns))
    instance = temp
    for j in range(0, len(chunks6List[i])):
        if j<7:
            instance[j][:] = chunks6List[i][j][:]
            sub6HypoChunk[j][0] = int(sub6HypoChunkList[i][j][0])
        else:
            if sub6HypoChunk[len(sub6HypoChunk)-1] == float(1):
                if firstPos == 0:
                    pos6Instances[0] = instance
                else:
                    temp = np.zeros(shape=(1,7,8))
                    temp[0] = instance
                    pos6Instances = np.vstack((pos6Instances, temp))
                firstPos = firstPos + 1
            else:
                if firstNeg == 0:
                    neg6Instances[0] = instance
                else:
                    temp = np.zeros(shape=(1,7,8))
                    temp[0] = instance
                    neg6Instances = np.vstack((neg6Instances, temp))
                firstNeg = firstNeg +1
            tempNum = j%7-1
            instance = np.delete(instance, tempNum, 0)
            sub6HypoChunk = np.delete(sub6HypoChunk, tempNum, 0)
            instance = np.vstack((instance, chunks6List[i][j][:]))
            sub6HypoChunk = np.vstack((sub6HypoChunk, int(sub6HypoChunkList[i][j][0])))

print "\n"
print "Subject_6"
print np.shape(SUB6)
print np.shape(SUB6HYPO)
print np.shape(SUB6INDICES)
#-------------------------------------------------------------------------------
SUB9 = np.zeros(shape=(1, numColumns))
SUB9HYPO = np.zeros(shape=(1,1)) # this is whether there will be a hypo in 30 minutes or not (what we're trying to predict)
SUB9INDICES = np.zeros(shape=(1,1)) # this will hold the indices for subject 1


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
           	  averageValue.append(int(word))
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
#classWeights = compute_class_weight("balanced", [0., 1.], np.ravel(SUB1HYPO))
#print classWeights
#classWeights[0] = classWeights[0]/129. # can only put class weight of 0 in, so divide it by the wiehgt of class weight of 1, more or less.
counter = 0 # must get to at least equaling 9, which is a minimum of 7 instances for a 30 minute period
#subForest = RandomForestClassifier(criterion="entropy", max_depth=7, max_features=9, random_state=0, warm_start=True, bootstrap=False, class_weight={0.:classWeights[0]}) # class_weight={0.:classWeights[0]}

##normalizing the data. This isn't relevant elsewhere
maxtestvals = np.zeros(shape=(1,8))
mintestvals = np.zeros(shape=(1,8))
maxtestvals[0] = SUB9[0]
mintestvals[0] = SUB9[0]
for i in range(1,len(SUB9)):      #This loop finds the man/mix of each feature
 for j in range(0,8):
  if SUB9[i][j] > maxtestvals[0][j]:
   maxtestvals[0][j] = SUB9[i][j]
  if SUB9[i][j] < mintestvals[0][j]:
   mintestvals[0][j] = SUB9[i][j]
for i in range(0,len(SUB9)):      #This loop applies the normalization function
 for j in range(0,8):
  SUB9[i][j] = (SUB9[i][j] - mintestvals[0][j])/(maxtestvals[0][j] - mintestvals[0][j])
#data should be normalized now.



sub9HypoChunk = np.zeros(shape=(1,1))
for i in range (0, np.size(SUB9,0)):
    if counter == 0:
        sub9HypoChunk[0] = SUB9HYPO[i][:]
        chunks[0] = SUB9[i][:]
        counter += 1
    else:
        chunks = np.vstack((chunks, SUB9[i][:]))
        sub9HypoChunk = np.vstack((sub9HypoChunk, SUB9HYPO[i][:]))
        counter += 1
    if SUB9INDICES[i][0] - 1 != SUB9INDICES[i-1][0] and counter < 9:
        temp = np.zeros(shape=(1, numColumns))
        chunks = temp
        counter = 0
        temp2 = np.zeros(shape=(1,1))
        sub9HypoChunk = temp2
    elif SUB9INDICES[i][0] - 1 != SUB9INDICES[i-1][0] and counter >= 9:
        sub9HypoChunkList.append(sub9HypoChunk)
        chunks9List.append(chunks)
        temp = np.zeros(shape=(1, numColumns))
        chunks = temp
        temp2 = np.zeros(shape=(1,1))
        sub9HypoChunk = temp2
        counter = 0

counter = 0
pos9Instances = np.zeros(shape=(1,7,numColumns))
firstPos = 0
firstNeg = 0
neg9Instances = np.zeros(shape=(1,7,numColumns))
for i in range(0, len(chunks9List)):
    temp2 = np.zeros(shape=(7,1))
    sub9HypoChunk = temp2
    temp = np.zeros(shape=(7, numColumns))
    instance = temp
    for j in range(0, len(chunks9List[i])):
        if j<7:
            instance[j][:] = chunks9List[i][j][:]
            sub9HypoChunk[j][0] = int(sub9HypoChunkList[i][j][0])
        else:
            if sub9HypoChunk[len(sub9HypoChunk)-1] == float(1):
                if firstPos == 0:
                    pos9Instances[0] = instance
                else:
                    temp = np.zeros(shape=(1,7,8))
                    temp[0] = instance
                    pos9Instances = np.vstack((pos9Instances, temp))
                firstPos = firstPos + 1
            else:
                if firstNeg == 0:
                    neg9Instances[0] = instance
                else:
                    temp = np.zeros(shape=(1,7,8))
                    temp[0] = instance
                    neg9Instances = np.vstack((neg9Instances, temp))
                firstNeg = firstNeg +1
            tempNum = j%7-1
            instance = np.delete(instance, tempNum, 0)
            sub9HypoChunk = np.delete(sub9HypoChunk, tempNum, 0)
            instance = np.vstack((instance, chunks9List[i][j][:]))
            sub9HypoChunk = np.vstack((sub9HypoChunk, int(sub9HypoChunkList[i][j][0])))

print "\n"
print "Subject_9"
print np.shape(SUB9)
print np.shape(SUB9HYPO)
print np.shape(SUB9INDICES)
#-------------------------------------------------------------------------------
SUB2 = np.zeros(shape=(1, numColumns))
SUB2HYPO = np.zeros(shape=(1,1)) # this is whether there will be a hypo in 30 minutes or not (what we're trying to predict)
SUB2INDICES = np.zeros(shape=(1,1)) # this will hold the indices for subject 1


firstLine = True
with open('Subject_7_part1.csv','r') as f:
    for line in f:
        featureNum = 0
        lineWords = []
        averageValue = []
        for word in line.split(','):
            if featureNum > 0 and featureNum < 9:
           	  lineWords.append(float(word))
           	  featureNum += 1
            elif featureNum == 9:
           	  averageValue.append(int(word))
            else:
              featureNum += 1
            # print lineWords

        if firstLine:
        	SUB2[0] = lineWords
        	SUB2HYPO[0] = averageValue
        	firstLine = False
        else:
        	SUB2 = np.vstack((SUB2, lineWords))
        	SUB2HYPO = np.vstack((SUB2HYPO, averageValue))

firstLine = True
with open('list_7_part1.csv','r') as f:
    for line in f:
    	lineWords = []
        for word in line.split(','):
        	lineWords.append(int(word))

        if firstLine:
        	SUB2INDICES[0] = lineWords
        	firstLine = False
        else:
        	SUB2INDICES = np.vstack((SUB2INDICES, lineWords))

##normalizing the data. This isn't relevant elsewhere
maxtestvals = np.zeros(shape=(1,8))
mintestvals = np.zeros(shape=(1,8))
maxtestvals[0] = SUB2[0]
mintestvals[0] = SUB2[0]
for i in range(1,len(SUB2)):      #This loop finds the man/mix of each feature
 for j in range(0,8):
  if SUB2[i][j] > maxtestvals[0][j]:
   maxtestvals[0][j] = SUB2[i][j]
  if SUB2[i][j] < mintestvals[0][j]:
   mintestvals[0][j] = SUB2[i][j]
for i in range(0,len(SUB2)):      #This loop applies the normalization function
 for j in range(0,8):
  SUB2[i][j] = (SUB2[i][j] - mintestvals[0][j])/(maxtestvals[0][j] - mintestvals[0][j])
#data should be normalized now.



sub2HypoChunk = np.zeros(shape=(1,1))
for i in range (0, np.size(SUB2,0)):
    if counter == 0:
        sub2HypoChunk[0] = SUB2HYPO[i][:]
        chunks[0] = SUB2[i][:]
        counter += 1
    else:
        chunks = np.vstack((chunks, SUB2[i][:]))
        sub2HypoChunk = np.vstack((sub2HypoChunk, SUB2HYPO[i][:]))
        counter += 1
    if SUB2INDICES[i][0] - 1 != SUB2INDICES[i-1][0] and counter < 2:
        temp = np.zeros(shape=(1, numColumns))
        chunks = temp
        counter = 0
        temp2 = np.zeros(shape=(1,1))
        sub2HypoChunk = temp2
    elif SUB2INDICES[i][0] - 1 != SUB2INDICES[i-1][0] and counter >= 2:
        sub2HypoChunkList.append(sub2HypoChunk)
        chunks2List.append(chunks)
        temp = np.zeros(shape=(1, numColumns))
        chunks = temp
        temp2 = np.zeros(shape=(1,1))
        sub2HypoChunk = temp2
        counter = 0

counter = 0
pos2Instances = np.zeros(shape=(1,7,numColumns))
firstPos = 0
firstNeg = 0
neg2Instances = np.zeros(shape=(1,7,numColumns))
for i in range(0, len(chunks2List)):
    temp2 = np.zeros(shape=(7,1))
    sub2HypoChunk = temp2
    temp = np.zeros(shape=(7, numColumns))
    instance = temp
    for j in range(0, len(chunks2List[i])):
        if j<7:
            instance[j][:] = chunks2List[i][j][:]
            sub2HypoChunk[j][0] = int(sub2HypoChunkList[i][j][0])
        else:
            if sub2HypoChunk[len(sub2HypoChunk)-1] == float(1):
                if firstPos == 0:
                    pos2Instances[0] = instance
                else:
                    temp = np.zeros(shape=(1,7,8))
                    temp[0] = instance
                    pos2Instances = np.vstack((pos2Instances, temp))
                firstPos = firstPos + 1
            else:
                if firstNeg == 0:
                    neg2Instances[0] = instance
                else:
                    temp = np.zeros(shape=(1,7,8))
                    temp[0] = instance
                    neg2Instances = np.vstack((neg2Instances, temp))
                firstNeg = firstNeg +1
            tempNum = j%7-1
            instance = np.delete(instance, tempNum, 0)
            sub2HypoChunk = np.delete(sub2HypoChunk, tempNum, 0)
            instance = np.vstack((instance, chunks2List[i][j][:]))
            sub2HypoChunk = np.vstack((sub2HypoChunk, int(sub2HypoChunkList[i][j][0])))

print "\n"
print "Subject_2"
print np.shape(SUB2)
print np.shape(SUB2HYPO)
print np.shape(SUB2INDICES)
#--------------------------------------------------------
TEMPFTESTLIST = []
firstLine = True
with open('subject7_instances.csv','r') as f:
    for line in f:
        featureNum = 0
        lineWords = []
        averageValue = []
        for word in line.split(','):
            if featureNum > 6:
           	  lineWords.append(float(word))
           	  featureNum += 1
            else:
              featureNum += 1
            # print lineWords

        TEMPFTESTLIST.append(lineWords)

##normalizing the data. This isn't relevant elsewhere
maxtestvals = np.zeros(shape=(1,56))
mintestvals = np.zeros(shape=(1,56))
maxtestvals[0] = TEMPFTESTLIST[0]
mintestvals[0] = TEMPFTESTLIST[0]
for i in range(1,len(TEMPFTESTLIST)):      #This loop finds the man/mix of each feature
 for j in range(0,56):
  if TEMPFTESTLIST[i][j] > maxtestvals[0][j]:
   maxtestvals[0][j] = TEMPFTESTLIST[i][j]
  if TEMPFTESTLIST[i][j] < mintestvals[0][j]:
   mintestvals[0][j] = TEMPFTESTLIST[i][j]
for i in range(0,len(TEMPFTESTLIST)):      #This loop applies the normalization function
 for j in range(0,56):
  TEMPFTESTLIST[i][j] = (TEMPFTESTLIST[i][j] - mintestvals[0][j])/(maxtestvals[0][j] - mintestvals[0][j])
#data should be normalized now.

allgenInstances = np.zeros(shape=(len(TEMPFTESTLIST),4,7))
for i  in range(0, len(TEMPFTESTLIST)):
	for j in range(0, 4):
		tempRow = []
		for k in range(0, 7):
			allgenInstances[i][j][k] = TEMPFTESTLIST[i][7*j + k]


print "\n"
print "Subject_2"
print np.shape(TEMPFTESTLIST)


#Adds an entry to passed in closestNeighbor array if the norm is smaller then the entries in the passed in closestNeighbor array.
def addClosestNeighbor(closestNeighbors, norm, type):
 for i in range(0,k):
  if(norm < closestNeighbors[i][1] or closestNeighbors[i][1] == -1):
   closestNeighbors[i][1] = norm
   closestNeighbors[i][0] = type
   break;
 return closestNeighbors

#Polls all entries in closestNeighbors and gives back the majority
def getMajority(closestNeighbors):
 defaultValue = 0
 for i in range(0,k):
  if(closestNeighbors[i][0] == 1):
   print "a positive!"
   defaultValue += (len(neg2Instances))
  else:
   defaultValue -= (len(pos2Instances))
 if (defaultValue > 0):
  return 1
 elif(defaultValue < 0):
  return -1
 else:
  print "ERROR: Try an odd k value"
  return -1

#Compares guesses to answers and converts it to a percentage accuracy.
def getAccuracy(guesses, answers):
 errors = 0;
 for i in range(0,len(guesses)):
  if(guesses[i] != answers[i]):
   errors = errors + 1
 print errors
 accuracy = 100 - 100*(float(errors)/float(len(guesses)))
 print "Accuracy: "
 print accuracy
 print "%"
 return accuracy

all1Instances = np.zeros(shape=(len(pos1Instances) + len(neg1Instances),7,8))
all1Instances = np.vstack((pos1Instances, neg1Instances))
all2Instances = np.zeros(shape=(len(pos2Instances) + len(neg2Instances),7,8))
all2Instances = np.vstack((pos2Instances, neg2Instances))
all4Instances = np.zeros(shape=(len(pos4Instances) + len(neg4Instances),7,8))
all4Instances = np.vstack((pos4Instances, neg4Instances))
all6Instances = np.zeros(shape=(len(pos6Instances) + len(neg6Instances),7,8))
all6Instances = np.vstack((pos6Instances, neg6Instances))
all9Instances = np.zeros(shape=(len(pos9Instances) + len(neg9Instances),7,8))
all9Instances = np.vstack((pos9Instances, neg9Instances))
allInstances = np.zeros(shape=(len(all1Instances) + len(all4Instances) + len(all6Instances) + len(all9Instances),7,8))
allInstances = np.vstack((all1Instances, all4Instances, all6Instances, all9Instances))
hypo = np.zeros(shape=(len(allInstances),1))
for i in range(0,len(pos1Instances)):
    hypo[i] = 1
for i in range(len(pos1Instances),len(pos1Instances) + len(neg1Instances)):
    hypo[i] = -1
for i in range(len(pos1Instances) + len(neg1Instances), len(pos1Instances) + len(neg1Instances) + len(pos4Instances)):
    hypo[i] = 1
for i in range(len(pos1Instances) + len(neg1Instances) + len(pos4Instances), len(pos1Instances) + len(neg1Instances) + len(pos4Instances) + len(neg4Instances)):
    hypo[i] = -1
for i in range(len(pos1Instances) + len(neg1Instances) + len(pos4Instances) + len(neg4Instances), len(pos1Instances) + len(neg1Instances) + len(pos4Instances) + len(neg4Instances) + len(pos6Instances)):
    hypo[i] = 1
for i in range(len(pos1Instances) + len(neg1Instances) + len(pos4Instances) + len(neg4Instances) + len(pos6Instances), len(pos1Instances) + len(neg1Instances) + len(pos4Instances) + len(neg4Instances) + len(pos6Instances) + len(neg6Instances)):
    hypo[i] = -1
for i in range(len(pos1Instances) + len(neg1Instances) + len(pos4Instances) + len(neg4Instances) + len(pos6Instances) + len(neg6Instances), len(pos1Instances) + len(neg1Instances) + len(pos4Instances) + len(neg4Instances) + len(pos6Instances) + len(neg6Instances) +len(pos9Instances)):
    hypo[i] = 1
for i in range(len(pos1Instances) + len(neg1Instances) + len(pos4Instances) + len(neg4Instances) + len(pos6Instances) + len(neg6Instances) +len(pos9Instances), len(pos1Instances) + len(neg1Instances) + len(pos4Instances) + len(neg4Instances) + len(pos6Instances) + len(neg6Instances) +len(pos9Instances) + len(neg9Instances)):
    hypo[i] = -1


transInstances = np.zeros(shape=(len(allInstances),8,7))
for i in range(0, len(allInstances)):
    transInstances[i] = allInstances[i].T
meanOfEachFeature = np.zeros(shape=(len(allInstances),4))
for i in range(0, len(allInstances)):
    for j in range(0, 4):
        meanOfEachFeature[i][j] = np.mean(transInstances[i][j])
trans2Instances = np.zeros(shape=(len(all2Instances),8,7))
for i in range(0, len(all2Instances)):
    trans2Instances[i] = all2Instances[i].T

def predictTrainingSet(k,f):
 resultY = np.zeros(shape=(len(allgenInstances),1))
 for i in range(0,len(allgenInstances)):
  print i
  closestNeighbors = np.zeros(shape=(k,2))
  closestNeighbors.fill(-1)
  for j in range(0,len(trans2Instances)):
   if(True):
    type = hypo[j]
    norm = np.linalg.norm(np.absolute(np.subtract(allgenInstances[i][f], trans2Instances[j][f])))
    closestNeighbors = addClosestNeighbor(closestNeighbors, norm, type)
  resultY[i] = getMajority(closestNeighbors)
 return resultY
 #getAccuracy(resultY, hypo2)
k=75
firstResults = predictTrainingSet(k,0) #230
k=75
secondResults = predictTrainingSet(k,1)
k=3
thirdResults = predictTrainingSet(k,2)
k=5
fourthResults = predictTrainingSet(k,3) #230 5?


probability = np.zeros(shape=(len(fourthResults),1))
predictions = np.zeros(shape=(len(fourthResults),1))
for i in range(0, len(allgenInstances)):
    if(firstResults[i] == 1):
        probability[i] += 40
    if(secondResults[i] == 1):
        probability[i] += 50
    if(thirdResults[i] == 1):
        probability[i] += 25
    if(fourthResults[i] == 1):
        probability[i] += 50
    if(probability[i] > 100):
        probability[i] = 100
    if(probability[i] >= 50):
        predictions[i] = 1
    elif(probability[i] < 50):
        predictions[i] = 0
    if(probability[i] >= 50):#translate probability to mean probability that prediction is right
        probability[i] = 2*(probability[i]-50)
    elif(probability[i] < 50):
        probability[i] = 100-(2*probability[i])

f = open("individual1_pred2.csv", "w")
for x in zip(probability,predictions):
    f.write("{},{}\n".format(x[0][0], x[1][0]))
f.close()

f = open("individual1_actual_output.csv", "w")
for x in zip(firstResults,secondResults,thirdResults,fourthResults):
    f.write("{},{},{},{}\n".format(x[0], x[1], x[2], x[3]))
f.close()
