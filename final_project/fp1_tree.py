import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier

#How to Run: python fp1_tree.py
#Purpose of program: Final project for General Population Model Test using decision tree as our model 

# Sources: 
#           http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#           https://stackoverflow.com/questions/42757892/how-to-use-warm-start


instance = np.zeros(shape=(7, 8)) # normally 8, might want to consider 7 and skip the first time column, because of time difference might be triggering it to think it is a good indicator
chunks = np.zeros(shape=(1, 8)) # normally 8, might want to consider 7 and skip the first time column, because of time difference might be triggering it to think it is a good indicator
chunksList = []
sub1HypoChunkList = []
SUB1 = np.zeros(shape=(1, 8)) # normally 8, might want to consider 7 and skip the first time column, because of time difference might be triggering it to think it is a good indicator
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

counter = 0 # must get to at least equaling 6, with a minimum of 7 instances for a 30 minute period

sub1Forest = RandomForestClassifier(criterion="entropy", max_depth=7, random_state=1, warm_start=True)

# have a problem, need to get SUB1HYPO, aka the training 0 and 1s, to line up with these chunks, will have to probably create a list of SUB1HYPO chunks that correspond to each respective chunk
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
    if i != 0:
        if SUB1INDICES[i][0] - 1 != SUB1INDICES[i-1][0] and counter < 6:
            temp = np.zeros(shape=(1, 8))
            chunks = temp
            counter = 0
            temp2 = np.zeros(shape=(1,1))
            sub1HypoChunk = temp2
        elif SUB1INDICES[i][0] - 1 != SUB1INDICES[i-1][0] and counter >= 6:
            sub1HypoChunkList.append(sub1HypoChunk)
            chunksList.append(chunks)
            temp = np.zeros(shape=(1, 8))
            chunks = temp
            temp2 = np.zeros(shape=(1,1))
            sub1HypoChunk = temp2
            counter = 0

# print np.shape(chunksList)
# print len(chunksList)
counter = 0
for i in range(0, len(chunksList)):
    temp2 = np.zeros(shape=(7,1))
    sub1HypoChunk = temp2
    temp = np.zeros(shape=(7, 8))
    instance = temp
    for j in range(0, len(chunksList[i])):
        if j==len(chunksList[i])-1:
            print i
            # print j
            # print instance
            # print sub1HypoChunk
            # print 5*"-"
        if j<7:
            instance[j][:] = chunksList[i][j][:]
            sub1HypoChunk[j][0] = sub1HypoChunkList[i][j][0]
        else:
            # for k in range(0, np.size(instance, 0)):
            #     for l in range(0, 8):
            sub1Forest.fit(instance, np.ravel(sub1HypoChunk))
            sub1Forest.n_estimators += 1
            tempNum = j%7-1
            instance = np.delete(instance, tempNum, 0)
            sub1HypoChunk = np.delete(sub1HypoChunk, tempNum, 0)
            instance = np.vstack((instance, chunksList[i][j][:]))
            sub1HypoChunk = np.vstack((sub1HypoChunk, sub1HypoChunkList[i][j][0]))

# print sub1HypoChunkList[0][349]
# print sub1HypoChunkList[0][348]
# print np.shape(sub1HypoChunkList)
# print np.shape(instance)
# print np.shape(chunks)
print "\n"
print "Subject_1"
print np.shape(SUB1)
print np.shape(SUB1HYPO)
print np.shape(SUB1INDICES)

# sub1Forest.fit(SUB1, np.ravel(SUB1HYPO))
# print sub1Forest.feature_importances_
importances = sub1Forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in sub1Forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
for f in range(SUB1.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(SUB1.shape[1]), importances[indices],
#        color="r", yerr=std[indices], align="center")
# plt.xticks(range(SUB1.shape[1]), indices)
# plt.xlim([-1, SUB1.shape[1]])
# plt.savefig('FeatureImportance.png')




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