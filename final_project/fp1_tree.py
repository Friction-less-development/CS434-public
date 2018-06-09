import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import auc, roc_auc_score
#How to Run: python fp1_tree.py
#Purpose of program: Final project for General Population Model Test using decision tree as our model 

# Sources: 
#           http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#           https://stackoverflow.com/questions/42757892/how-to-use-warm-start

numColumns = 8
instance = np.zeros(shape=(7, numColumns))
chunks = np.zeros(shape=(1, numColumns)) 
chunksList = []
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
classWeights = compute_class_weight("balanced", [0., 1.], np.ravel(SUB1HYPO))
# print classWeights
classWeights[0] = classWeights[0]/126. # can only put class weight of 0 in, so divide it by the wiehgt of class weight of 1, more or less.
counter = 0 # must get to at least equaling 6, which is a minimum of 7 instances for a 30 minute period
subForest = RandomForestClassifier(criterion="entropy", max_depth=7, max_features=6, random_state=0, warm_start=True, bootstrap=False, class_weight={0.:classWeights[0]}) # class_weight={0.:classWeights[0]}

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
            temp = np.zeros(shape=(1, numColumns))
            chunks = temp
            counter = 0
            temp2 = np.zeros(shape=(1,1))
            sub1HypoChunk = temp2
        elif SUB1INDICES[i][0] - 1 != SUB1INDICES[i-1][0] and counter >= 6:
            sub1HypoChunkList.append(sub1HypoChunk)
            chunksList.append(chunks)
            temp = np.zeros(shape=(1, numColumns))
            chunks = temp
            temp2 = np.zeros(shape=(1,1))
            sub1HypoChunk = temp2
            counter = 0

# print np.shape(chunksList)
# print len(chunksList)
# print np.shape(sub1HypoChunkList[0])
counter = 0
for i in range(0, len(chunksList)):
    temp2 = np.zeros(shape=(7,1))
    sub1HypoChunk = temp2
    temp = np.zeros(shape=(7, numColumns))
    instance = temp
    for j in range(0, len(chunksList[i])):
        # if j==len(chunksList[i])-1:
        #     print i
            # print j
            # print instance
            # print sub1HypoChunk
            # print 5*"-"
        if j<7:
            instance[j][:] = chunksList[i][j][:]
            sub1HypoChunk[j][0] = int(sub1HypoChunkList[i][j][0])
        else:
            # for k in range(0, np.size(instance, 0)):
            #     for l in range(0, 8):
            subForest.fit(instance, np.ravel(sub1HypoChunk))
            subForest.n_estimators += 1
            tempNum = j%7-1
            instance = np.delete(instance, tempNum, 0)
            sub1HypoChunk = np.delete(sub1HypoChunk, tempNum, 0)
            instance = np.vstack((instance, chunksList[i][j][:]))
            sub1HypoChunk = np.vstack((sub1HypoChunk, int(sub1HypoChunkList[i][j][0])))

# print sub1HypoChunkList[0][349]
# print sub1HypoChunkList[0][348]
# print np.shape(sub1HypoChunkList)
# print np.shape(instance)
# print np.shape(chunks)
print "\n"
print "Subject_1"
# print np.shape(SUB1)
# print np.shape(SUB1HYPO)
# print np.shape(SUB1INDICES)

# subForest.fit(SUB1, np.ravel(SUB1HYPO))
# print subForest.feature_importances_
importances = subForest.feature_importances_
std = np.std([tree.feature_importances_ for tree in subForest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
for f in range(SUB1.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
print "\n"
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(SUB1.shape[1]), importances[indices],
#        color="r", yerr=std[indices], align="center")
# plt.xticks(range(SUB1.shape[1]), indices)
# plt.xlim([-1, SUB1.shape[1]])
# plt.savefig('FeatureImportanceSub1.png') # after 1 subject




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
templ = []
chunksList = templ
sub4HypoChunk = np.zeros(shape=(1,1))
counter = 0
tempc = np.zeros(shape=(1, numColumns)) 
chunks = tempc
for i in range (0, np.size(SUB4,0)):
    if counter == 0:
        sub4HypoChunk[0] = SUB4HYPO[i][:]
        chunks[0] = SUB4[i][:]
        counter += 1
    else:
        chunks = np.vstack((chunks, SUB4[i][:]))
        sub4HypoChunk = np.vstack((sub4HypoChunk, SUB4HYPO[i][:]))
        counter += 1
    if i != 0:
        if SUB4INDICES[i][0] - 1 != SUB4INDICES[i-1][0] and counter < 6:
            temp = np.zeros(shape=(1, numColumns))
            chunks = temp
            counter = 0
            temp2 = np.zeros(shape=(1,1))
            sub4HypoChunk = temp2
        elif SUB4INDICES[i][0] - 1 != SUB4INDICES[i-1][0] and counter >= 6:
            sub4HypoChunkList.append(sub4HypoChunk)
            chunksList.append(chunks)
            temp = np.zeros(shape=(1, numColumns))
            chunks = temp
            temp2 = np.zeros(shape=(1,1))
            sub4HypoChunk = temp2
            counter = 0

# print np.shape(chunksList)
# print len(chunksList)
# print np.shape(sub4HypoChunkList[0])
counter = 0
for i in range(0, len(chunksList)):
    temp2 = np.zeros(shape=(7,1))
    sub4HypoChunk = temp2
    temp = np.zeros(shape=(7, numColumns))
    instance = temp
    for j in range(0, len(chunksList[i])):
        # if j==len(chunksList[i])-1:
        #     print i
            # print j
            # print instance
            # print sub4HypoChunk
            # print 5*"-"
        if j<7:
            instance[j][:] = chunksList[i][j][:]
            sub4HypoChunk[j][0] = sub4HypoChunkList[i][j][0]
        else:
            # for k in range(0, np.size(instance, 0)):
            #     for l in range(0, 8):
            subForest.fit(instance, np.ravel(sub4HypoChunk))
            subForest.n_estimators += 1
            tempNum = j%7-1
            instance = np.delete(instance, tempNum, 0)
            sub4HypoChunk = np.delete(sub4HypoChunk, tempNum, 0)
            instance = np.vstack((instance, chunksList[i][j][:]))
            sub4HypoChunk = np.vstack((sub4HypoChunk, sub4HypoChunkList[i][j][0]))
print "\n"
print "Subject_4"
# print np.shape(SUB4)
# print np.shape(SUB4HYPO)
# print np.shape(SUB4INDICES)

# subForest.fit(SUB4, np.ravel(SUB4HYPO))
# print subForest.feature_importances_
importances = subForest.feature_importances_
std = np.std([tree.feature_importances_ for tree in subForest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
for f in range(SUB4.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
print "\n"
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(SUB4.shape[1]), importances[indices],
#        color="r", yerr=std[indices], align="center")
# plt.xticks(range(SUB4.shape[1]), indices)
# plt.xlim([-1, SUB4.shape[1]])
# plt.savefig('FeatureImportanceSub2.png') # after 2 subjects

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
templ = []
chunksList = templ
sub6HypoChunk = np.zeros(shape=(1,1))
counter = 0
tempc = np.zeros(shape=(1, numColumns)) 
chunks = tempc
for i in range (0, np.size(SUB6,0)):
    if counter == 0:
        sub6HypoChunk[0] = SUB6HYPO[i][:]
        chunks[0] = SUB6[i][:]
        counter += 1
    else:
        chunks = np.vstack((chunks, SUB6[i][:]))
        sub6HypoChunk = np.vstack((sub6HypoChunk, SUB6HYPO[i][:]))
        counter += 1
    if i != 0:
        if SUB6INDICES[i][0] - 1 != SUB6INDICES[i-1][0] and counter < 6:
            temp = np.zeros(shape=(1, numColumns))
            chunks = temp
            counter = 0
            temp2 = np.zeros(shape=(1,1))
            sub6HypoChunk = temp2
        elif SUB6INDICES[i][0] - 1 != SUB6INDICES[i-1][0] and counter >= 6:
            sub6HypoChunkList.append(sub6HypoChunk)
            chunksList.append(chunks)
            temp = np.zeros(shape=(1, numColumns))
            chunks = temp
            temp2 = np.zeros(shape=(1,1))
            sub6HypoChunk = temp2
            counter = 0

# print np.shape(chunksList)
# print len(chunksList)
# print np.shape(sub6HypoChunkList[0])
counter = 0
for i in range(0, len(chunksList)):
    temp2 = np.zeros(shape=(7,1))
    sub6HypoChunk = temp2
    temp = np.zeros(shape=(7, numColumns))
    instance = temp
    for j in range(0, len(chunksList[i])):
        # if j==len(chunksList[i])-1:
        #     print i
            # print j
            # print instance
            # print sub6HypoChunk
            # print 5*"-"
        if j<7:
            instance[j][:] = chunksList[i][j][:]
            sub6HypoChunk[j][0] = sub6HypoChunkList[i][j][0]
        else:
            # for k in range(0, np.size(instance, 0)):
            #     for l in range(0, 8):
            subForest.fit(instance, np.ravel(sub6HypoChunk))
            subForest.n_estimators += 1
            tempNum = j%7-1
            instance = np.delete(instance, tempNum, 0)
            sub6HypoChunk = np.delete(sub6HypoChunk, tempNum, 0)
            instance = np.vstack((instance, chunksList[i][j][:]))
            sub6HypoChunk = np.vstack((sub6HypoChunk, sub6HypoChunkList[i][j][0]))
print "\n"
print "Subject_6"
# print np.shape(SUB6)
# print np.shape(SUB6HYPO)
# print np.shape(SUB6INDICES)

importances = subForest.feature_importances_
std = np.std([tree.feature_importances_ for tree in subForest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
for f in range(SUB6.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
print "\n"

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
templ = []
chunksList = templ
sub9HypoChunk = np.zeros(shape=(1,1))
counter = 0
tempc = np.zeros(shape=(1, numColumns)) 
chunks = tempc
for i in range (0, np.size(SUB9,0)):
    if counter == 0:
        sub9HypoChunk[0] = SUB9HYPO[i][:]
        chunks[0] = SUB9[i][:]
        counter += 1
    else:
        chunks = np.vstack((chunks, SUB9[i][:]))
        sub9HypoChunk = np.vstack((sub9HypoChunk, SUB9HYPO[i][:]))
        counter += 1
    if i != 0:
        if SUB9INDICES[i][0] - 1 != SUB9INDICES[i-1][0] and counter < 6:
            temp = np.zeros(shape=(1, numColumns))
            chunks = temp
            counter = 0
            temp2 = np.zeros(shape=(1,1))
            sub9HypoChunk = temp2
        elif SUB9INDICES[i][0] - 1 != SUB9INDICES[i-1][0] and counter >= 6:
            sub9HypoChunkList.append(sub9HypoChunk)
            chunksList.append(chunks)
            temp = np.zeros(shape=(1, numColumns))
            chunks = temp
            temp2 = np.zeros(shape=(1,1))
            sub9HypoChunk = temp2
            counter = 0

# print np.shape(chunksList)
# print len(chunksList)
# print np.shape(sub9HypoChunkList[0])
counter = 0
for i in range(0, len(chunksList)):
    temp2 = np.zeros(shape=(7,1))
    sub9HypoChunk = temp2
    temp = np.zeros(shape=(7, numColumns))
    instance = temp
    for j in range(0, len(chunksList[i])):
        # if j==len(chunksList[i])-1:
        #     print i
            # print j
            # print instance
            # print sub9HypoChunk
            # print 5*"-"
        if j<7:
            instance[j][:] = chunksList[i][j][:]
            sub9HypoChunk[j][0] = sub9HypoChunkList[i][j][0]
        else:
            # for k in range(0, np.size(instance, 0)):
            #     for l in range(0, 8):
            subForest.fit(instance, np.ravel(sub9HypoChunk))
            subForest.n_estimators += 1
            tempNum = j%7-1
            instance = np.delete(instance, tempNum, 0)
            sub9HypoChunk = np.delete(sub9HypoChunk, tempNum, 0)
            instance = np.vstack((instance, chunksList[i][j][:]))
            sub9HypoChunk = np.vstack((sub9HypoChunk, sub9HypoChunkList[i][j][0]))
print "\n"
print "Subject_9"
# print np.shape(SUB9)
# print np.shape(SUB9HYPO)
# print np.shape(SUB9INDICES)

importances = subForest.feature_importances_
std = np.std([tree.feature_importances_ for tree in subForest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]
for f in range(SUB9.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
print "\n"

TEST1 = np.zeros(shape=(1, numColumns)) 
firstLine = True
with open('sampleinstance_1.csv','r') as f:
    for line in f:
        featureNum = 0
        lineWords = []
        averageValue = []
        for word in line.split(','):
            if featureNum > 0 and featureNum < 9:
           	  lineWords.append(float(word))
           	  featureNum += 1
            else:
              featureNum += 1
            # print lineWords

        if firstLine:
        	TEST1[0] = lineWords
        	firstLine = False
        else:
        	TEST1 = np.vstack((TEST1, lineWords))

print "Sample 1"
# print np.shape(TEST1)
print subForest.predict(TEST1)
print subForest.predict_proba(TEST1)
print "\n"

TEST2 = np.zeros(shape=(1, numColumns)) 
firstLine = True
with open('sampleinstance_2.csv','r') as f:
    for line in f:
        featureNum = 0
        lineWords = []
        averageValue = []
        for word in line.split(','):
            if featureNum > 0 and featureNum < 9:
           	  lineWords.append(float(word))
           	  featureNum += 1
            else:
              featureNum += 1
            # print lineWords

        if firstLine:
        	TEST2[0] = lineWords
        	firstLine = False
        else:
        	TEST2 = np.vstack((TEST2, lineWords))

print "Sample 2"
# print np.shape(TEST2)
print subForest.predict(TEST2)
print subForest.predict_proba(TEST2)
print "\n"

TEST3 = np.zeros(shape=(1, numColumns)) 
firstLine = True
with open('sampleinstance_3.csv','r') as f:
    for line in f:
        featureNum = 0
        lineWords = []
        averageValue = []
        for word in line.split(','):
            if featureNum > 0 and featureNum < 9:
           	  lineWords.append(float(word))
           	  featureNum += 1
            else:
              featureNum += 1
            # print lineWords

        if firstLine:
        	TEST3[0] = lineWords
        	firstLine = False
        else:
        	TEST3 = np.vstack((TEST3, lineWords))

print "Sample 3"
# print np.shape(TEST3)
print subForest.predict(TEST3)
print subForest.predict_proba(TEST3)
print "\n"

TEST4 = np.zeros(shape=(1, numColumns)) 
firstLine = True
with open('sampleinstance_4.csv','r') as f:
    for line in f:
        featureNum = 0
        lineWords = []
        averageValue = []
        for word in line.split(','):
            if featureNum > 0 and featureNum < 9:
           	  lineWords.append(float(word))
           	  featureNum += 1
            else:
              featureNum += 1
            # print lineWords

        if firstLine:
        	TEST4[0] = lineWords
        	firstLine = False
        else:
        	TEST4 = np.vstack((TEST4, lineWords))

print "Sample 4"
# print np.shape(TEST4)
print subForest.predict(TEST4)
print subForest.predict_proba(TEST4)
print "\n"

TEST5 = np.zeros(shape=(1, numColumns)) 
firstLine = True
with open('sampleinstance_5.csv','r') as f:
    for line in f:
        featureNum = 0
        lineWords = []
        averageValue = []
        for word in line.split(','):
            if featureNum > 0 and featureNum < 9:
           	  lineWords.append(float(word))
           	  featureNum += 1
            else:
              featureNum += 1
            # print lineWords

        if firstLine:
        	TEST5[0] = lineWords
        	firstLine = False
        else:
        	TEST5 = np.vstack((TEST5, lineWords))

print "Sample 5"
# print np.shape(TEST5)
print subForest.predict(TEST5)
print subForest.predict_proba(TEST5)
print "\n"

# Below is if we want to use subject 2 part 1 as a verifier instead of sample instances
SUB2 = np.zeros(shape=(1, 8))
SUB2HYPO = np.zeros(shape=(1,1)) # this is whether there will be a hypo in 30 minutes or not (what we're trying to predict)
SUB2INDICES = np.zeros(shape=(1,1)) # this will hold the indices for subject 9

firstLine = True
with open('Subject_2_part1.csv','r') as f:
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
        	SUB2[0] = lineWords
        	SUB2HYPO[0] = averageValue
        	firstLine = False
        else:
        	SUB2 = np.vstack((SUB2, lineWords))
        	SUB2HYPO = np.vstack((SUB2HYPO, averageValue))

firstLine = True
with open('list2_part1.csv','r') as f:
    for line in f:
    	lineWords = []
        for word in line.split(','):
        	lineWords.append(int(word))

        if firstLine:
        	SUB2INDICES[0] = lineWords
        	firstLine = False
        else:
        	SUB2INDICES = np.vstack((SUB2INDICES, lineWords))

templ = []
chunksList = templ
sub2HypoChunk = np.zeros(shape=(1,1))
counter = 0
tempc = np.zeros(shape=(1, numColumns)) 
chunks = tempc
for i in range (0, np.size(SUB2,0)):
    if counter == 0:
        sub2HypoChunk[0] = SUB2HYPO[i][:]
        chunks[0] = SUB2[i][:]
        counter += 1
    else:
        chunks = np.vstack((chunks, SUB2[i][:]))
        sub2HypoChunk = np.vstack((sub2HypoChunk, SUB2HYPO[i][:]))
        counter += 1
    if i != 0:
        if SUB2INDICES[i][0] - 1 != SUB2INDICES[i-1][0] and counter < 6:
            temp = np.zeros(shape=(1, numColumns))
            chunks = temp
            counter = 0
            temp2 = np.zeros(shape=(1,1))
            sub2HypoChunk = temp2
        elif SUB2INDICES[i][0] - 1 != SUB2INDICES[i-1][0] and counter >= 6:
            sub2HypoChunkList.append(sub2HypoChunk)
            chunksList.append(chunks)
            temp = np.zeros(shape=(1, numColumns))
            chunks = temp
            temp2 = np.zeros(shape=(1,1))
            sub2HypoChunk = temp2
            counter = 0
print "Subject 2"
counter = 0
numberCorrect = 5
y = [0, 0, 0, 1, 0] # actual
X = [0, 0, 0, 1, 0] # predicted
# y = [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]
# X = [0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1]
numTotal = 5
falseP = 0 # false positives
falseN = 0 # false negatives
correctP = 1 # correct positives aka correct when hypo event will happen
correctN = 4 # correct negatives aka correct when there isn't a hypo event
for i in range(0, len(chunksList)):
    temp2 = np.zeros(shape=(7,1))
    sub2HypoChunk = temp2
    temp = np.zeros(shape=(7, numColumns))
    instance = temp
    for j in range(0, len(chunksList[i])):
        # if j==len(chunksList[i])-1:
        #     print i
            # print j
            # print instance
            # print sub9HypoChunk
            # print 5*"-"
        if j<7:
            instance[j][:] = chunksList[i][j][:]
            sub2HypoChunk[j][0] = sub2HypoChunkList[i][j][0]
        else:
            # for k in range(0, np.size(instance, 0)):
            #     for l in range(0, 8):
			if counter < 9:
				isRight = False
				oneExists = False
				predictForest = subForest.predict(instance)
				for k in range(0, 7):
					if sub2HypoChunk[k][0] > 0.5:
						oneExists = True
				if oneExists:
					y.append(1)
					for k in range(0, len(predictForest)):
						if predictForest[k] > 0:
							isRight = True
					if isRight:
						X.append(1)
					else:
						X.append(0)
				else:
					y.append(0)
					isRight = True
					for k in range(0, len(predictForest)):
						if predictForest[k] > 0.:
							isRight = False
					if isRight:
						X.append(0)
					else:
						X.append(1)
				if isRight:
					numberCorrect += 1
					if oneExists:
						correctP += 1
					else:
						correctN += 1
				else:
					if oneExists:
						falseN += 1
					else:
						falseP += 1
				print predictForest
				print sub2HypoChunk
				print "Correct: ", isRight
				print subForest.predict_proba(instance)
				numTotal += 1
				print "\n"
			tempNum = j%7-1
			instance = np.delete(instance, tempNum, 0)
			sub2HypoChunk = np.delete(sub2HypoChunk, tempNum, 0)
			instance = np.vstack((instance, chunksList[i][j][:]))
			sub2HypoChunk = np.vstack((sub2HypoChunk, sub2HypoChunkList[i][j][0]))
			counter += 1

print "number correct: ", numberCorrect
print "correct positives: ", correctP # a
print "false negatives: ", falseN # b
print "false positives: ", falseP # c
print "correct negatives: ", correctN # d
# tpRate = correctP/(correctP+falseN) # true positive rate
# fpRate = falseP/(falseP + correctN) # false positive rate
# print y
# print X
print "Area under ROC: ", roc_auc_score(y, X)
print "total: ", numTotal
