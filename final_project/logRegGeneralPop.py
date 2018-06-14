import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import auc, roc_curve, roc_auc_score
from datetime import datetime
from dateutil import tz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

numColumns = 8
instance = np.zeros(shape=(7, numColumns))
chunks = np.zeros(shape=(1, numColumns))
chunksList = []
sub1HypoChunkList = []
sub4HypoChunkList = []
sub6HypoChunkList = []
sub9HypoChunkList = []
sub2HypoChunkList = [] # for verification subject
X = []
y = []
SUB1 = np.zeros(shape=(1, numColumns))
SUB1HYPO = np.zeros(shape=(1,1)) # this is whether there will be a hypo in 30 minutes or not (what we're trying to predict)
SUB1INDICES = np.zeros(shape=(1,1)) # this will hold the indices for subject 1

print "Reading files . . ."

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

counter = 0 # must get to at least equaling 6, which is a minimum of 7 instances for a 30 minute period

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
	else:
		if SUB1INDICES[i][0] + 1 != SUB1INDICES[i+1][0]:
			temp = np.zeros(shape=(1, numColumns))
			chunks = temp
			counter = 0
			temp2 = np.zeros(shape=(1,1))

counter = 0
for i in range(0, len(chunksList)):
    temp2 = np.zeros(shape=(7,1))
    sub1HypoChunk = temp2
    temp = np.zeros(shape=(7, numColumns))
    instance = temp
    for j in range(0, len(chunksList[i])):

        if j<7:
            instance[j][:] = chunksList[i][j][:]
            sub1HypoChunk[j][0] = int(sub1HypoChunkList[i][j][0])
        else:
            # for k in range(0, np.size(instance, 0)):
            #     for l in range(0, 8):
            X.append(np.ravel(instance))
            y.append(sub1HypoChunk[6])
            tempNum = j%7-1
            instance = np.delete(instance, tempNum, 0)
            sub1HypoChunk = np.delete(sub1HypoChunk, tempNum, 0)
            instance = np.vstack((instance, chunksList[i][j][:]))
            sub1HypoChunk = np.vstack((sub1HypoChunk, int(sub1HypoChunkList[i][j][0])))


print "Subject_1"



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
	else:
		if SUB4INDICES[i][0] + 1 != SUB4INDICES[i+1][0]:
			temp = np.zeros(shape=(1, numColumns))
			chunks = temp
			counter = 0
			temp2 = np.zeros(shape=(1,1))

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
            X.append(np.ravel(instance))
            y.append(sub1HypoChunk[6])
            tempNum = j%7-1
            instance = np.delete(instance, tempNum, 0)
            sub4HypoChunk = np.delete(sub4HypoChunk, tempNum, 0)
            instance = np.vstack((instance, chunksList[i][j][:]))
            sub4HypoChunk = np.vstack((sub4HypoChunk, sub4HypoChunkList[i][j][0]))

print "Subject_4"



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
	else:
		if SUB6INDICES[i][0] + 1 != SUB6INDICES[i+1][0]:
			temp = np.zeros(shape=(1, numColumns))
			chunks = temp
			counter = 0
			temp2 = np.zeros(shape=(1,1))

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
            X.append(np.ravel(instance))
            y.append(sub1HypoChunk[6])
            tempNum = j%7-1
            instance = np.delete(instance, tempNum, 0)
            sub6HypoChunk = np.delete(sub6HypoChunk, tempNum, 0)
            instance = np.vstack((instance, chunksList[i][j][:]))
            sub6HypoChunk = np.vstack((sub6HypoChunk, sub6HypoChunkList[i][j][0]))

print "Subject_6"


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
xAccuracy = [] # store accuracies for our predictions
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
	else:
		if SUB9INDICES[i][0] + 1 != SUB9INDICES[i+1][0]:
			temp = np.zeros(shape=(1, numColumns))
			chunks = temp
			counter = 0
			temp2 = np.zeros(shape=(1,1))

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
            X.append(np.ravel(instance))
            y.append(sub1HypoChunk[6])
            tempNum = j%7-1
            instance = np.delete(instance, tempNum, 0)
            sub9HypoChunk = np.delete(sub9HypoChunk, tempNum, 0)
            instance = np.vstack((instance, chunksList[i][j][:]))
            sub9HypoChunk = np.vstack((sub9HypoChunk, sub9HypoChunkList[i][j][0]))

print "Subject_9"
print "Done.\n"

print "Calculating logistic regression . . . \n"

X = np.array(X)
y = np.ravel(np.array(y))
# x, y, z = X.shape
# X = X.reshape(x, y*z)

print "Shape of X: ", X.shape
print "Shape of y: ", y.shape

print "fitting model"
logreg = LogisticRegression()
X = StandardScaler().fit_transform(X)
logreg.fit(X, y)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# X_train, X_test, y_train, y_test = train_test_split(StandardScaler().fit_transform(X), y, test_size=0.33, random_state=42)
#
# print "num 1s genPop, y train, y test"
#
# n = np.array(np.where(y == 1))
# print n.shape
#
# n1 = np.array(np.where(y_train == 1))
# print n1.shape
#
# n1 = np.array(np.where(y_test == 1))
# print n1.shape
#
#
# logreg.fit(X_train, y_train)
# y_pred = logreg.predict(X_test)
# print("Accuracy: ", logreg.score(X_test, y_test))
# print "Confusion matrix: "
# print "[[ TN  FP]"
# print " [ FN  TP]]"
# confusion_matrix = confusion_matrix(y_test, y_pred)
# print(confusion_matrix)
# print(classification_report(y_test, y_pred))
# logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
# print "ROC score: ", logit_roc_auc
#
# # print "\n\n Model coefs: "
# # print logreg.coef_
# print "\n\n"

FTEST = np.zeros(shape=(1, numColumns)) # used to break each into instances
TEMPFTESTLIST = [] # used to get all the data from gen test instances file
FTESTLIST = [] # actual list containing all FTESTs
with open('general_test_instances.csv','r') as f:
    for line in f:
        featureNum = 0
        lineWords = []
        averageValue = []
        for word in line.split(','):
            lineWords.append(float(word))
            featureNum += 1
            # print lineWords

        TEMPFTESTLIST.append(lineWords)

for i  in range(0, np.size(TEMPFTESTLIST, 0)):
	temp = np.zeros(shape=(1, numColumns))
	FTEST = temp
	for j in range(0, 7):
		tempRow = []
		for k in range(0, numColumns):
			tempRow.append(TEMPFTESTLIST[i][j+7*k])
		# print tempRow
		# print np.shape(tempRow)
		if j == 0:
			FTEST[0] = tempRow
		else:
			FTEST = np.vstack((FTEST, tempRow))
	FTESTLIST.append(FTEST)

print np.shape(FTESTLIST)

print "Starting Subject general pop Test"
print 32*"-"
predictX = []
xAccuracy = []

X_test = []

for i in range(0, np.size(FTESTLIST, 0)): # np.size(FTESTLIST, 0)
    X_test.append(np.ravel(FTESTLIST[i]))

y_pred = logreg.predict(X_test)
y_score = logreg.decision_function(X_test)

f3 = open('general_pred3.csv', 'w')
for i in range(0, len(y_pred)):
	stringVar = str(y_score[i]) + "," + str(int(y_pred[i])) + "\n"
	f3.write(stringVar)
f3.close()
