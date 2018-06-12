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
#How to Run: python fp1_tree.py
#Purpose of program: Final project for Individual Test Subject 7 using decision tree as our model

# Sources: 
#           http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#           https://stackoverflow.com/questions/42757892/how-to-use-warm-start
#			https://stackoverflow.com/questions/31159157/different-result-with-roc-auc-score-and-auc
#           https://stackoverflow.com/questions/4770297/convert-utc-datetime-string-to-local-datetime-with-python


from_zone = tz.tzutc()
to_zone = tz.tzlocal()
numColumns = 9
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
with open('Subject_7_part1.csv','r') as f:
    for line in f:
        featureNum = 0
        lineWords = []
        averageValue = []
        for word in line.split(','):
            if featureNum > 0 and featureNum < 9:
           	  lineWords.append(float(word))
           	  featureNum += 1
            elif featureNum == 0:
                utc = datetime.strptime(word, '%Y-%m-%dT%H:%M:%SZ')
                lineWords.append(utc.hour)
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
with open('list_7_part1.csv','r') as f:
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
subForest = RandomForestClassifier(criterion="entropy", max_depth=7, max_features=6, random_state=0, warm_start=False, bootstrap=False, class_weight={0.:classWeights[0]}) # class_weight={0.:classWeights[0]}

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
for i in range(0, len(chunksList)): # len(chunksList)
    temp2 = np.zeros(shape=(7,1))
    sub1HypoChunk = temp2
    temp = np.zeros(shape=(7, numColumns))
    instance = temp
    for j in range(0, len(chunksList[i])):
        if j%10 == 0:
            print i
            print j
            print len(chunksList[i])
            print len(chunksList)
            print "\n"
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

print "\n"
print "Subject_7"
# print np.shape(SUB1)
# print np.shape(SUB1HYPO)
# print np.shape(SUB1INDICES)

# subForest.fit(SUB1, np.ravel(SUB1HYPO))
# print subForest.feature_importances_
# importances = subForest.feature_importances_
# std = np.std([tree.feature_importances_ for tree in subForest.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]
# for f in range(SUB1.shape[1]):
#     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
# print "\n"

FTEST = np.zeros(shape=(1, numColumns)) # used to break each into instances
TEMPFTESTLIST = [] # used to get all the data from gen test instances file
FTESTLIST = [] # actual list containing all FTESTs
with open('subject7_instances.csv','r') as f:
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

# print FTESTLIST[0]
print np.shape(FTESTLIST)
# # print "\n"
# # print FTESTLIST[0][0]
# # print FTESTLIST[0][1]
print "Starting Subject 7 Part 2 Test"
print 32*"-"
predictX = []
xAccuracy = []
for i in range(0, np.size(FTESTLIST, 0)): # np.size(FTESTLIST, 0)
	isOne = False
	predictForest = subForest.predict(FTESTLIST[i])
	predictForestProb = subForest.predict_proba(FTESTLIST[i])
	xAccuracy.append(np.amax(predictForestProb))
	for k in range(0, len(predictForest)):
		if predictForest[k] > 0.5:
			isOne = True
	if isOne:
		predictX.append(1)
	else:
		predictX.append(0)
	if i%10 == 0:
		print i

f3 = open('individual1_pred1.csv', 'w')
for i in range(0, len(predictX)):
	stringVar = str(xAccuracy[i]) + "," + str(predictX[i]) + "\n"
	f3.write(stringVar)
f3.close()

exit() # comment out to run sample tests and verification tests on subject 2 data

# END OF TEST CODE

TEST1 = np.zeros(shape=(1, numColumns)) 
firstLine = True
with open('sampleinstance_1.csv','r') as f:
    for line in f:
        featureNum = 0
        lineWords = []
        averageValue = []
        for word in line.split(','):
            if featureNum < 9:
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
print np.shape(TEST1)
print TEST1
print subForest.predict(TEST1)
sampleInstProb = subForest.predict_proba(TEST1)
xAccuracy.append(np.amax(sampleInstProb))
# print subForest.predict_proba(TEST1)
print "\n"

TEST2 = np.zeros(shape=(1, numColumns)) 
firstLine = True
with open('sampleinstance_2.csv','r') as f:
    for line in f:
        featureNum = 0
        lineWords = []
        averageValue = []
        for word in line.split(','):
            if featureNum < 9:
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
sampleInstProb = subForest.predict_proba(TEST2)
xAccuracy.append(np.amax(sampleInstProb))
# print subForest.predict_proba(TEST2)
print "\n"

TEST3 = np.zeros(shape=(1, numColumns)) 
firstLine = True
with open('sampleinstance_3.csv','r') as f:
    for line in f:
        featureNum = 0
        lineWords = []
        averageValue = []
        for word in line.split(','):
            if featureNum < 9:
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
sampleInstProb = subForest.predict_proba(TEST3)
xAccuracy.append(np.amax(sampleInstProb))
# print subForest.predict_proba(TEST3)
print "\n"

TEST4 = np.zeros(shape=(1, numColumns)) 
firstLine = True
with open('sampleinstance_4.csv','r') as f:
    for line in f:
        featureNum = 0
        lineWords = []
        averageValue = []
        for word in line.split(','):
            if featureNum < 9:
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
sampleInstProb = subForest.predict_proba(TEST4)
xAccuracy.append(np.amax(sampleInstProb))
# print subForest.predict_proba(TEST4)
print "\n"

TEST5 = np.zeros(shape=(1, numColumns)) 
firstLine = True
with open('sampleinstance_5.csv','r') as f:
    for line in f:
        featureNum = 0
        lineWords = []
        averageValue = []
        for word in line.split(','):
            if featureNum < 9:
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
sampleInstProb = subForest.predict_proba(TEST5)
xAccuracy.append(np.amax(sampleInstProb))
# print subForest.predict_proba(TEST5)
print "\n"

exit() # used to stop it from going to some odd tests on subject 1
# Below is if we want to use subject 2 part 1 as a verifier instead of sample instances
SUB2 = np.zeros(shape=(1, 8))
SUB2HYPO = np.zeros(shape=(1,1)) # this is whether there will be a hypo in 30 minutes or not (what we're trying to predict)
SUB2INDICES = np.zeros(shape=(1,1)) # this will hold the indices for subject 9

firstLine = True
with open('Subject_1.csv','r') as f:
    for line in f:
        featureNum = 0
        lineWords = []
        averageValue = []
        for word in line.split(','):
            if featureNum < 9:
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
with open('list1.csv','r') as f:
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
print "Subject 1"
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
numInstances = 50 # number of instances from Subject 2 to run, 295 takes quite a while to run
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
			if counter < numInstances:
				isRight = False
				oneExists = False
				predictForest = subForest.predict(instance)
				predictForestProb = subForest.predict_proba(instance)
				xAccuracy.append(np.amax(predictForestProb))
				for k in range(0, 7):
					if sub2HypoChunk[k][0] > 0.5:
						oneExists = True
				if oneExists:
					y.append(1)
					for k in range(0, len(predictForest)):
						if predictForest[k] > 0.5:
							isRight = True
					if isRight:
						X.append(1)
					else:
						X.append(0)
				else:
					y.append(0)
					isRight = True
					for k in range(0, len(predictForest)):
						if predictForest[k] > 0.5:
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
				# print subForest.predict_proba(instance)
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
precisionP = correctP/float((correctP+falseP))
recallR = correctP/float((correctP+falseN))
fMeasure = (2*precisionP*recallR)/float((recallR+precisionP))
wAccuracy = (correctP+correctN)/float((correctP+falseN+falseP+correctN))
print "Precision: ", precisionP
print "Recall: ", recallR
print "F-measure: ", fMeasure
print "Accuracy: ", wAccuracy

rocScore = roc_auc_score(y, X)
print "Area under ROC: ", rocScore
print "total: ", numTotal
f = open('runOutput.txt', 'w')
tempStr = "number correct: " + str(numberCorrect) + "\n"
f.write(tempStr)
tempStr = "correct positives: " + str(correctP) + "\n"
f.write(tempStr)
tempStr = "false negatives: " + str(falseN) + "\n"
f.write(tempStr)
tempStr = "false positives: " + str(falseP) + "\n"
f.write(tempStr)
tempStr = "correct negatives: " + str(correctN) + "\n"
f.write(tempStr)
tempStr = "Precision: " + str(precisionP) + "\n"
f.write(tempStr)
tempStr = "Recall: " + str(recallR) + "\n"
f.write(tempStr)
tempStr = "F-measure: " + str(fMeasure) + "\n"
f.write(tempStr)
tempStr = "Accuracy: " + str(wAccuracy) + "\n"
f.write(tempStr)
tempStr = "Area under ROC: " + str(rocScore) + "\n"
f.write(tempStr)
tempStr = "total: " + str(numTotal) + "\n"
f.write(tempStr)
f.close()
f = open('gold.csv','w')
for i in range(0, len(y)):
	stringVar = str(y[i]) + "\n"
	f.write(stringVar)
f.close()

f2 = open('pred.csv', 'w')
for i in range(0, len(X)):
	stringVar = str(xAccuracy[i]) + "," + str(X[i]) + "\n"
	f2.write(stringVar)
f2.close()
