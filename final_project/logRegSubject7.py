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
# indice file. Should be list7_part1 for subject 7
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

subject7X = []
subject7y = []
instanceCounter = 0

counter = 0
for i in range(0, len(chunksList)): # len(chunksList)
    temp2 = np.zeros(shape=(7,1))
    sub1HypoChunk = temp2
    temp = np.zeros(shape=(7, numColumns))
    instance = temp
    for j in range(0, len(chunksList[i])):
        # if j%10 == 0:
        #     print i
        #     print j
        #     print len(chunksList[i])
        #     print len(chunksList)
        #     print "\n"
        #     print instance
        #     print sub1HypoChunk
        #     print 5*"-"
        if j<7:
            instance[j][:] = chunksList[i][j][:]
            sub1HypoChunk[j][0] = int(sub1HypoChunkList[i][j][0])
        else:
            # for k in range(0, np.size(instance, 0)):
            #     for l in range(0, 8):
            # subForest.fit(instance, np.ravel(sub1HypoChunk)) #instance is 7 x 9 , sub1HypoChunk is 1 x 7, 10th column where the answers are
            # subForest.n_estimators += 1

            subject7X.append(np.ravel(instance))
            subject7y.append(sub1HypoChunk[6])


            instanceCounter += 1

            tempNum = j%7-1
            instance = np.delete(instance, tempNum, 0)
            sub1HypoChunk = np.delete(sub1HypoChunk, tempNum, 0)
            instance = np.vstack((instance, chunksList[i][j][:]))
            sub1HypoChunk = np.vstack((sub1HypoChunk, int(sub1HypoChunkList[i][j][0])))

# print(instance.shape)

X = np.array(subject7X)
y = np.ravel(np.array(subject7y))
# x, y, z = subject7X.shape
# subject7X = subject7X.reshape(x, y*z)

print "\n"
print "Subject_7"



print "fitting model"
logreg = LogisticRegression()
X = StandardScaler().fit_transform(X)
logreg.fit(X, y)

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

print np.shape(FTESTLIST)

print "Starting Subject 7 Test"
print 32*"-"
predictX = []
xAccuracy = []

X_test = []

for i in range(0, np.size(FTESTLIST, 0)): # np.size(FTESTLIST, 0)
    X_test.append(np.ravel(FTESTLIST[i]))

y_pred = logreg.predict(X_test)
y_score = logreg.decision_function(X_test)

f3 = open('individual2_pred3.csv', 'w')
for i in range(0, len(y_pred)):
	stringVar = str(y_score[i]) + "," + str(int(y_pred[i])) + "\n"
	f3.write(stringVar)
f3.close()

# X_train, X_test, y_train, y_test = train_test_split(subject7X, subject7y, test_size=0.3, random_state=0)
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
# print "\n\n Model coefs: "
# print logreg.coef_
# print "\n\n"
