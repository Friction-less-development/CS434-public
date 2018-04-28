import numpy as np
import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt

##Following reads in train.
X = np.zeros(shape=(284,30))    ##30 features
Y = np.zeros(shape=(284,1))     ##Whether it is Benign or Malignant. Corresponds with X
firstLine = 0
with open('knn_train.csv','r') as f:
 for line in f:
  firstVal = True
  features = []
  result = []
  for word in line.split(","):
   if firstVal == True:
    result.append(float(word))
    firstVal = False
   else:
    features.append(float(word))
  X[firstLine] = features
  Y[firstLine] = result
  firstLine = firstLine + 1
##Normalizing training data. This isn't relevant elsewhere.
maxvals = np.zeros(shape=(1,30))
minvals = np.zeros(shape=(1,30))
maxvals[0] = X[0]
minvals[0] = X[0]
for i in range(1,284):      #This loop finds the man/mix of each feature
 for j in range(0,30):
  if X[i][j] > maxvals[0][j]:
   maxvals[0][j] = X[i][j]
  if X[i][j] < minvals[0][j]:
   minvals[0][j] = X[i][j]
for i in range(0,284):      #This loop applies the normalization function to each value.
 for j in range(0,30):
  X[i][j] = (X[i][j] - minvals[0][j])/(maxvals[0][j] - minvals[0][j])
##Data should be normalized now.

##Reading in testing data
testX = np.zeros(shape=(284,30))   #30 features
testY = np.zeros(shape=(284,1))    #Whether it is benign or malignant. Corresponds with testX
firstLine = 0
with open('knn_test.csv','r') as f:
 for line in f:
  firstVal = True
  features = []
  result = []
  for word in line.split(","):
   if firstVal == True:
    result.append(float(word))
    firstVal = False
   else:
    features.append(float(word))
  testX[firstLine] = features
  testY[firstLine] = result
  firstLine = firstLine + 1

##normalizing the data. This isn't relevant elsewhere
maxtestvals = np.zeros(shape=(1,30))
mintestvals = np.zeros(shape=(1,30))
maxtestvals[0] = testX[0]
mintestvals[0] = testX[0]
for i in range(1,284):      #This loop finds the man/mix of each feature
 for j in range(0,30):
  if testX[i][j] > maxtestvals[0][j]:
   maxtestvals[0][j] = testX[i][j]
  if testX[i][j] < mintestvals[0][j]:
   mintestvals[0][j] = testX[i][j]
for i in range(0,284):      #This loop applies the normalization function
 for j in range(0,30):
  testX[i][j] = (testX[i][j] - mintestvals[0][j])/(maxtestvals[0][j] - mintestvals[0][j])
#data should be normalized now.

###END OF INITIALIZATION




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
  defaultValue = defaultValue + closestNeighbors[i][0]
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
 for i in range(0,284):
  if(guesses[i] != answers[i]):
   errors = errors + 1
 print errors
 accuracy = 1 - (float(errors)/284)
 print "Accuracy: "
 print accuracy
 print "%"
 return accuracy

#Predicts test set against training entries. Also against previously evaluated testing entries.
def predictTestingSet(k):
 resultY = np.zeros(shape=(284,1))
 for i in range(0,284):
  closestNeighbors = np.zeros(shape=(k,2))
  closestNeighbors.fill(-1)
  for j in range(0,284 + i):
   if(j < 284):
    norm = np.linalg.norm(np.absolute(np.subtract(testX[i], X[j])))
    closestNeighbors = addClosestNeighbor(closestNeighbors, norm, Y[j])
   else:        #This enables it to be evaluated against previously evaluated test data.
    norm = np.linalg.norm(np.absolute(np.subtract(testX[i], testX[j-284]))) #testX[j-284] should not catch up with testX[i]
    closestNeighbors = addClosestNeighbor(closestNeighbors, norm, resultY[j-284])
  resultY[i] = getMajority(closestNeighbors)

 print "Using K = "
 print k
 print "\n"
 return getAccuracy(resultY, testY)

#Predicts training set against the training set
def predictTrainingSet(k):
 resultY = np.zeros(shape=(284,1))
 for i in range(0,284):
  closestNeighbors = np.zeros(shape=(k,2))
  closestNeighbors.fill(-1)
  for j in range(0,284):
   norm = np.linalg.norm(np.absolute(np.subtract(X[i], X[j])))
   closestNeighbors = addClosestNeighbor(closestNeighbors, norm, Y[j])
  resultY[i] = getMajority(closestNeighbors)

 return getAccuracy(resultY, Y)

#Predicts the training set against the training set MINUS the same entry.
def leaveOneOutCrossValidation(k):
 resultY = np.zeros(shape=(284,1))
 for i in range(0,284):
  closestNeighbors = np.zeros(shape=(k,2))
  closestNeighbors.fill(-1)
  for j in range(0,284):
   if(i != j):      #Do not evaluate if we are subtracting a value from itself. (This is the leave-one-out part)
    norm = np.linalg.norm(np.absolute(np.subtract(X[i], X[j])))
    closestNeighbors = addClosestNeighbor(closestNeighbors, norm, Y[j])
  resultY[i] = getMajority(closestNeighbors)

 return getAccuracy(resultY, Y)

#This begins the user input/performance
while True:
 k = input("What is k (odd)? Or press -1 to graph errors between 1 and 53. ")
 if(k & 1):
  break;
if(k == -1):
 k = 1
 xaxis = []
 yaxis = []
 secondXaxis = []
 secondYaxis = []
 leaveOneOutCrossValidationErrorXaxis = []
 leaveOneOutCrossValidationErrorYaxis = []
 for i in range(0,25):
  plt.figure(1)
  plt.xlabel('k')
  plt.ylabel('Accuracy')
  plt.axis([0, 52, 0, 1])

  leaveOneOutCrossValidationErrorXaxis.append(k)
  leaveOneOutCrossValidationErrorYaxis.append(leaveOneOutCrossValidation(k))

  xaxis.append(k)
  yaxis.append(predictTestingSet(k))

  secondXaxis.append(k)
  secondYaxis.append(predictTrainingSet(k))

  k = k + 2

 plt.plot(xaxis, yaxis, 'ro', label='Test Data Error')
 plt.plot(secondXaxis, secondYaxis, 'bo', label='Training Error')
 plt.plot(leaveOneOutCrossValidationErrorXaxis, leaveOneOutCrossValidationErrorYaxis, 'go', label='Leave-one-out')
 plt.legend(loc='bottom right');
 plt.savefig('errors.png')
else:
 predictTestingSet(k)
