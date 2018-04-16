import numpy as np
from math import exp
import csv
from sklearn.linear_model import LogisticRegression


np.set_printoptions(suppress=True)
X = np.zeros(shape=(1400,256))
y = np.zeros(shape=(1400,1))
firstLine = 0
with open('usps-4-9-train.csv','r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        featureNum = 0
        lineWords = []
        outcome = []
        for word in row:
            if featureNum < 256:
                lineWords.append(float(word))
                featureNum += 1
            else:
                outcome.append(float(word))
        X[firstLine] = lineWords
        y[firstLine] = outcome
        firstLine += 1


X = np.divide(X, 255)


# sigmoid function
def sigmoid(x, w):
    yhat = 1 / (1 + np.exp(-(np.dot(x, w))))
    return yhat

def linReg(learningRate, iterations, xSet, ySet):
<<<<<<< HEAD
    w = np.zeros(256)
=======
    gradientMagnitudeVector = []
    
    w = np.zeros(256)    
>>>>>>> origin/master
    for iteration in range (0, iterations):
        delVector = np.zeros(256)
        for i in range (0, len(xSet)):
            yhat = sigmoid(X[i], w)
            yhatMinusYi = np.subtract(yhat, ySet[i])
<<<<<<< HEAD
            delVector = np.add(delVector, np.multiply(yhatMinusYi, X[i]))
        w = np.subtract(w, np.multiply(learningRate, delVector))
#         if(iteration%20 == 0):
#             print("weights v2: ", w[0,0:5])

    return w
=======
            delVector = np.add(delVector, np.multiply(yhatMinusYi, X[i])) 
        gradientMagnitudeVector.append(np.linalg.norm(delVector))
        w = np.subtract(w, np.multiply(learningRate, delVector))

    #print(gradientMagnitudeVector)
    
    return w, gradientMagnitudeVector
>>>>>>> origin/master

def predict(xSet, weightVector):
    yPredictions = sigmoid(xSet, weightVector)
    for i in range (0, len(yPredictions)):
        if yPredictions[i] < .5:
            yPredictions[i] = 0
        else:
            yPredictions[i] = 1
    return yPredictions

def accuracy(predictions, actual):
    nums = []
    for i in range (len(predictions)):
        nums.append(np.abs(predictions[i]- actual[i]))
    return 100 - np.mean(nums) * 100


# Calculate coefficients
dataset = X
l_rate = 0.3
n_epoch = 1000


# test with sklearn
model = LogisticRegression()
model.fit(X,np.ravel(y))
print("First 5 sklearn coefs:        ", model.coef_[0,0:5])


learnedWeights, gradientMagnitudeData = linReg(.001, 1000, X, y)
print("First 5 of our model's coefs: ", learnedWeights[0:5])
predictions = predict(X, learnedWeights)
print("Accuracy: ", accuracy(predictions, y))


# # Experiment with different learning rates and note gradient convergence
# learnedWeights, gradientMagnitudeDataPoint0001 = linReg(.0001, 100, X, y)
# predictions = predict(X, learnedWeights)
# print("Accuracy: ", accuracy(predictions, y))
# learnedWeights, gradientMagnitudeDataPoint001 = linReg(.001, 100, X, y)
# predictions = predict(X, learnedWeights)
# print("Accuracy: ", accuracy(predictions, y))
# learnedWeights, gradientMagnitudeDataPoint01 = linReg(.01, 100, X, y)
# predictions = predict(X, learnedWeights)
# print("Accuracy: ", accuracy(predictions, y))


<<<<<<< HEAD
print("Accuracy: ", accuracy(predictions, y))

X = np.zeros(shape=(800,256))
y = np.zeros(shape=(800,1))
firstLine = 0
with open('usps-4-9-test.csv','r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        featureNum = 0
        lineWords = []
        outcome = []
        for word in row:
            if featureNum < 256:
                lineWords.append(float(word))
                featureNum += 1
            else:
                outcome.append(float(word))
        X[firstLine] = lineWords
        y[firstLine] = outcome
        firstLine += 1


X = np.divide(X, 255)

predictions = predict(X, learnedWeights)

print("Accuracy: ", accuracy(predictions, y))
=======
print("\n\ndone")
>>>>>>> origin/master
