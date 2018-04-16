import numpy as np
from math import exp
import csv
from sklearn.linear_model import LogisticRegression

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# training data
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

# test data
X_test = np.zeros(shape=(800,256))
y_test = np.zeros(shape=(800,1))
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
        X_test[firstLine] = lineWords
        y_test[firstLine] = outcome
        firstLine += 1
X_test = np.divide(X_test, 255)


# sigmoid function
def sigmoid(x, w):
    yhat = 1 / (1 + np.exp(-(np.dot(x, w))))
    return yhat

# batch learning logistic regression
def linReg(learningRate, iterations, xSet, ySet):
    gradientMagnitudeVector = []

    w = np.zeros(256)
    for iteration in range (0, iterations):
        delVector = np.zeros(256)
        for i in range (0, len(xSet)):
            yhat = sigmoid(X[i], w)
            yhatMinusYi = np.subtract(yhat, ySet[i])
            delVector = np.add(delVector, np.multiply(yhatMinusYi, X[i]))
        gradientMagnitudeVector.append(np.linalg.norm(delVector))
        w = np.subtract(w, np.multiply(learningRate, delVector))

    #print(gradientMagnitudeVector)

    return w, gradientMagnitudeVector

# batch learning logistic regression with regularization
def linRegWithRegularization(learningRate, iterations, xSet, ySet):
    lam = .01
    gradientMagnitudeVector = []

    w = np.zeros(256)
    for iteration in range (0, iterations):
        delVector = np.zeros(256)
        for i in range (0, len(xSet)):
            yhat = sigmoid(X[i], w)
            yhatMinusYi = np.subtract(yhat, ySet[i])
            delVector = np.add(delVector, np.multiply(yhatMinusYi, X[i]))
        gradientMagnitudeVector.append(np.linalg.norm(delVector))
        
        
        regularize = np.linalg.norm(np.power(w, 2))
        regularize = np.multiply(float(.5), np.multiply(lam, regularize))
        
        w = np.add(np.subtract(w, np.multiply(learningRate, delVector)) , regularize)

    #print(gradientMagnitudeVector)

    return w, gradientMagnitudeVector

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


learnedWeights, gradientMagnitudeData = linReg(.001, 100, X, y)
print("First 5 of our model's coefs: ", learnedWeights[0:5])

print("\n\n")

predictions = predict(X, learnedWeights)
print("Train accuracy: ", accuracy(predictions, y))

predictions = predict(X_test, learnedWeights)
print("Test accuracy: ", accuracy(predictions, y_test))

print("\n\n")

learnedWeights, gradientMagnitudeData = linRegWithRegularization(.001, 100, X, y)
predictions = predict(X, learnedWeights)
print("Train accuracy: ", accuracy(predictions, y))

predictions = predict(X_test, learnedWeights)
print("Test accuracy: ", accuracy(predictions, y_test))



# Experiment with different learning rates and note gradient convergence
learnedWeights, gradientMagnitudeDataPoint0001 = linReg(.0001, 100, X, y)
predictions = predict(X, learnedWeights)
print("Accuracy: ", accuracy(predictions, y))
learnedWeights, gradientMagnitudeDataPoint001 = linReg(.001, 100, X, y)
predictions = predict(X, learnedWeights)
print("Accuracy: ", accuracy(predictions, y))
learnedWeights, gradientMagnitudeDataPoint01 = linReg(.01, 100, X, y)
predictions = predict(X, learnedWeights)
print("Accuracy: ", accuracy(predictions, y))


plt.figure(1)
plt.subplot(311)
plt.ylabel('Gradient Magnitude')
plt.xlabel('Iterations')
xAxis = range(100)
plt.title('Gradient Magnitude VS Iterations: Learning Rate = 01')
plt.plot(xAxis, gradientMagnitudeDataPoint01, 'ro', label='Training Data')
plt.legend()
plt.subplot(312)
plt.ylabel('Gradient Magnitude')
plt.xlabel('Iterations')
plt.title('Gradient Magnitude VS Iterations: Learning Rate = 001')
plt.plot(xAxis, gradientMagnitudeDataPoint001, 'ro', label='Training Data')
plt.legend()
plt.subplot(313)
plt.ylabel('Gradient Magnitude')
plt.xlabel('Iterations')
plt.title('Gradient Magnitude VS Iterations: Learning Rate = 0001')
plt.tight_layout()
plt.plot(xAxis, gradientMagnitudeDataPoint0001, 'ro', label='Training Data')
plt.legend()
plt.savefig('p2part1.png')