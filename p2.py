#Authors: Rex Henzie, Benjamin Richards, and Michael Giovannoni

#HOW TO RUN: python p2.py
#Folder must contain usps-4-9-train.csv and usps-4-9-test.csv data files

import numpy as np
from math import exp
import csv
from sklearn.linear_model import LogisticRegression

import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# read training data
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

# read test data
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
    w = np.zeros(256)
    for iteration in range (0, iterations):
        delVector = np.zeros(256)
        for i in range (0, len(xSet)):
            yhat = sigmoid(X[i], w)
            yhatMinusYi = np.subtract(yhat, ySet[i])
            delVector = np.add(delVector, np.multiply(yhatMinusYi, X[i]))
        w = np.subtract(w, np.multiply(learningRate, delVector))

    return w

# batch learning logistic regression with regularization
def linRegWithRegularization(learningRate, iterations, xSet, ySet):
    lam = .001
    w = np.zeros(256)
    for iteration in range (0, iterations):
        delVector = np.zeros(256)
        for i in range (0, len(xSet)):
            yhat = sigmoid(X[i], w)
            yhatMinusYi = np.subtract(yhat, ySet[i])
            delVector = np.add(delVector, np.multiply(yhatMinusYi, X[i]))

        regularize = np.linalg.norm(np.linalg.norm(w))
        regularize = np.multiply(float(.5), np.multiply(lam, regularize))

        w = np.add(np.subtract(w, np.multiply(learningRate, delVector)) , regularize)

    return w

# calculate predictions given a weight vector
def predict(xSet, weightVector):
    yPredictions = sigmoid(xSet, weightVector)
    for i in range (0, len(yPredictions)):
        if yPredictions[i] < .5:
            yPredictions[i] = 0
        else:
            yPredictions[i] = 1
    return yPredictions

# calculate accuracy
def accuracy(predictions, actual):
    nums = np.zeros(len(predictions))
    for i in range (len(predictions)):
        nums[i] = (float(np.abs(predictions[i]- actual[i])))
    #print(np.mean(nums, dtype=np.float32))
    return 100.0 - np.mean(nums) * 100.0

# # Test model by comparing with sklearn's logistic regression function
# model = LogisticRegression()
# model.fit(X,np.ravel(y))
# print("First 5 sklearn coefs:        ", model.coef_[0,0:5])
# learnedWeights = linReg(.001, 100, X, y)
# print("First 5 of our model's coefs: ", learnedWeights[0:5])

print("\n\n")

print("Logistic regression: \n")
learnedWeights = linReg(.001, 100, X, y)
predictions = predict(X, learnedWeights)
print("Training set accuracy: ", round(accuracy(predictions, y), 1))
predictions = predict(X_test, learnedWeights)
print("Test set accuracy: ", accuracy(predictions, y_test))


print("\n")

print("Logistic regression with regularization: \n")
learnedWeights = linRegWithRegularization(.001, 100, X, y)
predictions = predict(X, learnedWeights)
print("Training set accuracy: ", round(accuracy(predictions, y),1))
predictions = predict(X_test, learnedWeights)
print("Test set accuracy: ", accuracy(predictions, y_test))