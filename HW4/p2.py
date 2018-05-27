import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#sources: 	http://sebastianraschka.com/Articles/2014_pca_step_by_step.html
#			https://stackoverflow.com/questions/13224362/principal-component-analysis-pca-in-python

X = np.zeros(shape=(1, 784))

firstLine = True
lines = 1
with open('data-1.txt','r') as f:
    for line in f:
        lineWords = []
        averageValue = []
        for word in line.split(','):
           lineWords.append(float(word))
        if firstLine:
            X[0] = lineWords
            firstLine = False
        else:
            X = np.vstack((X, lineWords))
            if(lines%100 == 0):
            	print lines
            lines += 1
meanX = np.zeros(shape=(784, 1))
meanX = np.mean(X, axis=0)
X = X - meanX

R = np.cov(X, rowvar=False)
from scipy import linalg as LA

evals, evecs = LA.eig(R)
idx = np.argsort(evals)[::-1]
evecs = evecs[:,idx]

evals = evals[idx]
evecs = evecs[:, :10]
transformed = evecs.real


fileName = "eigen"
fileExtension = ".png"
for i in range(0, 10):
	fileTitle = "Eigen"
	fileTitle += str(i)
	plt.title(fileTitle)
	plt.imshow(np.reshape(transformed[:, i],(28,28)))
	fileSave = fileName
	fileSave += str(i)
	fileSave += fileExtension
	plt.savefig(fileSave)


transformedX = np.dot(evecs.T, X.T).T

print( transformedX.shape)

closestFittingTracker = np.zeros(shape=(10, 1))
closestFitting = np.zeros(shape=(10, 1))

for i in range(0, 6000):
	for j in range(0, 10):
		if transformedX[i, j] > closestFittingTracker[j, 0]:
			closestFittingTracker[j, 0] = transformedX[i, j]
			closestFitting[j,0] = i
print(closestFittingTracker)
print(closestFitting)

fileName = "eigenmatch"
fileExtension = ".png"
for i in range(0, 10):
	fileTitle = "Eigenmatch"
	fileTitle += str(i)
	plt.title(fileTitle)
	plt.imshow(np.reshape(X[int(closestFitting[i]), : ],(28,28)))
	fileSave = fileName
	fileSave += str(i)
	fileSave += fileExtension
	plt.savefig(fileSave)
