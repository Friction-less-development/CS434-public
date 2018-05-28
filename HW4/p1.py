#Authors: Rex Henzie, Benjamin Richards, and Michael Giovannoni

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from random import randint
import math


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
#             if(lines%100 == 0):
#               print (lines
            lines += 1



print(X.shape)

# k-means function
def kmeans(data, k, iterations):
    print("Starting K-means.\n K = ", k, "\n")
    
    # start with k random centers chosen from our data set
    rows, columns = data.shape
    centers = []
    SSElist = []
    for s in range (0, k):
        centers.append(data[randint(0,rows)])
    
    previousLabels = []
    currentLabels = []
    iters = 0
    notConverged = True
    
    while iters <= iterations - 1:
        previousLabels = currentLabels
        currentLabels = calcClusters(centers, data)

        # recompute centers to be the mean of each vector labeled as belonging to that cluster 
        for n in range(0, k):
            indicesBelongingToCluster = np.where(np.array(currentLabels) == n)
            clusterData = data[indicesBelongingToCluster]
#             print("\n", clusterData.shape,  "\n")
            centers[n] = clusterData.mean(axis = 0)
        
        #compute SSE
        SSElist.append(computeSSE(centers, data, currentLabels))
#         print( "Iteration: ", iters, " ", SSElist[iters], "\n")
        
        iters += 1
        
    print("\n Done")
    return SSElist

# compute SSE
def computeSSE(centers, data, labels):
    indvClusterSSE = []
     
    for n in range(0, len(centers)):
        indicesBelongingToCluster = np.where(np.array(labels) == n)
        clusterData = data[indicesBelongingToCluster]
        
        for s in clusterData:
            indvClusterSSE.append(np.sqrt((np.linalg.norm(centers[n]-s)**2)))
    
    return round(sum(indvClusterSSE))
    
# calculates the closest center for each sample, returns a list of sample labels
def calcClusters(centers, samples):
    
    clusterLabels = []
    centerList = centers
    
    for s in samples:
        
        clusterLabels.append(calcBestCluster(centerList, s))
        
    
    return clusterLabels

# calculates the closest center for a given vector (sample)
def calcBestCluster(centers, sample):
    
    bestDistance = np.linalg.norm(centers[0]-sample)
    bestCenterIdx = 0
    
    # check euclidean distance between sample and each cluster center
    for i in range (0, len(centers)):
        dist = np.linalg.norm(centers[i]-sample)
        if dist < bestDistance:
            bestDistance = dist
            bestCenterIdx = i

    return bestCenterIdx
        
    


multSSE = []
iterations = 10
maxK = 20
maxK += 1
for k in range(2, maxK):
#     print("K = ", k, "\n")
    SSElist = kmeans(X, k, iterations)
    multSSE.append(SSElist)
    


labels = []
for k in range(2, maxK):
    labels.append("K = " + str(k))

myX = range(1, iterations+1)

plt.figure(figsize=(10,8))
plt.ylabel('SSE')
plt.xlabel('iterations')
plt.title('K-Means Error')

for k in range(len(multSSE)):
    plt.plot(myX, multSSE[k])
    

plt.legend(labels)
plt.show()
# plt.savefig('p1k2i20.png')


myX = range(2, maxK)
plt.figure(figsize=(10,8))
plt.ylabel('SSE')
plt.xlabel('# of clusters')
plt.title('SSE compared to K')
kYvector = []
print("!!!!!!", iterations, " ", maxK)
for t in range(len(multSSE)):
#     print("t = ", t, "iterations = ", iterations, "\n")
    kYvector.append(multSSE[t][iterations-1])
    
    
plt.plot(myX, kYvector)
plt.show()



iterations = 10
k = 2
SSElist = kmeans(X, k, iterations)

label = "K = " + str(k)
plt.figure(figsize=(10,8))
plt.ylabel('Sum of Errors')
plt.xlabel('iterations')
plt.plot(myX, SSElist, color="blue", label=label)
plt.title('K-Means Error')
plt.legend()
plt.show()
# plt.savefig('p1k2i20.png')

