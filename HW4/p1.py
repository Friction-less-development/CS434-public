

import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from random import randint
import math


# In[9]:


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
#             	print (lines
            lines += 1


# In[10]:


print(X.shape)


# In[47]:


def kmeans(data, k):
    # start with k random centers chosen from our data set
    rows, columns = data.shape
    centers = []
    SSElist = []
    for s in range (0, k):
        centers.append(data[randint(0,rows)])
    print("Starting center list length: ", len(centers))
#     print(centers)
    
    previousLabels = []
    currentLabels = []
    iters = 0
    notConverged = True
    
    while notConverged:
        previousLabels = currentLabels
        currentLabels = calcClusters(centers, data)
        
#         print("currentLabels: ", currentLabels)
#         print("\n")
#         print("previousLabels: ", previousLabels)
#         print("\n")
        
        # check to make sure the labels aren't the same (it's converged)
        if previousLabels == currentLabels:
            print("Converged after ", iters, " iterations")
            return
        
#         if iters%5 == 0:
#             print(iters, " iterations\n")
#             for s in range (0, k):
#                 print(np.linalg.norm(centers[s]), " ")
        
        
               
        
        # recompute centers to be the mean of each vector labeled as belonging to that cluster 
        for n in range(0, k):
            indicesBelongingToCluster = np.where(np.array(currentLabels) == n)
            clusterData = data[indicesBelongingToCluster]
#             print("\n", clusterData.shape,  "\n")
            centers[n] = clusterData.mean(axis = 0)
        
        #compute SSE
        SSElist.append(computeSSE(centers, data, currentLabels))
        print( "Iteration: ", iters, " ", SSElist[iters], "\n")
        
        iters += 1
        
    return

# compute SSE
def computeSSE(centers, data, labels):
    indvClusterSSE = []
     
    for n in range(0, len(centers)):
        indicesBelongingToCluster = np.where(np.array(labels) == n)
        clusterData = data[indicesBelongingToCluster]
        
        for s in clusterData:
            indvClusterSSE.append(np.linalg.norm(centers[n]-s))
    
    return sum(indvClusterSSE)
    
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
        
    
    

    
kmeans(X, 40)
    

