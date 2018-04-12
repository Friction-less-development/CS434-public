import numpy as np

np.set_printoptions(suppress=True)
y = np.array([170, 191, 189, 180.34, 171, 176.53, 187, 185.42, 190, 181, 180, 175, 188, 170, 185])
x = np.array([[1, 50, 166], [1, 57, 196], [1, 50, 191], [1, 53.34, 180.34], [1, 54, 174], [1, 55.88, 176.53], [1, 57, 177], [1, 55.88, 208.28], [1, 57, 199], [1, 54, 181], [1, 55, 178], [1, 53, 172], [1, 57, 185], [1, 49.5, 165], [1, 57, 188]])

xt = np.transpose(x)
xtx = np.dot(xt, x)
print xtx
XInv = np.linalg.inv(xtx)
XInvXT = np.dot(XInv, xt)
w = np.dot(XInvXT, y)
print w
sumSq = 0.0
for i in range(0, 3):
	x[:,i] *= w[i]

# calculate sum of squares (SSE)
for j in range(0, 15):
	sumSq += (y[j]-(x[j][1]+x[j][2]+x[j][0]))*(y[j]-(x[j][1]+x[j][2]+x[j][0]))
	print y[j]
	print x[j][1]
	print x[j][2]
	print x[j][0]
	print "---"

print x
print sumSq # sum of sqaures