import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#source: http://sebastianraschka.com/Articles/2014_pca_step_by_step.html

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

mean_list = []
for i in range(0, 784):
	mean_list.append(np.mean(X[i, :]))

mean_vector = np.array(mean_list)
print('Mean Vector:\n', mean_vector)
cov_mat = np.cov(X)
print('Covariance Matrix:\n', cov_mat)
plt.title("Mean Image")
print np.shape(mean_vector)
plt.imshow(np.reshape(mean_vector,(28,28)))
plt.savefig("mean.png")

eig_val_cov, eig_vec_cov = np.linalg.eig(cov_mat)
# for i in range(len(eig_val_cov)):
# 	eigvec_cov = eig_vec_cov[:,i].reshape(1,6000).T


# 	print('Eigenvector {}: \n{}'.format(i+1, eig_vec_cov))
# 	print('Eigenvalue {} from covariance matrix: {}'.format(i+1, eig_val_cov[i]))
# 	print(40 * '-')

# Make a list of (eigenvalue, eigenvector) tuples
eig_pairs = [(np.abs(eig_val_cov[i]), eig_vec_cov[:,i]) for i in range(len(eig_val_cov))]

# Sort the (eigenvalue, eigenvector) tuples from high to low
eig_pairs.sort(key=lambda x: x[0], reverse=True)

matrix_w = np.hstack((eig_pairs[0][1].reshape(6000,1), eig_pairs[1][1].reshape(6000,1), eig_pairs[2][1].reshape(6000,1), eig_pairs[3][1].reshape(6000,1), eig_pairs[4][1].reshape(6000,1), eig_pairs[5][1].reshape(6000,1), eig_pairs[6][1].reshape(6000,1), eig_pairs[7][1].reshape(6000,1), eig_pairs[8][1].reshape(6000,1), eig_pairs[9][1].reshape(6000,1)))

print('Matrix W:\n', matrix_w)
print matrix_w
fileName = "eigen"
fileExtension = ".png"
for i in range(0, 10):
	fileTitle = "Eigen"
	fileTitle += str(i)
	plt.title(fileTitle)
	plt.imshow(np.reshape(matrix_w[i],(28,28)))
	fileSave = fileName
	fileSave += i
	fileSave += fileExtension
	plt.savefig(fileSave)

# print X
# plt.figure(1)
# plt.title("Test Picture")
# plt.imshow(np.reshape(X[0],(28,28)))
# plt.savefig("p2.png")