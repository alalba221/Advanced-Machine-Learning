import shadow.utils
shadow.utils.set_seed(0, cudnn_deterministic=True)  # set seeds for reproducibility
#%matplotlib inline
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
import random
import math as m

n_samples = 1000  # number of samples to generate
noise = 0.05  # noise to add to sample locations
X, y = datasets.make_moons(n_samples=n_samples, noise=noise)


class my_kmeans:
    def __init__(self, clusers=2):
        self.k = clusers
        
    def cal_dis(self, data, centeroids):
        dis = []
        for i in range(len(data)):
            dis.append([])
            for j in range(self.k):
                dis[i].append(m.sqrt((data[i, 0] - centeroids[j, 0])**2 + (data[i, 1]-centeroids[j, 1])**2))
        return np.asarray(dis)    
    
    def divide(self, data, dis):
        clusterRes = [0] * len(data)
        for i in range(len(data)):
            seq = np.argsort(dis[i])
            clusterRes[i] = seq[0]

        return np.asarray(clusterRes)
    
    def centeroids(self, data, clusterRes):
        centeroids_new = []
        for i in range(self.k):
            idx = np.where(clusterRes == i)
            sum = data[idx].sum(axis=0)
            avg_sum = sum/len(data[idx])
            centeroids_new.append(avg_sum)
        centeroids_new = np.asarray(centeroids_new)
        return centeroids_new[:, 0: 2]
    
    def cluster(self, data, centeroids):
        clulist = self.cal_dis(data, centeroids)
        clusterRes = self.divide(data, clulist)
        centeroids_new = self.centeroids(data, clusterRes)
        err = centeroids_new - centeroids
        return err, centeroids_new, clusterRes
    
    def fit(self,data):
        clu = random.sample(data[:, 0:2].tolist(), 2)  
        clu = np.asarray(clu)
        err, clunew,  clusterRes = self.cluster(data, clu)
        while np.any(abs(err) > 0):
            #print(clunew)
            err, clunew,  clusterRes = self.cluster(data, clunew)

        clulist = self.cal_dis(data, clunew)
        clusterResult = self.divide(data, clulist)

        return clusterResult
    

def myKNN(S, k, sigma=2.0):
    N = len(S)
    A = np.zeros((N,N))

    for i in range(N):
        dist_with_index = zip(S[i], range(N))
        dist_with_index = sorted(dist_with_index, key=lambda x:x[0])
        neighbours_id = [dist_with_index[m][1] for m in range(k+1)] # xi's k nearest neighbours

        for j in neighbours_id: # xj is xi's neighbour
            A[i][j] = np.exp(-S[i][j]/2/sigma/sigma)
            A[j][i] = A[i][j] # mutually

    return A

def calLaplacianMatrix(adjacentMatrix):

    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(adjacentMatrix, axis=1)

    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix

    # normailze
    # D^(-1/2) L D^(-1/2)
    sqrtDegreeMatrix = np.diag(1.0 / (degreeMatrix ** (0.5)))
    return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)

def euclidDistance(x1, x2, sqrt_flag=False):
    res = np.sum((x1-x2)**2)
    if sqrt_flag:
        res = np.sqrt(res)
    return res

def Distance(X):
    X = np.array(X)
    S = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(i+1, len(X)):
            S[i][j] = 1.0 * euclidDistance(X[i], X[j])
            S[j][i] = S[i][j]
    return S

clusters = 2

Similarity = Distance(X)
Adjacent = myKNN(Similarity, k=5)
Laplacian = calLaplacianMatrix(Adjacent)
x, V = np.linalg.eig(Laplacian)
x = zip(x, range(len(x)))
x = sorted(x, key=lambda x:x[0])
H = np.vstack([V[:,i] for (v, i) in x[:clusters]]).T
result = my_kmeans(2).fit(H)
plt.title('spectral cluster result')
plt.scatter(X[:,0], X[:,1],marker='o',c=result)
plt.show()