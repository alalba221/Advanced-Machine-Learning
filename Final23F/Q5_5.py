import numpy as np
import matplotlib.pyplot as plt

m = 10000
n = 3
ITER = 5000

y_0 = np.random.choice([0, 1], size=(m,1) , p=[.5, .5])
y_1 = np.array([y*2-1 for y in y_0])

X = np.random.rand(m,n)
x0 = np.ones((m,1))
X = np.hstack((x0,X))

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def L1(w, X, y):
    I = np.ones((X.shape[0],1))
    n = X.shape[1]
    Y = np.tile(y,(1,n))
    YXW = np.dot(Y*X,w)
    Delta = np.log(I + np.exp(-YXW))
    return np.sum(np.dot(np.transpose(I),Delta),axis=0)

def dL1(w, X, y): 
    I = np.ones((X.shape[0],1)) 
    n = X.shape[1]
    Y = np.tile(y,(1,n))
    YXW = np.dot(Y*X,w)
    Delta = I - sigmoid(YXW)
    return -np.dot(np.transpose(Y*X), Delta)
    

def GD1(w, X, y, epoch, lr):
    l_list = []
    for i in range(epoch):
        dw= dL1(w,X, y)
        w -= lr * dw
        l =  L1(w, X, y)
        l_list = np.append(l_list,l)
    return w, l_list

w1 = np.zeros([n+1,1])



lr =1/ np.linalg.norm(X,"fro")
lr = lr*lr

w1,L1_LIST= GD1(w1,X,y_1,ITER,lr)
iters = list(range(0,ITER))

print(w1)
plt.plot(iters,L1_LIST,label = "{-1,1}")
plt.legend()
plt.show()