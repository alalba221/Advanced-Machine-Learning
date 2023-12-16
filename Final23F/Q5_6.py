import numpy as np
import matplotlib.pyplot as plt

m = 10000
n = 3
ITER = 50
lr = 0.01

y_0 = np.random.choice([0, 1], size=(m,1) , p=[.5, .5])
y_1 = np.array([y*2-1 for y in y_0])
I = np.ones((m,1))
X = np.random.rand(m,n)
x0 = np.ones((m,1))
X = np.hstack((x0,X))

batchsize = 4000
T = 100000
t_list = range(T)

from time import time
import datetime
def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def L(w, X, y):
    Xw =np.dot(X,w)   
    YtXw = np.dot(np.transpose(y),Xw)
    Delta = np.log(I+np.exp(Xw))
    return np.sum(-YtXw+np.dot(np.transpose(I),Delta),axis=0)
 

def dL(w, X, y):
    Xw=np.dot(X,w)
    distance = sigmoid(Xw)-y    
    return np.dot(np.transpose(X),distance)


def dL_SGD(w, X, y,index_list):
    X_sub = X[index_list,:]
    y_sub = y[index_list,:]
    return dL(w, X_sub, y_sub)


def SGD(w, X, y, epoch, lr, batchsize):
    l_list = []
    idx_list = np.random.choice(range(X.shape[0]), batchsize,replace = False)
    for i in range(epoch):   
        dw= dL_SGD(w,X, y,idx_list)
        w -= lr * dw
        l =  L(w, X, y)
        l_list = np.append(l_list,l)
    return w, l_list

w_hat = np.zeros([n+1,1])

for t in t_list:
    time0 = time()
    w0 = np.zeros([n+1,1])
    weight = (1/T)*(t+1)/(T+1)
    w0,L0_LIST = SGD(w0,X,y_0,ITER,lr,batchsize)
    w_hat +=  w0 *weight
    print(t ,time()-time0)
print(w_hat)


