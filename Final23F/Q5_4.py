import numpy as np
import matplotlib.pyplot as plt

from time import time
import datetime

m = 100
n = 3

ITER = 50000
lr = 0.0001
y_0 = np.random.choice([0, 1], size=(m,1) , p=[.5, .5])
y_1 = np.array([y*2-1 for y in y_0])

X = np.random.rand(m,n)

x0 = np.ones((m,1))
X = np.hstack((x0,X))

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def L(w, X, y):
    I = np.ones((X.shape[0],1))
    Xw =np.dot(X,w)
    YtXw = np.dot(np.transpose(y),Xw)
    Delta = np.log(I+np.exp(Xw))
    return np.sum(-YtXw+np.dot(np.transpose(I),Delta),axis=0)
 

def dL(w, X, y):
    Xw=np.dot(X,w)
    distance = sigmoid(Xw)-y
    return np.dot(np.transpose(X),distance)

def GD(w, X, y, epoch, lr):
    l_list = []
    times = []
    time0 = time()
    
    for i in range(epoch):
        dw= dL(w,X, y)
        w -= lr * dw
        l =  L(w, X, y)
        times = np.append(times,time()-time0)
        l_list = np.append(l_list,l)
    return w, l_list, times


def H(X,w):
    I = np.ones((X.shape[0],1))
    Xw =np.dot(X,w)
    h = sigmoid(Xw)
    I_h = I- h
    Delta = h*I_h
    Delta = Delta.ravel()
    W = np.diag(Delta)
    WX = np.dot(W,X)
    H =  np.dot(np.transpose(X),WX)
    return H
    
def Newton(w,X,y,epoch, lr):
    l_list = []
    times = []
    time0 = time()

    for i in range(epoch):
        Hessian = H(X,w)
        Gradient = dL(w, X, y)
        w = w - lr*np.dot(np.linalg.pinv(Hessian),Gradient)
        l =  L(w, X, y)
        l_list = np.append(l_list,l)
        times = np.append(times,time()-time0)
    return w, l_list, times

w0 = np.zeros([n+1,1])
wnew = np.zeros([n+1,1])


L0_LIST = []
Lnew_LIST = []

w_new,Lnew_LIST, times_new = Newton(wnew,X,y_0,ITER,lr)
w0,L0_LIST,times = GD(w0,X,y_0,ITER,lr)
#plt.plot(times,L0_LIST)
plt.plot(times_new,Lnew_LIST, color = "g", label='Newton')
plt.plot(times,L0_LIST,color = "b",label='GD')

plt.legend()
plt.show()