import numpy as np
import matplotlib.pyplot as plt


m_list= [100,10000]
n=3

ITER = 50000
lr = 0.0001



from time import time
import datetime
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

def GD(w, X, y, thresdhold, lr):
    l_list = []
    times = []
    time0 = time()
    
    l = 0
    err = np.inf
    while err > thresdhold:
    #for i in range(epoch):
        dw= dL(w,X, y)
        w -= lr * dw
        l_new =  L(w, X, y)
        err = np.linalg.norm(l_new-l,1)/X.shape[0]
        times = np.append(times,time()-time0)
        l_list = np.append(l_list,l_new)
        l = l_new
    return w, l_list, times

def dL_SGD(w, X, y,index_list):
    X_sub = X[index_list,:]
    y_sub = y[index_list,:]

    return dL(w, X_sub, y_sub)


def SGD(w, X, y, thresdhold, lr, batchsize):
    w= w.copy()
    l_list = []
    idx_list = np.random.choice(range(X.shape[0]), batchsize,replace = False)
    times = []
    time0 = time()

    l = 0
    err = np.inf

    while err > thresdhold:
    #for i in range(epoch):   
        dw= dL_SGD(w,X, y,idx_list)
        w -= lr * dw
        l_new =  L(w, X, y)
        err = np.linalg.norm(l_new-l,1)/X.shape[0]
        l_list = np.append(l_list,l_new)
        l = l_new
        times = np.append(times,time()-time0)
    return w, l_list, times


for m in m_list:
    thresdhold = 0.0001/(m**2)
    y_0 = np.random.choice([0, 1], size=(m,1) , p=[.5, .5])
    
    I = np.ones((m,1))
    X = np.random.rand(m,n)
    x0 = np.ones((m,1))
    X = np.hstack((x0,X))

    w_gd = np.zeros([n+1,1])
    w_sgd = np.zeros([n+1,1])
    w_gd,L_GD,time_gd = GD(w_gd,X,y_0,thresdhold,lr)

    w_sgd,L_SGD,time_sgd= SGD(w_sgd,X,y_0,thresdhold,lr,m//2)
    
    iters = range(ITER)
        # plt.plot(iters,L_GD, 'or' ,label = "GD")
        # plt.plot(iters,L_SGD,'--'  ,label = "SGD")
    plt.plot(time_gd,L_GD ,label = "GD")
    plt.plot(time_sgd,L_SGD ,label = "SGD")
    plt.legend()
        
    plt.show()

print(L_GD[-1])