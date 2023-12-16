import numpy as np
import matplotlib.pyplot as plt

m = 100
n = 3

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
    for i in range(epoch):   
        dw= dL(w,X, y)
        w -= lr * dw
        l =  L(w, X, y)
        l_list = np.append(l_list,l)
    return w, l_list

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

w0 = np.zeros([n+1,1])
w1 = np.zeros([n+1,1])

ITER = 50000
w0,L0_LIST = GD(w0,X,y_0,ITER,0.0001)
w1,L1_LIST= GD1(w1,X,y_1,ITER,0.0001)
iters = list(range(0,ITER))

print(w0)
print(w1)

plt.plot(iters,L0_LIST, 'or' ,label = "{0,1}")
plt.plot(iters,L1_LIST,'--'  ,label = "{-1,1}")
plt.legend()
plt.show()


