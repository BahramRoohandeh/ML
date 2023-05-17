# Full implementation of Linear Regression 
 
import numpy as np
import copy
 
# Data set
X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

b0 = 785.1811367994083
w0 = np.array([ 0.39133535, 18.75376745, 3.36032453, -26.42131618])

# step1 : computing model out put for every input

def model_output(x, w , b):
    m = x.shape[0]
    y = np.zeros(m)
    for i in range(m):
        f_x_i = np.dot(x[i],w) +b
        y[i] = f_x_i
    return y
model_output(X_train, w0 , b0)

# setp2 : computing cost of model according to w , b 
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_x_i = model_output(x, w, b)[i]
        err =  f_x_i - y[i]
        cost_i = err ** 2
        cost = (cost + cost_i)
    return cost / (2*m)
compute_cost(X_train,y_train, w0 , b0)

#step 3: computing gradient of model
def compute_gradient(x, y, w, b):
    m,n = x.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        f_x_i = model_output(x, w, b)[i]
        err =  f_x_i - y[i]
  
        for j in range(n):
            dj_dw[j] =dj_dw[j] +(err * x[i, j]) 
        dj_db =dj_db +  err
    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw , dj_db
compute_gradient(X_train, y_train, w0, b0)

#setp 4: computnig optimum w , b using gradient descent
def gradient_descent(x, y, w, b, alpha, iterations, compute_gradient):
    
    m = x.shape[0]
    w = copy.deepcopy(w)
    for i in range(iterations):
        dj_dw , dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw            
        b = b - alpha * dj_db

    return w, b
        
    
gradient_descent(X_train, y_train, np.zeros_like(w0), 0, 5.0e-7, 1000, compute_gradient)
