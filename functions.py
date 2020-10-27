"""
Available functions:

- radial_basis_function_1d
- sigmoid_1d
- sigmoid

GRADIENTS:
- square_error_gradient
-sigmoid_gradient

LOSSES:
- square_loss
- cross_entropy_loss

KERNELS:
- linreg_kernel
- gaussian_kernel
- kernel_Sigma
"""

import numpy as np

def radial_basis_function_1d(x, c, h):
    return np.exp(-0.5*(x-c)**2/h**2)

def sigmoid_1d(x, v, b):
    return 1/(1+np.exp(-v*x-b))

def sigmoid(a):
    return 1/(1+np.exp(-a))

def square_error_gradient(Phi, y, w):
    return -2*Phi.T@(y-Phi@w)

def sigmoid_gradient(Phi, y, w):
    preds = sigmoid(Phi@w)
    return -np.sum(X.T@(y-preds), axis=1).reshape(-1,1)/y.shape[0]

def square_loss(Phi, y, w):
    return (y-Phi@w)**2

def cross_entropy_loss(Phi, y, w):
    y = y.flatten()
    preds = sigmoid(Phi@w).flatten()
    cost = -y*np.log(preds)-(1-y)*np.log(1-preds)
    return cost.sum()/y.shape[0]

def linreg_kernel(x1, x2, sigma_sq_w, sigma_sq_b):
    return sigma_sq_w*x1*x2 + sigma_sq_b

def gaussian_kernel(x1, x2, sigma_sq_f=1, l=1):
    return sigma_sq_f*np.exp(-1/2*(x1-x2)**2/l**2)

def kernel_Sigma(X1, X2, kernel, *args):
    Sigma = np.zeros((X1.size, X2.size))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            Sigma[i, j] = kernel(x1, x2, *args)
            
    return Sigma