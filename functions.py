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
- binary_cross_entropy
- multiclass_cross_entropy

KERNELS:
- linreg_kernel
- gaussian_kernel
- kernel_Sigma
"""

import numpy as np

def radial_basis_function_1d(x, c, h):
    """
    One-dimensional radial basis function
    
    Parameters:
    ===========
    x (array): vector of data
    c (float): center of the radial basis function
    h (float): bandwidth
    
    """
    return np.exp(-0.5*(x-c)**2/h**2)

def sigmoid_1d(x, v, b):
    return 1/(1+np.exp(-v*x-b))

def sigmoid(a):
    """
    Simple sigmoid function
    
    """
    return 1/(1+np.exp(-a))

def square_error_gradient(Phi, y, w):
    return -2*Phi.T@(y-Phi@w)

def sigmoid_gradient(Phi, y, w):
    preds = sigmoid(Phi@w)
    return -np.sum(X.T@(y-preds), axis=1).reshape(-1,1)/y.shape[0]

def square_loss(Phi, y, w):
    return (y-Phi@w)**2

def softmax(X, W):
    """
    A softmax function used for multiclass classification.
    
    N - number of samples
    D - number of features
    K - number of classes
    
    Parameters:
    ===========
    X (array): (NxK) design matrix of features
    W (array): (KxD) matrix of weights for every class
    
    Returns:
    ========
    scores (array): (NxK) matrix of normalized scores
    
    """
    scores = np.exp(X@W.T)/np.exp(X@W.T).sum(axis=1).reshape(-1,1)
    return scores

def binary_cross_entropy(y_true, y_pred):
    """
    N - number of samples
    
    Parameters:
    ===========
    y_true (array): (Nx1) column vector of true labels
    y_pred (array): (Nx1) column vector of predicted labels
    
    Returns:
    ========
    cost (float)
    
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    cost = -y_true*np.log(y_pred)-(1-y_true)*np.log(1-y_pred)
    return cost.sum()/y_true.size

def multiclass_cross_entropy(Y_true, Y_pred):
    """
    N - number of samples
    K - number of classes
    
    Parameters:
    ===========
    Y_true (array): (NxK) oneHot-endcoded matrix of true labels
    Y_pred (array): (NxK) matrix of predicted class probabilities
    
    Returns:
    ========
    cost (float)
    
    """
    cost = 0
    K = Y_true.shape[1]
    for k in np.arange(K):
        cost += binary_cross_entropy(Y_true[:, [k]], Y_pred[:, [k]])
    return cost

#############
## KERNELS ##
#############

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