"""
Available functions:

- radial_basis_function_1d
- sigmoid_1d
- square_error_gradient
- square_loss
"""

import numpy as np

def radial_basis_function_1d(x, c, h):
    return np.exp(-0.5*(x-c)**2/h**2)

def sigmoid_1d(x, v, b):
    return 1/(1+np.exp(-v*x-b))

def square_error_gradient(Phi, y, w):
    return -2*Phi.T@(y-Phi@w)

def square_loss(Phi, y, w):
    return (y-Phi@w)**2