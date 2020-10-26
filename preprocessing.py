"""
Available functions:

- make_Phi_polynomial_1d
- make_Phi_rbf_1d
- standardize
"""

import numpy as np
from MLPR_toolbox.functions import radial_basis_function_1d

def make_Phi_polynomial_1d(X, orders):
    Phi = np.zeros((X.shape[0], orders.size))
    for i, o in enumerate(orders):
        Phi[:, [i]] = X**o
    return Phi

def make_Phi_rbf_1d(X, centers, h):
    """
    RBFs from one-feature
    
    Parameters:
    ============
    X (array): data, the shape is (N, 1)
    """
    Phi = np.zeros((X.shape[0], centers.size))
    for i, c in enumerate(centers):
        Phi[:, [i]] = radial_basis_function_1d(X, c, h)
    return Phi

def standardize(X):
    X_st = np.zeros(X.shape)
    for i in range(X.shape[1]):
        x = X[:, i]
        X_st[:, [i]] = ((x-x.mean())/x.std()).reshape(-1,1)
    return X_st