"""
Available functions:

- make_Phi_polynomial_1d
- make_Phi_rbf_1d
- standardize
- train_val_test_split
- oneHotEncode
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

def train_val_test_split(X, y, shares=np.array([0.5, 0.25, 0.25]), random_state=None):
    assert shares.sum() == 1, "shares must sum to 1"
    
    if random_state:
        np.random.seed(random_state)
    
    idxs = np.arange(y.size)
    train_idx = idxs[:int(y.size*shares[0])]
    val_idx = idxs[int(y.size*shares[0]): int(y.size*shares[0])+int(y.size*shares[1])]
    test_idx = idxs[-int(y.size*shares[2]):]

    Xy = np.hstack((X,y))
    Xy = np.random.permutation(Xy)

    return Xy[train_idx, :-1], Xy[val_idx, :-1], Xy[test_idx, :-1], Xy[train_idx, [-1]], Xy[val_idx, [-1]], Xy[test_idx, [-1]]

def oneHotEncode(y):
    oh = np.zeros((y.shape[0], np.unique(y).size))
    for i, c in enumerate(np.sort(np.unique(y))):
        oh[np.where(y==c), i] += 1
    return oh