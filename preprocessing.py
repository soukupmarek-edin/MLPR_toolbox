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
    """
    Creates polynomial design matrix, where
    N - number of observations
    D - number of polynomial orders
    
    Parameters:
    ===========
    X (array): (Nx1) column vector of data
    orders (array): (D,) vector of polynomial orders
    
    Returns:
    ========
    Phi (array): (NxD) matrix of created features
    
    """
    Phi = np.zeros((X.shape[0], orders.size))
    for i, o in enumerate(orders):
        Phi[:, [i]] = X**o
    return Phi

def make_Phi_rbf_1d(X, centers, h):
    """
    RBFs of one array of data.
    
    N - number of observations
    D - number of centers
    
    Parameters:
    ============
    X (array): (Nx1) column vector of data
    centers (array): (D,) vector of centers of the required RBFs
    h (float): bandwidth of the RBFs
    
    Returns:
    ========
    Phi (array): (NxD) matrix of the transformed data
    
    """
    Phi = np.zeros((X.shape[0], centers.size))
    for i, c in enumerate(centers):
        Phi[:, [i]] = radial_basis_function_1d(X, c, h)
    return Phi

def standardize(X):
    """
    Deducts mean from an array and divides it by standard deviation.
    N - number of observations
    D - number of features
    
    Parameters:
    ===========
    X (array): (NxD) matrix of features
    
    Returns:
    ========
    X_st (array): (NxD) matrix of standardized features
    
    """
    X_st = np.zeros(X.shape)
    for i in range(X.shape[1]):
        x = X[:, i]
        X_st[:, [i]] = ((x-x.mean())/x.std()).reshape(-1,1)
    return X_st

def train_val_test_split(X, y, shares=np.array([0.5, 0.25, 0.25]), random_state=None):
    """
    Splits data to training, validation and test set.
    N - number of observations
    D - number of features
    
    Parameters:
    ===========
    X (array): (NxD) matrix of data
    y (array): (Nx1) vector of labels 
    shares (array): (3,) array of the share of each dataset
    random_state (int): random state of the split, default = None
    
    Returns:
    ========
    X_train, X_val, X_test: matrices of features
    y_train, y_val, y_test: vectors of labels
    
    """
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
    """
    One-hot encoding of multiclass vector, where
    N - number of observations
    K - number of classes
    
    Parameters:
    ===========
    y (array): (Nx1) vector of labels
    
    Returns:
    ========
    oh (array): (NxK) one-hot encoded matrix of labels
    
    """
    oh = np.zeros((y.shape[0], np.unique(y).size))
    for i, c in enumerate(np.sort(np.unique(y))):
        oh[np.where(y==c), i] += 1
    return oh