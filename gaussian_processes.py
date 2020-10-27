import numpy as np
from scipy import stats
from MLPR_toolbox.functions import kernel_Sigma

class GaussianProcess:
    
    def __init__(self, X_train):
        self.X_train = X_train
        self.N = X_train.shape[0]
        
    def prior(self, kernel, *kernel_params):
        Sigma = kernel_Sigma(self.X_train, self.X_train, kernel, *kernel_params)
        prior = stats.multivariate_normal(np.zeros(self.N), Sigma + 1e-7*np.eye(self.N))
        return prior
    
    def posterior(self, y_train, X_test, kernel, *kernel_params):
        X_train = self.X_train
        self.M = X_test.shape[0]
        N, M = self.N, self.M
        
        K_X_X = kernel_Sigma(X_train, X_train, kernel, *kernel_params)
        K_Xs_X = kernel_Sigma(X_test, X_train, kernel, *kernel_params)
        K_X_Xs = kernel_Sigma(X_train, X_test, kernel, *kernel_params)
        K_Xs_Xs = kernel_Sigma(X_test, X_test, kernel, *kernel_params)
        
        self.sigma_sq_y = np.var(y_train)*np.eye(y_train.size)
        self.mean = K_Xs_X@np.linalg.inv(K_X_X+self.sigma_sq_y)@y_train
        self.cov = K_Xs_Xs - K_Xs_X@np.linalg.inv(K_X_X+self.sigma_sq_y)@K_X_Xs + 1e-7*np.eye(M)
        posterior = stats.multivariate_normal(self.mean.flatten(), self.cov)
        
        return posterior