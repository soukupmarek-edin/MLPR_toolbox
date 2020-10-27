import numpy as np
from scipy import stats

class BayesianLinearRegression:
    """
    A Bayesian model for linear regression

    Parameters:
    ===========
    Phi (array): design matrix of features
    y (array): column vector of labels
    w0 (array): column vector of prior means for weights
    V0 (array): prior covariance matrix
    """
    
    def __init__(self, Phi, y, w0, V0):
        self.w0, self.V0, self.Phi, self.y = w0, V0, Phi, y
        self.sigma_sq_y = y.var()

    def gaussian_prior(self):
        w0, V0, Phi, y = self.w0, self.V0, self.Phi, self.y
        
        return stats.multivariate_normal(mean=w0.flatten(), cov=V0)

    def gaussian_posterior(self):
        w0, V0, Phi, y = self.w0, self.V0, self.Phi, self.y
        sigma_sq_y = self.sigma_sq_y
        
        Vn = sigma_sq_y*np.linalg.inv((sigma_sq_y*np.linalg.inv(V0) + Phi.T@Phi))
        wn = Vn@np.linalg.inv(V0)@w0 + 1/sigma_sq_y*Vn@Phi.T@y
        self.wn, self.Vn = wn, Vn
        self.posterior = stats.multivariate_normal(self.wn.flatten(), self.Vn)

        return self.posterior

    def prediction_density(self, Phi_pred):
        wn, Vn = self.wn, self.Vn
        sigma_sq_y = self.sigma_sq_y
        
        mean = Phi_pred.T@wn
        variance = Phi_pred.T@Vn@Phi_pred + sigma_sq_y
        pred_dens = stats.norm(mean, np.sqrt(variance))
        
        return mean, variance, pred_dens