import numpy as np
from scipy import stats

class BayesClassifier:
    """
    Bayesian classifier. Implements both Naive Bayes and non-naive Bayes.
    N - number of observations
    D - number of features
    K - number of classes
    
    Parameters:
    ===========
    X (array): (NxD) design matrix of data
    y (array): (Nx1) column vector of labels
    
    """
    
    def __init__(self, X, y):
        self.X, self.y = X, y

    def fit(self, naive=False):
        """
        Fits the classifier.
        
        Parameters:
        ===========
        naive (bool): if True, the covariance matrix will forced to be diagonal and thus the classifier will be Naive Bayes.
        
        """
        X, y = self.X, self.y

        classes = np.unique(y)
        self.distributions = []

        for i, c in enumerate(classes):
            mu = X[y==c].mean(axis=0)
            Sigma = np.cov(X[y==c].T)
            if naive:
                Sigma = np.diag(np.diag(Sigma))
            dist = stats.multivariate_normal(mu, Sigma)
            self.distributions.append(dist)

        self.pi = np.unique(y, return_counts=True)[1]/X.shape[0]
        
    def predict_scores(self, X):
        """
        Predicts the scores of each observation for each class.
        N - number of observations
        D - number of features
        K - number of classes
        
        Parameters:
        ===========
        X (array): (NxD) matrix of validation/testing features
        
        Returns:
        ========
        scores (array): (NxK) matrix of scores
        
        """
        scores = np.zeros((X.shape[0], len(self.distributions)))
        for i, dist in enumerate(self.distributions):
            scores[:, [i]] = (dist.pdf(X)*self.pi[i]).reshape(-1,1)
        scores = scores/scores.sum(axis=1).reshape(-1,1)
        return scores
    
    def predict_labels(self, X):
        """
        Predicts the labels according to the highest score of each observation.
        N - number of observations
        D - number of features
        K - number of classes
        
        Parameters:
        ===========
        X (array): (NxD) matrix of validation/testing features
        
        Returns:
        ========
        classes (array): (NxK) matrix of predicted labels for each observation
        
        """
        scores = self.predict_scores(X)
        return scores.argmax(axis=1)