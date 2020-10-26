import numpy as np
from scipy import stats

class BayesClassifier:
    
    def __init__(self, X, y):
        self.X, self.y = X, y

    def fit(self, naive=False):
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
        scores = np.zeros((X.shape[0], len(self.distributions)))
        for i, dist in enumerate(self.distributions):
            scores[:, [i]] = (dist.pdf(X)*self.pi[i]).reshape(-1,1)
        scores = scores/scores.sum(axis=1).reshape(-1,1)
        return scores
    
    def predict_labels(self, X):
        scores = self.predict_scores(X)
        return scores.argmax(axis=1)