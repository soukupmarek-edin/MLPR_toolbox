from MLPR_toolbox.functions import sigmoid, binary_cross_entropy
from scipy.optimize import minimize


class LogisticRegression:
    """
    Binary class logistic regression model.
    
    N - number of observations
    D - number of features
    
    Parameters:
    ===========
    X (array): (NxD) design matrix of features
    y (array): (Nx1) columns vector of binary labels
    """
    
    def __init__(self, X, y):
        self.X, self.y = X, y
    
    def fit(self, eps=0):
        """
        Fit logistic regression model.
        
        Parameters:
        ===========
        eps (int): robustness parameter. Default: eps=0
        
        Returns:
        ========
        w_opt (array): a column vector of fitted weights
        """
        def fun(w, eps):
            y_pred = (1-eps)*sigmoid(X_train@w) + eps/2
            cost = binary_cross_entropy(y_train, y_pred)
            return cost
        self.w_opt = minimize(lambda w: fun(w.reshape(-1,1), eps ), x0=np.zeros(2), method='BFGS')['x'].reshape(-1,1)
        return self.w_opt
    
    def predict(self, X):
        return sigmoid(X@self.w_opt)
    
    def fit_predict(self, X, eps):
        self.fit(eps)
        return self.predict(X)