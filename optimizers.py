"""
Gradient descent
"""

import numpy as np

class GradientDescent:
    
    def __init__(self, Phi, y, gradient, loss):
        self.Phi, self.y = Phi, y
        self.gradient = gradient
        self.loss = loss

    def _update_w(self, w, eta):
        return w-eta*self.gradient(self.Phi, self.y, w)
    
    def find_weights(self, w0, eta=0.01, smpl=1, max_steps=1000, max_loss_diff=1e-5, verbose=True):

        self.losses = np.zeros(max_steps+1)
        self.losses[0] = 1e5
        size = int(self.Phi.shape[0]*smpl)
        idxs = np.arange(self.Phi.shape[0])
        w = np.copy(w0)

        for i in range(max_steps):
            w = self._update_w(w, eta)
            smpl_idx = np.random.choice(idxs, size=size, replace=False)
            Phi_ = self.Phi[smpl_idx, :]
            y_ = self.y[smpl_idx, :]
            self.losses[i+1] = np.sum(self.loss(Phi_, y_, w))

            if np.abs(self.losses[i]-self.losses[i-1]) < max_loss_diff:
                break
                
        if verbose:
            print(f'stopped after {i} steps')
            
        return w