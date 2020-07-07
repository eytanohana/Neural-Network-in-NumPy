import numpy as np

class Layer:
    """Represents a single layer in a fully-connected
    Feed-Forward neural network. 
    
    A single layer is represented by its weight matrix 
    which has the shape (ninputs, noutputs)"""
    
    def __init__(self, ninputs, noutputs):
        boundary = np.sqrt(6 / (ninputs + noutputs))
        self.weights = np.random.uniform(-boundary, boundary, (ninputs, noutputs))
        
    def forward(self, X):
        # Save X and Z for backpropagation.
        self.X = X
        self.Z = X @ self.weights
        return self.Z
    

def ReLU(X):
    return np.maximum(X, 0)

def softmax(X):
    exp = np.exp(X - X.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)