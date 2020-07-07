import numpy as np


class NN:
    """
    Represents a neural network consisting of
    multiple layers.
    """
    
    def __init__(self, layers):
        self.layers = list(layers)
        
    def forward(self, X):
        '''
        Forward an input X through the network
        by forwarding it through each layer.
        '''
        # we save the intermediate values we 
        # need for backpropagation later in
        # self.grads.
        self.grads = [X]
        
        for layer in self.layers[:-1]:
            Z = layer.forward(X)
            self.grads.append(Z)
            X = ReLU(Z)
            self.grads.append(X)
            
        Z = self.layers[-1].forward(X)
        Yhat = softmax(Z)
        self.grads.append(Yhat)
        return Yhat
    
    def predict(self, X):
        probs = self.forward(X)
        return probs.argmax(axis=1)
        
    def __str__(self):
        model = ''
        for i, layer in enumerate(self.layers, 1):
            model += str(i) + ') ' + str(layer) + '\n'
        return model
        

class Layer:
    """
    Represents a single layer in a fully-connected
    Feed-Forward neural network. 
    
    A single layer is represented by its weight matrix 
    which has the shape (ninputs, noutputs)
    """
    
    def __init__(self, ninputs, noutputs):
        # glorot uniform
        boundary = np.sqrt(6 / (ninputs + noutputs))
        self.weights = np.random.uniform(-boundary, boundary, (ninputs, noutputs))
        
    def forward(self, X):
        X = np.atleast_2d(X)
        return X @ self.weights
    
    def __str__(self):
        return f'Fully connected layer: ({self.weights.shape[0]}, {self.weights.shape[1]})'

    

def ReLU(X):
    '''ReLU(X) = X if X > 0 else 0'''
    return np.maximum(X, 0)

def softmax(X):
    exp = np.exp(X - X.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)

def cross_entropy(Yhat, Y):
    ylogy = Y * np.log(Yhat)
    return -ylogy.sum()