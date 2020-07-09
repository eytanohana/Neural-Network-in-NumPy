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
        
        Parameters
        ----------
        X: nd-array - The input to the network. (nsamples x nfeatures)
        
        Returns
        -------
        Yhat: nd-array - The output of the network. (nsamples x nclasses)
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
    
    def backward(self, Y):
        """
        Perform backpropagation through the network.
        
        This function expects to be called only after an 
        input has been forwarded.
        
        Parameters
        ----------
        Y: numpy nd-array - The true output labels for the current batch.
        
        Returns
        -------
        grads: A list of the gradient matrices. 
            1 for each layer: [dJ/dW1, dJ/dW2, ...].
        """
        gradients = []
        
        Yhat = self.grads.pop()
        # we reverse the gradients list that we 
        # built in the forward method because we 
        # need those elements in reverse order for backprop.
        self.grads.reverse()
        
        δ = Yhat - Y
        for X, Z, layer in zip(self.grads[::2], self.grads[1::2], self.layers[::-1]):
            W = layer.weights
            
            dW = X.T @ δ
            gradients.append(dW)
            
            δ = (δ @ W.T) * dReLU(Z)
            
        X = self.layers[0]
        dW = X.T @ δ
        gradients.append(dW)
        
        assert len(gradients) == len(self.layers), (len(gradients), len(self.layers))
        
        for dW, W in zip(gradients, self.layers[::-1]):
            assert dW.shape == W.weights.shape, (dW.shape, W.weights.shape)
        
        return gradients[::-1]
        
    
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

def dReLU(X):
    """ReLU'(X) = 1 if X > 0 else 0"""
    return (X > 0).astype(float)

def softmax(X):
    exp = np.exp(X - X.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)

def cross_entropy(Yhat, Y):
    ylogy = Y * np.log(Yhat)
    return -ylogy.sum()