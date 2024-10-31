import numpy as np

class Sigmoid:
    def __call__(self, x):
        '''
        Forward pass: Compute the sigmoid activation.
        
        Parameters
        ----------
        - x: argument of sigmoid ( 1 / (1 + e^(-x)))
        '''
        return  1 / (1 + np.exp(-x))

    def backward(self, sigm):
        '''
        Backward pass: Compute the derivative of the sigmoid function.
        The derivative is sigmoid(x) * (1 - sigmoid(x)).

        Parameters
        ----------
        - sigm: output of sigmoid (sigm = sigmoid(x))
        '''
        return sigm * (1 - sigm)