import numpy as np
from src import initializers, activations
            
class Dense:
    '''
    Fully-Connected layer implementation.
    
    Parameters
    ----------
    **units**: int 
        Number of neurons in layer

    **activation**: str
        Activation function to use

    **weights_initializer**: str, optional, default = 'glorot_uniform'
        Initializer for weights matrix

    **bias_initializer**: str, optional, default = 'glorot_uniform'
        Initializer for bias vector   
    '''
    def __init__(
        self,
        units: int,
        activation: str,
        weights_initializer: str = 'glorot_uniform',
        bias_initializer: str = 'glorot_uniform',
        ):
        self.units = units
        self.activation = activations.get(activation)()
        self.weights_initializer = initializers.get(weights_initializer)()
        self.bias_initializer = initializers.get(bias_initializer)()
        
    def build(self, prev_units: int):
        '''
        Initialize weights and biases.

        Parameters
        ----------
        **prev_units**: int
            Number of neurons in the previous layer
        '''
        self.weights = self.weights_initializer(shape=(prev_units, self.units))
        self.bias = self.bias_initializer(shape=(1, self.units))

    def forward(self, prev_out: np.ndarray):
        '''
        Compute linear combination `linear = prev_out @ weights + broadcast(bias)`\n
        and apply activation function `out = activation(linear)` on it if specified
        
        Parameters
        ----------
        **prev_out**: np.ndarray
            Output of the previous layer

        Returns
        -------
        **out**: np.ndarray
            Output of the current layer

        Notes
        -----
        **Shapes assumption**: 
            `prev_out.shape = (m, n_prev)`
            `weights.shape = (n_prev, n_curr)`
            `bias.shape = (1, n_curr)`
            `out.shape = (m, n_prev)`
            where:
                **m**: samples
                **n_prev**: previous layer units
                **n_curr**: current layer units
        '''    
        self.input = prev_out
        out = self.input @ self.weights + self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out
    
    def backward(self, prev_grad: np.ndarray, lr: float):
        """
        Compute gradient of current layer `grad = activation_dfn(output) * prev_grad`\n
        Update weights `weights -= lr * (input.T @ grad)`\n
        Update bias `bias -= lr * (I @ grad)`

        Parameters
        ----------
        **prev_grad**: np.ndarray
            Output of the previous layer
        **lr**: float
            Learning rate

        Returns
        -------
        **curr_grad**: np.ndarray
            Return `grad * weights.T`
        """
        out = self.forward(self.input)
        grad = self.activation.backward(out) * prev_grad

        self.weights -= lr * (self.input.T @ grad)
        self.bias -= lr * (np.ones((1, grad.shape[0])) @ grad)
        
        return grad @ self.weights.T