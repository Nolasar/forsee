import numpy as np
from src import initializers, activations
from src.layers.layer import Layer

class Dense(Layer):
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
        
    def build(self, input_size):
        '''
        Initialize weights and biases.

        Parameters
        ----------
        **input_size**: int or tuple
            Number of neurons in the previous layer. 
            Tuple recieved from conv2d layer, it's need for supporting flatten operation
        '''
        if isinstance(input_size, tuple):
            units_in, *_ = input_size[0]
        else:
            units_in = input_size

        self.weights = self.weights_initializer(shape=(units_in, self.units))
        self.bias = self.bias_initializer(shape=(1, self.units))

    def forward(self, input: np.ndarray):
        '''
        Compute linear combination `linear = input @ weights + broadcast(bias)`\n
        and apply activation function `out = activation(linear)` on it if specified
        
        Parameters
        ----------
        **input**: np.ndarray
            Output of the previous layer

        Returns
        -------
        **out**: np.ndarray
            Output of the current layer

        Notes
        -----
        **Shapes assumption**: 
            `input.shape = (m, n_prev)`
            `weights.shape = (n_prev, n_curr)`
            `bias.shape = (1, n_curr)`
            `out.shape = (m, n_curr)`
            where:
                **m**: samples
                **n_prev**: previous layer units
                **n_curr**: current layer units
        '''    
        self.input = input
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