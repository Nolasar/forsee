import numpy as np
from src import initializers, activations
from src.layers.layer import Layer

class Dense(Layer):
    """
    Fully-connected layer implementation.

    Parameters
    ----------
    units : int
        Number of neurons in the layer.
    activation : str
        Activation function to use.
    weights_initializer : str, optional, default='glorot_uniform'
        Initializer for the weights matrix.
    bias_initializer : str, optional, default='glorot_uniform'
        Initializer for the bias vector.
    """
    def __init__(
        self,
        units: int,
        activation: str,
        weights_initializer: str = 'glorot_uniform',
        bias_initializer: str = 'glorot_uniform',
    ):
        """
        Initialize the Dense layer.
        """
        self.units = units
        self.activation = activations.get(activation)()
        self.weights_initializer = initializers.get(weights_initializer)()
        self.bias_initializer = initializers.get(bias_initializer)()

    def build(self, input_size):
        """
        Initialize weights and biases based on input size.

        Parameters
        ----------
        input_size : int or tuple
            Number of input features or neurons in the previous layer.
            If a tuple is provided, it supports flattened Conv2D output.
        """
        if isinstance(input_size, tuple):
            units_in, *_ = input_size[0]
        else:
            units_in = input_size

        # Initialize weights and bias
        self.weights = self.weights_initializer(shape=(units_in, self.units))
        self.bias = self.bias_initializer(shape=(1, self.units))

    def forward(self, input: np.ndarray):
        """
        Perform the forward pass of the Dense layer.

        Parameters
        ----------
        input : np.ndarray
            Input data of shape (samples, input_units).

        Returns
        -------
        out : np.ndarray
            Output of the layer after applying weights, bias, and activation.
            Shape: (samples, units).
        """
        self.input = input  # Save input for backward pass
        # Linear combination
        out = self.input @ self.weights + self.bias
        # Apply activation function if specified
        if self.activation is not None:
            out = self.activation(out)
        return out

    def backward(self, prev_grad: np.ndarray):
        """
        Perform the backward pass to compute gradients and propagate them.

        Parameters
        ----------
        prev_grad : np.ndarray
            Gradient of the loss with respect to the layer's output.

        Returns
        -------
        curr_grad : np.ndarray
            Gradient of the loss with respect to the layer's input.
        """
        # Compute gradient of activation
        out = self.forward(self.input)
        grad = self.activation.backward(out) * prev_grad

        # Compute gradients for weights and biases
        self.dweights = self.input.T @ grad
        self.dbias = np.ones((1, grad.shape[0])) @ grad

        # Compute gradient for the previous layer
        return grad @ self.weights.T

    def get_params(self):
        """
        Get the trainable parameters of the Dense layer.

        Returns
        -------
        dict
            Dictionary containing {'weights': weights, 'bias': bias}.
        """
        return {'weights': self.weights, 'bias': self.bias}

    def get_grads(self):
        """
        Get the gradients of the trainable parameters.

        Returns
        -------
        dict
            Dictionary containing {'weights': dweights, 'bias': dbias}.
        """
        return {'weights': self.dweights, 'bias': self.dbias}
