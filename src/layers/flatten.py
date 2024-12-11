import numpy as np
from src.layers.layer import Layer
import math


class Flatten(Layer):
    """
    A Flatten layer to reshape input from a multi-dimensional tensor 
    (e.g., output of a Conv2D layer) to a 2D array suitable for fully connected layers.
    """

    def __init__(self):
        """
        Initialize the Flatten layer.
        """
        super().__init__()
        self.units = None  # Total number of units after flattening
        self.input_size = None  # Original input size

    def build(self, input_size):
        """
        Build the Flatten layer by computing the total number of units 
        based on the input shape.

        Parameters
        ----------
        input_size : tuple
            The shape of the input tensor (channels, height, width).
        """
        self.units = math.prod(input_size)  # Calculate the flattened size
        self.input_size = input_size  # Store the input size for backward pass

    def forward(self, input: np.ndarray):
        """
        Flatten the input tensor into a 2D array.

        Parameters
        ----------
        input : np.ndarray
            Input tensor of shape (samples, channels, height, width).

        Returns
        -------
        output : np.ndarray
            Flattened array of shape (samples, channels * height * width).
        """
        # Reshape input to (samples, units)
        return input.reshape(input.shape[0], self.units)

    def backward(self, dout: np.ndarray, lr):
        """
        Reshape the gradient to the original input shape.

        Parameters
        ----------
        dout : np.ndarray
            Gradient of loss with respect to the output of the Flatten layer, 
            of shape (samples, channels * height * width).

        Returns
        -------
        dinput : np.ndarray
            Reshaped gradient of shape (samples, channels, height, width).
        """
        # Reshape dout back to original input shape
        return dout.reshape(dout.shape[0], *self.input_size)
