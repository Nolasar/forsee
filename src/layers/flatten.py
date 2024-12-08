import numpy as np
from src.layers.layer import Layer
import math

class Flatten(Layer):
    def __init__(self):
        pass

    def build(self, input_size):
        self.units = math.prod(input_size)
        self.input_size = input_size

    def forward(self, input: np.ndarray):
        """
        Flatten an output of conv2d layer.

        Parameters
        ----------
        input : np.ndarray
            Conv2d layer output with shape (samples, channels, H, W)

        Returns
        -------
        output : np.ndarray
            2d numpy array of shape (samples, channels * H * W)
        """
        return input.reshape(input.shape[0], self.units)

    def backward(self, dout, lr):
        return dout.reshape(dout.shape[0], *self.input_size)