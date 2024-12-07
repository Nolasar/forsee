import numpy as np
from src.layers.layer import Layer
import math

class Flatten(Layer):
    def __init__(self):
        self.units = None

    def build(self, input_size):
        self.units = math.prod(input_size)
        
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
        samples, channels, h, w = input.shape
        output = input.reshape(samples, channels * h * w)
        return output

    def backward(self):
        return super().backward()