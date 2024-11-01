import numpy as np

class Layer:
    def __init__(self):
        pass

    def build(self):
        raise NotImplemented
    
    def forward(self):
        raise NotImplemented
    
    def backward(self):
        raise NotImplemented