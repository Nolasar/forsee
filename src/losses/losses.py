import numpy as np

class BinaryCrossEntropy:
    def __call__(self, y_true, y_pred):
        return np.mean(
            -y_true * np.log(y_pred) - (1-y_true) * np.log(1-y_pred)
        )
    
    def backward(self, y_true, y_pred):
        return -y_true / y_pred + (1-y_true) / (1-y_pred)