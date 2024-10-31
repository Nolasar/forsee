import numpy as np

class Accuracy:
    def __call__(self, y_pred, y_true):
        t_pn = np.sum(np.round(y_pred) == y_true)
        total = len(y_true)
        return t_pn / total