import numpy as np

class BinaryCrossEntropy:
    def __call__(self, y_pred, y_true):
        return np.mean(
            -y_true * np.log(y_pred) - (1-y_true) * np.log(1-y_pred)
        )
    
    def backward(self, y_true, y_pred):
        return -y_true / y_pred + (1-y_true) / (1-y_pred)
    

class CrossEntropy:
    def __call__(self, y_pred, y_true):
        """
        Forward pass for the cross-entropy loss.

        Args:
            y_pred (np.ndarray): Predicted probabilities of shape (N, C),
                                       where N is the batch size and C is the number of classes.
            y_true (np.ndarray): True labels in one-hot encoding of shape (N, C) or as class indices (N,).

        Returns:
            float: The average cross-entropy loss over the batch.
        """
        # If y_true are provided as class indices, convert to one-hot encoding
        if y_true.ndim == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]

        # Clip y_pred to avoid log(0)
        self.y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        self.y_true = y_true

        # Compute cross-entropy loss
        loss = -np.sum(y_true * np.log(self.y_pred)) / y_pred.shape[0]
        return loss

    def backward(self):
        """
        Backward pass for the cross-entropy loss.

        Returns:
            np.ndarray: Gradient of the loss with respect to the y_pred, of shape (N, C).
        """
        # Gradient of cross-entropy w.r.t y_pred
        grad_input = (self.y_pred - self.y_true) / self.y_pred.shape[0]
        return grad_input

