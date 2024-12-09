import numpy as np

class BinaryCrossEntropy:
    """
    Binary Cross-Entropy (BCE) loss function for binary classification tasks.
    """
    def __call__(self, y_pred, y_true):
        """
        Compute the forward pass for binary cross-entropy loss.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted probabilities, shape (N,), where N is the batch size.
        y_true : np.ndarray
            True labels, shape (N,).

        Returns
        -------
        float
            The binary cross-entropy loss averaged over the batch.
        """
        # Store predictions and true labels for backward pass
        self.y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)  # Clip to avoid log(0)
        self.y_true = y_true

        # Compute binary cross-entropy loss
        return np.mean(
            -y_true * np.log(self.y_pred) - (1 - y_true) * np.log(1 - self.y_pred)
        )
    
    def backward(self):
        """
        Compute the backward pass for binary cross-entropy loss.

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to the predictions (y_pred), shape (N,).
        """
        # Gradient computation for BCE
        return -self.y_true / self.y_pred + (1 - self.y_true) / (1 - self.y_pred)


class CrossEntropy:
    """
    Cross-Entropy (CE) loss function for multi-class classification tasks.
    """
    def __call__(self, y_pred, y_true):
        """
        Compute the forward pass for cross-entropy loss.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted probabilities, shape (N, C), where N is the batch size
            and C is the number of classes.
        y_true : np.ndarray
            True labels, shape (N, C) in one-hot encoding or (N,) as class indices.

        Returns
        -------
        float
            The average cross-entropy loss over the batch.
        """
        # If y_true is provided as class indices, convert to one-hot encoding
        if y_true.ndim == 1:
            y_true = np.eye(y_pred.shape[1])[y_true]

        # Clip y_pred to avoid log(0)
        self.y_pred = np.clip(y_pred, 1e-12, 1 - 1e-12)
        self.y_true = y_true

        # Compute cross-entropy loss
        return -np.sum(y_true * np.log(self.y_pred)) / y_pred.shape[0]

    def backward(self):
        """
        Compute the backward pass for cross-entropy loss.

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to the predictions (y_pred),
            shape (N, C).
        """
        # Gradient computation for CE
        return (self.y_pred - self.y_true) / self.y_pred.shape[0]


