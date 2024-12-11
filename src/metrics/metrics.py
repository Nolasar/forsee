import numpy as np

class Accuracy:
    """
    Accuracy metric for multiclass classification.
    """
    def __call__(self, y_pred, y_true):
        """
        Calculate the accuracy for multiclass predictions.

        Parameters
        ----------
        y_pred : np.ndarray
            Predicted probabilities or logits of shape (N, C),
            where N is the number of samples, and C is the number of classes.
        y_true : np.ndarray
            True labels of shape (N,) or (N, C).
            If shape is (N,), labels are class indices.
            If shape is (N, C), labels are one-hot encoded.

        Returns
        -------
        float
            Accuracy as the fraction of correctly predicted samples.
        """
        # If y_true is one-hot encoded, convert to class indices
        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)
        
        # Convert predicted probabilities to class indices
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Compare predicted classes with true labels
        correct_predictions = np.sum(y_pred_classes == y_true)

        # Calculate accuracy as the fraction of correct predictions
        total_samples = y_true.shape[0]
        return correct_predictions / total_samples
