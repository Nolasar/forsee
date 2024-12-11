import numpy as np

class Sigmoid:
    """
    Sigmoid activation function.
    """
    def __call__(self, x):
        """
        Forward pass: Compute the sigmoid activation.

        Parameters
        ----------
        x : np.ndarray
            Input array.

        Returns
        -------
        np.ndarray
            Sigmoid activation: 1 / (1 + e^(-x)).
        """
        return 1 / (1 + np.exp(-x))

    def backward(self, sigm):
        """
        Backward pass: Compute the derivative of the sigmoid function.

        Parameters
        ----------
        sigm : np.ndarray
            Output of the sigmoid function.

        Returns
        -------
        np.ndarray
            Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x)).
        """
        return sigm * (1 - sigm)


class Softmax:
    """
    Softmax activation function.
    """
    def __call__(self, x):
        """
        Forward pass: Compute the softmax probabilities.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (N, C), where N is the batch size 
            and C is the number of classes.

        Returns
        -------
        np.ndarray
            Softmax probabilities of the same shape as x.
        """
        # Stabilize computation by subtracting max along each row
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)  # Store output for backward pass
        return self.out

    def backward(self, grad_output):
        """
        Backward pass: Compute the gradient of the loss with respect to the input.

        Parameters
        ----------
        grad_output : np.ndarray
            Gradient of the loss with respect to the output of the softmax function. 
            Shape is (N, C).

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to the input. Shape is (N, C).
        """
        # Batch multiply Jacobian with grad_output
        grad_input = self.out * (grad_output - np.sum(grad_output * self.out, axis=1, keepdims=True))

        return grad_input


class Relu:
    """
    ReLU (Rectified Linear Unit) activation function.
    """
    def __call__(self, x):
        """
        Forward pass: Compute the ReLU activation.

        Parameters
        ----------
        x : np.ndarray
            Input array of shape (N, C), where N is the batch size 
            and C is the number of features.

        Returns
        -------
        np.ndarray
            Output after applying ReLU, same shape as input x.
        """
        self.input = x  # Store input for backward pass
        self.out = np.maximum(0, x)  # ReLU: f(x) = max(0, x)
        return self.out

    def backward(self, grad_output):
        """
        Backward pass: Compute the gradient of the loss with respect to the input.

        Parameters
        ----------
        grad_output : np.ndarray
            Gradient of the loss with respect to the output of the ReLU function. 
            Shape is (N, C).

        Returns
        -------
        np.ndarray
            Gradient of the loss with respect to the input x. Shape is (N, C).
        """
        grad_input = grad_output * (self.input > 0).astype(np.float32)

        return grad_input

