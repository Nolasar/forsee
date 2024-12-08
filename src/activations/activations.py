import numpy as np

class Sigmoid:
    def __call__(self, x):
        '''
        Forward pass: Compute the sigmoid activation.
        
        Parameters
        ----------
        - x: argument of sigmoid ( 1 / (1 + e^(-x)))
        '''
        return  1 / (1 + np.exp(-x))

    def backward(self, sigm):
        '''
        Backward pass: Compute the derivative of the sigmoid function.
        The derivative is sigmoid(x) * (1 - sigmoid(x)).

        Parameters
        ----------
        - sigm: output of sigmoid (sigm = sigmoid(x))
        '''
        return sigm * (1 - sigm)
    

class Softmax:
    def __call__(self, x):
        """
        Forward pass for the softmax function.
        
        Args:
            x (np.ndarray): Input array of shape (N, C), where N is the batch size 
                            and C is the number of classes.
                            
        Returns:
            np.ndarray: Softmax probabilities of the same shape as x.
        """
        # Stabilize computation by subtracting max along each row
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)  # Store output for backward pass
        return self.out


    def backward(self, grad_output):
        """
        Backward pass for the softmax function using broadcasting.

        Args:
            grad_output (np.ndarray): Gradient of the loss with respect to the output 
                                    of the softmax function. Shape is (N, C).

        Returns:
            np.ndarray: Gradient of the loss with respect to the input x. Shape is (N, C).
        """
        # self.out shape: (N, C)
        N, C = self.out.shape

        # Compute the outer product y * y^T for each sample
        y = self.out[:, :, np.newaxis]  # Shape (N, C, 1)
        jacobian = np.eye(C) * y - np.matmul(y, y.transpose(0, 2, 1))  # Shape (N, C, C)

        # Batch multiply Jacobian with grad_output
        grad_input = np.einsum("nij,nj->ni", jacobian, grad_output)  # Shape (N, C)

        return grad_input
