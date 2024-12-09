class Layer:
    """
    Abstract base class for neural network layers.
    All layers should inherit from this class and implement required methods.
    """

    def __init__(self):
        """
        Initialize a base layer.
        """
        pass

    def build(self):
        """
        Build the layer (e.g., initialize weights and compute output shapes).
        This method must be implemented in derived classes.
        
        Raises
        ------
        NotImplementedError
            If not implemented in a derived class.
        """
        raise NotImplementedError("The build method must be implemented by the subclass.")

    def forward(self):
        """
        Perform the forward pass for the layer.
        This method must be implemented in derived classes.
        
        Raises
        ------
        NotImplementedError
            If not implemented in a derived class.
        """
        raise NotImplementedError("The forward method must be implemented by the subclass.")

    def backward(self):
        """
        Perform the backward pass for the layer.
        This method must be implemented in derived classes.
        
        Raises
        ------
        NotImplementedError
            If not implemented in a derived class.
        """
        raise NotImplementedError("The backward method must be implemented by the subclass.")

    def get_params(self):
        """
        Get trainable parameters of the layer.

        Returns
        -------
        dict
            A dictionary containing parameter names as keys and references
            to the parameters as values. Default implementation returns an empty dictionary.
        """
        return {}

    def get_grads(self):
        """
        Get gradients of trainable parameters of the layer.

        Returns
        -------
        dict
            A dictionary containing parameter names as keys and references
            to the gradients as values. Default implementation returns an empty dictionary.
        """
        return {}
