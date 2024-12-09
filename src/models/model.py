from typing import List 
from src.layers.layer import Layer
from src.layers.dense import Dense
from src.layers.conv2d import Conv2d
from src.layers.flatten import Flatten
from src.layers.maxpool import MaxPool2D
from src.optimazers.adam import Adam

class Sequential:
    """
    Sequential model implementation for building and training a neural network.
    """

    def __init__(self, *args):
        """
        Initialize the Sequential model with a list of layers.

        Parameters
        ----------
        *args : List[Layer] or Layer
            Instances of the Layer class in the order they are to be connected.
        """
        self.layers = [*args]  # Store the layers in order
        self.history = {
            'loss': [],
            'metric': []
        }  # Dictionary to save training loss and metric history

    def _build(self, X):
        """
        Connect layers with each other and initialize their weights.

        Parameters
        ----------
        X : np.ndarray
            Input data to infer the input shape for each layer.
        """
        input_size = X.shape[1:]  # Infer input shape from the feature matrix

        for layer in self.layers:
            # Build the current layer
            layer.build(input_size)

            # Update input size for the next layer
            if isinstance(layer, (Dense, Flatten)):
                input_size = layer.units
            elif isinstance(layer, (Conv2d, MaxPool2D)):
                input_size = layer.output_size

    def _feedforward(self, X):
        """
        Perform forward propagation through all layers.

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Output of the last layer.
        """
        prev_out = X  # Store output of the previous layer
        for layer in self.layers:
            prev_out = layer.forward(prev_out)  # Pass through each layer
        return prev_out

    def _backprop(self, loss_grad):
        """
        Perform backward propagation through all layers.

        Parameters
        ----------
        loss_grad : np.ndarray
            Gradient of the loss with respect to the output layer.
        """
        output_grad = loss_grad  # Store the gradient for the current layer
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad)  # Backpropagate gradient

    def train(self, X, y, loss_fn, metric_fn, epochs=250, lr=0.001) -> None:
        """
        Train the model using the provided loss and metric functions.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target vector.
        loss_fn : class <loss>
            Loss function instance (e.g., MSE).
        metric_fn : class <metric>
            Metric function instance (e.g., Accuracy).
        epochs : int, optional, default=250
            Number of training iterations.
        lr : float, optional, default=0.001
            Learning rate.

        Returns
        -------
        None
        """
        self._build(X)  # Initialize weights and biases

        optimizer = Adam(self.layers, lr)  # Create an Adam optimizer instance

        for epoch in range(epochs):
            # Perform forward pass
            y_pred = self._feedforward(X)

            # Compute the loss
            loss = loss_fn(y_true=y, y_pred=y_pred)

            # Compute the gradient of the loss
            dloss = loss_fn.backward()

            # Perform backward pass
            self._backprop(dloss)

            # Update weights using the optimizer
            optimizer.step()

            # Calculate the metric
            metric = metric_fn(y_true=y, y_pred=y_pred)

            # Save loss and metric history
            self.history['loss'].append(loss)
            self.history['metric'].append(metric)

            # Print progress
            print(f'Epoch {epoch} | Loss {loss:.6f} | Metric {metric:.3f}')

    def predict(self, X):
        """
        Perform a forward pass to generate predictions.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.

        Returns
        -------
        np.ndarray
            Model predictions.
        """
        return self._feedforward(X)  # Generate predictions using forward pass
