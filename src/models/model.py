from typing import List 
from src.layers.layer import Layer
from src.layers.dense import Dense
from src.layers.conv2d import Conv2d

class Sequential:
    def __init__(self, *args):
        '''
        Parameters
        ----------
        ***args**: List[Layer] or Layer
            Takes instances of Layer class in input order
        '''
        self.layers: List[Layer] = [*args]
        self.history = {
            'loss': [],
            'metric': []
        }

    def _build(self, X):
        '''
        Connect layers with each other and initialize weights
        '''
        if isinstance(self.layers[0], Conv2d):
            if len(X.shape) == 3:
                X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])

        prev_units = X.shape[1]

        for layer in self.layers:
            layer.build(prev_units)
            
            if isinstance(layer, Dense):
                prev_units = layer.units
            elif isinstance(layer, Conv2d):
                prev_units = layer.filters


    def _feedforward(self, X):
        '''
        Implementation of forward propagation  

        '''
        prev_out = X
        for layer in self.layers:
            prev_out = layer.forward(prev_out)

        return prev_out
    
    def _backprop(self, loss_grad, lr):
        '''
        Implementation of backward propagation  

        '''
        prev_grad = loss_grad
        for layer in reversed(self.layers):
            prev_grad = layer.backward(prev_grad, lr)

    def train(self, X, y, loss_fn, metric_fn, epochs = 250, lr = 0.001) -> None:
        '''
        Main function for training model

        Parameters
        ----------
        **X**: np.ndarray
            feature matrix
        **y**: np.ndarray
            target vector
        **loss_fn**: class \<loss\>
            loss function (`MSE`, for example)
        **metric_fn**: class \<metric\>
            metric function (`Accuracy`, for example)
        **epochs**: int, optional, default = 250
            number of learning iterations
        **lr**: float, optional, default = 0.001
            learning rate

        Returns
        -------
        None
        '''
        self._build(X) # initialize weights 

        for epoch in range(epochs):

            y_pred = self._feedforward(X) # calculate predictions 

            loss = loss_fn(y_true=y, y_pred=y_pred) # compute loss

            dloss = loss_fn.backward(y_true=y, y_pred=y_pred) # compute gradient of loss

            self._backprop(dloss, lr) # compute grads and update weights and biases
            
            metric = metric_fn(y_true=y, y_pred=y_pred) # calculate metric
            
            self.history['loss'].append(loss) # save loss history 
            self.history['metric'].append(metric) # save metric history

            print(f'Epoch {epoch} | Loss {loss:.6f} | Metric {metric: .3f}')

    
    def predict(self, X):
        return self._feedforward(X)