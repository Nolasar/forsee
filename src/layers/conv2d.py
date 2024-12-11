import numpy as np
from src import initializers, activations
from src.layers.layer import Layer

class Conv2d(Layer):
    """
    A 2D convolutional layer with support for forward and backward passes.
    """

    def __init__(
        self,
        channels_out: int,
        kernel_size: tuple[int, int],
        activation: str = 'relu',
        kernel_initializer: str = 'glorot_uniform',
        bias_initializer: str = 'zeros',
        stride: int = 1,
        padding: int = 0,
        random_state: int = 42
    ):
        """
        Initialize the Conv2d layer.

        Parameters
        ----------
        channels_out : int
            Number of output channels (filters).
        kernel_size : tuple[int, int]
            Size of the convolutional kernel (height, width).
        activation : str, optional
            Activation function, by default 'relu'.
        kernel_initializer : str, optional
            Initializer for the kernel weights, by default 'glorot_uniform'.
        bias_initializer : str, optional
            Initializer for the bias, by default 'zeros'.
        stride : int, optional
            Stride of the convolution, by default 1.
        padding : int, optional
            Padding size, by default 0.
        """
        self.channels_out = channels_out
        self.kernel_size = kernel_size
        self.activation = activations.get(activation)() if activation is not None else None
        self.kernel_initializer = initializers.get(kernel_initializer)()
        self.bias_initializer = initializers.get(bias_initializer)()
        self.stride = stride
        self.pad = padding
        self.rnd_state = random_state

    def build(self, input_size: tuple):
        """
        Build the layer by initializing kernel, bias, and output dimensions.

        Parameters
        ----------
        input_size : tuple
            Shape of the input as (channels_in, height, width).
        """
        self.input_size = (input_size[0], input_size[1] + 2 * self.pad, input_size[2] + 2 * self.pad)

        h_out = (self.input_size[1] - self.kernel_size[0]) // self.stride + 1
        w_out = (self.input_size[2] - self.kernel_size[1]) // self.stride + 1

        self.output_size = (self.channels_out, h_out, w_out)

        # Initialize kernel and bias
        self.kernel = self.kernel_initializer(shape=(self.channels_out, self.input_size[0], *self.kernel_size), 
                                              random_state=self.rnd_state)
        self.bias = self.bias_initializer(shape=(self.channels_out,), random_state=self.rnd_state)

        # Initialize the Functional utility
        self.tools = Functional()

    def forward(self, image: np.ndarray):
        """
        Perform the forward pass of the Conv2d layer.

        Parameters
        ----------
        image : np.ndarray
            Input image of shape (samples, channels_in, height, width).

        Returns
        -------
        output : np.ndarray
            Output of the convolutional layer with shape (samples, channels_out, h_out, w_out).
        """
        samples = image.shape[0]

        # Apply padding if required
        if self.pad != 0:
            image_pad = np.pad(
                array=image,
                pad_width=((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad)),
                mode='constant'
            )
        else:
            image_pad = image

        # Convert the image to columnar format
        self.image_col = self.tools.im2col(image=image_pad, window_size=self.kernel_size, stride=self.stride)

        # Flatten the kernel for matrix multiplication
        self.kernel_col = self.kernel.reshape(self.channels_out, -1)

        # Perform convolution as a matrix multiplication

        # Apply activation if specified
        self.pre_activation = (self.kernel_col @ self.image_col) + self.bias.reshape(-1, 1)
 
        if self.activation is not None:
            self.out_activated = self.activation(self.pre_activation)
        else:
            self.out_activated = self.pre_activation
        
        # Reshape the output to its final shape
        return self.out_activated.reshape(samples, self.channels_out, *self.output_size[1:])


    def backward(self, dout: np.ndarray, lr):
        """
        Perform the backward pass of the Conv2d layer.

        Parameters
        ----------
        dout : np.ndarray
            Gradient of the loss with respect to the output, shape (samples, channels_out, h_out, w_out).

        Returns
        -------
        dimage : np.ndarray
            Gradient of the loss with respect to the input, shape (samples, channels_in, h_in, w_in).
        """
        samples = dout.shape[0]

        # Reshape the gradient to columnar format
        dout_col = dout.reshape(samples, self.channels_out, -1)

        # Apply activation's derivative

        if self.activation is not None:
            dactivation = self.activation.backward(self.pre_activation)  
            dout_col = dactivation * dout_col

        # Compute gradient of the kernel
        dkernel_col = np.einsum('scm,snm->cn', dout_col, self.image_col)
        self.dkernel = dkernel_col.reshape(self.channels_out, self.input_size[0], *self.kernel_size)

        # Compute gradient of the bias
        self.dbias = np.einsum('schw->c', dout)

        # Compute gradient of the input
        dimage_col = np.einsum('cn,scm->snm', self.kernel_col, dout_col)

        # Convert columnar gradient back to image format
        dimage = self.tools.col2im(
            columnar=dimage_col,
            window_size=self.kernel_size,
            image_size=(samples, *self.input_size),
            stride=self.stride
        )

        # Remove padding if applied
        if self.pad > 0:
            dimage = dimage[:, :, self.pad:-self.pad, self.pad:-self.pad]

        return dimage

    def get_params(self):
        """
        Get the parameters of the Conv2d layer.

        Returns
        -------
        dict
            Dictionary containing {'kernel': self.kernel, 'bias': self.bias}.
        """
        return {'kernel': self.kernel, 'bias': self.bias}

    def get_grads(self):
        """
        Get the gradients of the Conv2d layer.

        Returns
        -------
        dict
            Dictionary containing {'kernel': self.dkernel, 'bias': self.dbias}.
        """
        return {'kernel': self.dkernel, 'bias': self.dbias}



class Functional:
    """
    A utility class for handling operations like im2col and col2im
    for convolutional layers.
    """

    def __init__(self):
        """
        Initialize the Functional utility class.
        """
        pass

    def _generate_indices(self, image_size: tuple, window_size: tuple, stride: int):
        """
        Generate indices for slicing image patches for im2col.

        Parameters
        ----------
        image_size : tuple
            Shape of the image as (batch_size, channels, height, width).
        window_size : tuple
            Size of the convolution kernel as (kernel_height, kernel_width).
        stride : int
            Stride of the convolution.

        Returns
        -------
        i : ndarray
            Row indices for slicing patches.
        j : ndarray
            Column indices for slicing patches.
        d : ndarray
            Depth indices for slicing patches.
        """
        _, channels, h_in, w_in = image_size
        h_k, w_k = window_size

        # Compute output dimensions
        h_out = (h_in - h_k) // stride + 1
        w_out = (w_in - w_k) // stride + 1

        # Create row indices for patches
        i0 = np.repeat(np.arange(h_k), w_k)
        i0 = np.tile(i0, channels)
        i1 = stride * np.repeat(np.arange(h_out), w_out)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)

        # Create column indices for patches
        j0 = np.tile(np.arange(w_k), h_k * channels)
        j1 = stride * np.tile(np.arange(w_out), h_out)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        # Create depth indices for patches
        d = np.repeat(np.arange(channels), h_k * w_k).reshape(-1, 1)

        return i, j, d

    def im2col(self, image: np.ndarray, window_size: tuple, stride: int):
        """
        Convert an image to columnar format for matrix multiplication.

        Parameters
        ----------
        image : np.ndarray
            Input image of shape (batch_size, channels, height, width).
        window_size : tuple
            Size of the convolution kernel as (kernel_height, kernel_width).
        stride : int
            Stride of the convolution.

        Returns
        -------
        columnar : np.ndarray
            Image patches in columnar format of shape
            (batch_size, kernel_size * channels, output_size).
        """
        samples, channels = image.shape[:2]

        # Generate slicing indices
        i, j, d = self._generate_indices(
            image_size=image.shape,
            window_size=window_size,
            stride=stride
        )

        # Extract patches and reshape into columnar format
        columnar = image[:, d, i, j].reshape((samples, np.prod(window_size) * channels, -1))
        return columnar

    def col2im(self, columnar: np.ndarray, window_size: tuple, image_size: tuple, stride: int):
        """
        Convert columnar patches back to image format.

        Parameters
        ----------
        columnar : np.ndarray
            Columnar patches of shape (batch_size, kernel_size * channels, output_size).
        window_size : tuple
            Size of the convolution kernel as (kernel_height, kernel_width).
        image_size : tuple
            Shape of the original image as (batch_size, channels, height, width).
        stride : int
            Stride of the convolution.

        Returns
        -------
        image : np.ndarray
            Reconstructed image of shape (batch_size, channels, height, width).
        """
        # Generate slicing indices
        i, j, d = self._generate_indices(
            image_size=image_size,
            window_size=window_size,
            stride=stride
        )

        # Initialize empty image
        image = np.zeros(shape=image_size, dtype=columnar.dtype)

        # Add values back to image using columnar patches
        np.add.at(image, (slice(None), d, i, j), columnar)
        return image
