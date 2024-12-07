import numpy as np
from src import initializers
from src.layers.layer import Layer

class Conv2d(Layer):
    def __init__(
        self,
        filters: int,
        kernel_size: tuple[int, int],
        stride: int = 1,
        padding: int = 0,
        chunk_size: int = 1_000,
        kernel_initializer: str = 'glorot_uniform',
        bias_initializer: str = 'zeros'
        ):
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.chunk_size = chunk_size
        self.kernel_initializer = initializers.get(kernel_initializer)()
        self.bias_initializer = initializers.get(bias_initializer)()
        self.output_size = None

    def build(self, input_size):
        '''
        Parameters
        ----------
        **channels_in**: int
            Number of input channels
        '''
        channels_in, h, w = input_size

        h_o = (h + 2*self.padding - self.kernel_size[0]) // self.stride + 1
        w_o = (w + 2*self.padding - self.kernel_size[1]) // self.stride + 1

        self.output_size = (self.filters, h_o, w_o)

        print('build ...')
        self.kernels = self.kernel_initializer(shape=(self.filters, channels_in, *self.kernel_size))    
        self.bias = self.bias_initializer(shape=(self.filters, 1))
        print('build ends')

    def forward(self, prev_out: np.ndarray):
        '''
        Parameters
        ----------
        prev_out : np.ndarray
            Output of the previous layer
            shape: (samples, channels, height, width)
        '''
        return self._convolution(
            image=prev_out,
            kernels=self.kernels,
            chunk_size=self.chunk_size,
            stride=self.stride,
            padding=self.padding
            )
    
    def backward(self):
        pass
    
    
    def _convolution(self, image:np.ndarray, kernels:np.ndarray, chunk_size:int=1_000, stride:int=1, padding:int=0):
        """
        Performs a convolution (cross-correlation in terms of math) between image and kernel.

        Parameters
        ----------
        image : np.ndarray
            Numpy array with shape (samples, channels, H_im, W_im)
        kernel : np.ndarray
            Numpy array with shape (filters, channels, H_k, W_k)
        chunk_size : int
            Size of chunks. Chunks need for optimize matmul operation of large matrices
        stride : int
            Step size of the sliding window in the convolution.
        padding : int
            Amount of padding to apply to the input array before convolution.

        Returns
        -------
        output : np.ndarray
            Result of convolution. Numpy array with shape (samples, filters, H_out, W_out)

        Notes
        -----
        - H_out = (H_im + 2 * padding - H_k) // stride + 1
        - W_out = (W_im + 2 * padding - W_k) // stride + 1
        """
        # Params of input image and kernels
        samples, channels, H_im, W_im = image.shape
        filters, _, H_k, W_k = kernels.shape

        # Output shape
        H_out = (H_im + 2 * padding - H_k) // stride + 1
        W_out = (W_im + 2 * padding - W_k) // stride + 1    

        # Add padding to input image
        if padding != 0:
            image = np.pad(image, pad_width=((0,0), (0,0), (padding,padding), (padding,padding)), mode='constant')
        
        # Calculate indices if not provided
        print('generate indices ...')
        (i, j, d) = self._calculate_indices(window_size=(H_k, W_k), output_size=(H_out, W_out), channels=channels, stride=stride)
        print('completed!')

        print('flatten image ...')
        # Rearrange the input image into a flattened column representation using `im2col` technique
        img_col = image[:, d, i, j].reshape((samples, channels * H_k * W_k, -1))     
        print('comleted!')

        # Flatten the kernel into a 2D matrix where each row corresponds to a single filter
        kernel_col = kernels.reshape((filters, -1))
       
        # Perform the matrix multiplication between the flattened kernel and image columns.
        # This operation applies the convolution in a vectorized manner.
        output_col = np.empty((samples, filters, H_out * W_out))

        print('matmul operation ...')
        for k in range(0, samples, chunk_size):
            chunk = img_col[k:k+chunk_size]
            output_col[k:k+chunk_size] = kernel_col @ chunk + self.bias

            del chunk
            print(f'{k+chunk_size} images processed ...')
        
        # Reshape the result back into the output tensor shape, aligning with the convolutional layer's expected dimensions.
        output = output_col.reshape((samples, filters, H_out, W_out))

        del output_col, img_col, kernel_col

        return output
    
    
    def _calculate_indices(self, window_size: tuple, output_size: tuple, channels: int, stride:int = 1):
        """
        Calculate flattened indices for extracting patches from an input array.

        This function computes the row, column, and depth indices required to transform 
        an input array into a flattened column format (e.g., for `im2col` operations). 
        These indices correspond to the spatial locations of the sliding window patches 
        and their respective channels.

        Parameters
        ----------
        window_size : tuple of int
            The height and width of the sliding window (kernel size), represented as 
            `(h_k, w_k)`.
        output_size : tuple of int
            The height and width of the output feature map, represented as `(h_out, w_out)`.
        channels : int
            The number of channels in the input array.
        stride : int, optional
            The step size for sliding the window across the input array (default is 1).

        Returns
        -------
        i1 : np.ndarray
            The flattened row indices of the patches for all channels.
        j1 : np.ndarray
            The flattened column indices of the patches for all channels.
        d : np.ndarray
            The channel indices repeated for each patch.

        Notes
        -----
        The indices generated by this function can be used to extract patches from an input 
        array or to reconstruct the input from patches. It is particularly useful for 
        implementing operations like `im2col` efficiently.

        Examples
        --------
        Compute indices for a 3x3 sliding window with an output size of (4, 4), 3 channels, 
        and a stride of 1:

        >>> window_size = (3, 3)
        >>> output_size = (4, 4)
        >>> channels = 3
        >>> stride = 1
        >>> i1, j1, d = calculate_indices(window_size, output_size, channels, stride)
        >>> i1.shape, j1.shape, d.shape
        ((144,), (144,), (144,))
        """
        h_k, w_k = window_size
        h_out, w_out = output_size

        i0 = np.concatenate([np.tile(np.repeat(np.arange(i*stride, i*stride + h_k), w_k), w_out) for i in range(h_out)])

        j0 = np.tile(np.concatenate([np.tile(np.arange(j*stride, j*stride + w_k), h_k) for j in range(w_out)]), h_out)

        i1 = np.tile(i0, channels)
        j1 = np.tile(j0, channels)

        no_idx = i0.shape[0]
        d = np.repeat(np.arange(channels), no_idx)

        del i0, j0

        return (i1, j1, d)
