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

    def forward(self, image:np.ndarray):
        '''
        Parameters
        ----------
        prev_out : np.ndarray
            Output of the previous layer
            shape: (samples, channels, height, width)
        '''  
        samples, channels, h, w = image.shape

        # Add padding to input image
        image_pad = np.pad(
            array=image, 
            pad_width=((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), 
            mode='constant')
        
        self.image = image_pad

        # Rearrange the input image into a flattened column representation using `im2col` technique
        print('Im2col function ...')
        self.img_col = self._im2col(
            image=image_pad,
            window_size=self.kernel_size,
            output_size=self.output_size[1:],
            stride=self.stride
        )
        print('Im2col comleted!')

        # Flatten a kernels
        self.kernel_col = self.kernels.reshape(self.filters, -1)

        # Init output column repr
        output_col = np.empty((samples, self.filters, self.output_size[1] * self.output_size[2]))      

        # Perform the matrix multiplication between the flattened kernel and image columns.
        # This operation applies the convolution in a vectorized manner.
        print('matmul operation ...')
        for k in range(0, samples, self.chunk_size):
            chunk = self.img_col[k:k + self.chunk_size]
            output_col[k:k + self.chunk_size] = self.kernel_col @ chunk + self.bias
            print(f'{k+self.chunk_size} images processed ...')          

        return output_col.reshape(samples, self.filters, self.output_size[1], self.output_size[2])
    
    
    def backward(self, dout:np.ndarray, lr):
        samples = dout.shape[0]

        dout_col = dout.reshape(samples, self.filters, -1)

        dimg_col = self.kernel_col.T @ dout_col

        dimage = self._col2im(dimg_col, self.image.shape, self.kernel_size, self.output_size[1:], self.stride)

        dkernel = dout_col @ self.img_col.transpose(0, 2, 1)

        if self.padding > 0:
            dimage = dimage[:, :, self.padding:-self.padding, self.padding:-self.padding]

        return dimage
    
    def _im2col(self, image: np.ndarray, window_size: tuple, output_size: tuple, stride: int):
        """
        Transform the input image into column representation for efficient convolution.
        """
        samples, channels, _, _ = image.shape
        i, j, d = self._calculate_indices(window_size, output_size, channels, stride)
        col = image[:, d, i, j].reshape((samples, channels * np.prod(window_size), -1))
        return col
    
    def _col2im(self, cols: np.ndarray, image_shape: tuple, window_size: tuple, output_size: tuple, stride: int):
        samples, channels, H_im, W_im = image_shape
        h_k, w_k = window_size

        i, j, d = self._calculate_indices(window_size, output_size, channels, stride)
        cols_reshaped = cols.reshape(channels * h_k * w_k, -1, samples).transpose(2, 0, 1)
        image = np.zeros((samples, channels, H_im, W_im), dtype=cols.dtype)

        # # Debugging output
        # print(f"Calculated indices:\ni: {i.shape}, j: {j.shape}, d: {d.shape}")
        # print(f"Max index values: i: {i.max()}, j: {j.max()}, d: {d.max()}")
        # print(f"Image shape: {image.shape}")

        # # Check if indices exceed bounds
        # if i.max() >= H_im or j.max() >= W_im:
        #     raise ValueError(f"Index out of bounds: i.max()={i.max()}, H_im={H_im}, j.max()={j.max()}, W_im={W_im}")

        np.add.at(image, (slice(None), d, i, j), cols_reshaped)

        return image

    
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
        """
        h_k, w_k = window_size
        h_out, w_out = output_size

        i0 = np.repeat(np.arange(h_k), w_k)
        i0 = np.tile(i0, channels)
        i1 = stride * np.repeat(np.arange(h_out), w_out)
        i = i0.reshape(-1, 1) + i1.reshape(1, -1)

        j0 = np.tile(np.arange(w_k), h_k * channels)
        j1 = stride * np.tile(np.arange(w_out), h_out)
        j = j0.reshape(-1, 1) + j1.reshape(1, -1)

        d = np.repeat(np.arange(channels), h_k * w_k).reshape(-1, 1)

        return i, j, d


