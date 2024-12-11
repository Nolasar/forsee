import numpy as np
from src.layers.layer import Layer
from src.layers.conv2d import Functional

class MaxPool2d(Layer):
    def __init__(
        self,
        kernel_size:tuple[int, int],
        stride:int=1
        ):
        self.kernel_size = kernel_size
        self.stride = stride
        super().__init__()

    def build(self, input_size:tuple):
        """
        Parameters
        ----------
        input_size : tuple
            shape (channnels, h_in, w_in)

        """
        self.input_size = input_size

        self.output_size = (self.input_size[0], 
                            (self.input_size[1] - self.kernel_size[0]) // self.stride + 1, 
                            (self.input_size[2] - self.kernel_size[1]) // self.stride + 1)
        
        self.tools = Functional()
        
    def forward(self, image:np.ndarray):
        samples = image.shape[0]

        self.image_col = self.tools.im2col(image, self.kernel_size, self.stride)
        
        image_col_reshaped = self.image_col.reshape(
            samples, 
            self.input_size[0],
            np.prod(self.kernel_size), 
            np.prod(self.output_size[1:])
            )

        output = np.max(image_col_reshaped, axis=2)

        self.idx_max = np.argmax(image_col_reshaped, axis=2)

        return output.reshape(samples, *self.output_size)
    
    def backward(self, dout:np.ndarray, lr):
        """
        Parameters
        ----------
        dout : np.ndarray
            shape (samples, channels, h_out, w_out)

        """
        samples, channels, h_out, w_out = dout.shape

        dout_col = dout.reshape(samples, channels, -1)

        batch_idx = np.arange(samples)[:, np.newaxis, np.newaxis]
        channels_idx = np.arange(channels)[np.newaxis, :, np.newaxis]
        hw_idx = np.arange(h_out*w_out)[None, None, :]

        dimage_col = np.zeros_like(self.image_col)
        
        np.add.at(dimage_col, (batch_idx, channels_idx*np.prod(self.kernel_size) + self.idx_max, hw_idx), dout_col)

        dimage = self.tools.col2im(dimage_col, self.kernel_size, (samples, *self.input_size), self.stride)
        
        return dimage
