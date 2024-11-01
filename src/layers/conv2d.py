import numpy as np
from src import initializers
from src.layers.layer import Layer

class Conv2d(Layer):
    def __init__(
        self,
        kernel_size,
        stride: int = 1,
        padding: int = 0,
        kernel_initializer: str = 'glorot_uniform'
        ):
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.kernel_initializer = initializers.get(kernel_initializer)()

    def build(self, prev_units = None):
        '''
        **prev_units**: int, default = None
            For compatibility with Dense layer
        '''
        self.kernel = self.kernel_initializer(shape=self.kernel_size)
        
    def forward(self, prev_out: np.ndarray):
        out = convolution(matrix=prev_out, kernel=self.kernel, stride=self.stride, padding=self.padding)
        return out

    def backward(self):
        pass



def convolution(matrix:np.ndarray, kernel:np.ndarray, stride:int = 1, padding:int = 0) -> np.ndarray:
    n_samples = matrix.shape[0]
    n = matrix[0].shape[0]
    f = kernel.shape[0]

    size = int((n + 2*padding - f) / stride) + 1

    if padding != 0:
        matrix = np.pad(matrix, pad_width=((0,0), (padding,padding), (padding,padding)), constant_values=0)

    print(matrix.shape)
    out = np.array([np.sum(matrix[:, i:i+f, j:j+f] * kernel, axis=(1,2))
                    for j in range(0, size*stride, stride) 
                    for i in range(0, size*stride, stride)])
    
    return out.T.reshape(n_samples,size,size)