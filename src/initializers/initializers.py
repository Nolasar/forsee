import numpy as np

class GlorotUniform:
    def __call__(self, shape, random_state = 42):
        rng = np.random.default_rng(random_state)
        limit = np.sqrt(6 / (shape[0] + shape[1]))
        return rng.uniform(low=-limit, high=limit, size=shape)   
    
class Zeros:
    def __call__(self, shape, random_state=None):
        return np.zeros(shape=shape)
    
