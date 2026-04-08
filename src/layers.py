import numpy as np

from src.autograd import Tensor


class Linear:
    # Fully connected layer: y = xW + b
    def __init__(self, in_features, out_features, rng):
        scale = np.sqrt(2.0 / in_features)
        self.W = Tensor(rng.normal(0.0, scale, size=(in_features, out_features)), requires_grad=True)
        self.b = Tensor(np.zeros((1, out_features), dtype=np.float32), requires_grad=True)

    def __call__(self, x):
        return x @ self.W + self.b

    def parameters(self):
        return [self.W, self.b]
