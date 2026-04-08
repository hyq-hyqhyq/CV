import numpy as np

try:
    import cupy as cp
except ImportError:
    cp = None

_BACKEND = np


def set_backend(name):
    # Switch array backend between numpy(cpu) and cupy(gpu).
    global _BACKEND
    name = name.lower()
    if name == "cpu":
        _BACKEND = np
        return
    if name == "gpu":
        if cp is None:
            raise ImportError("CuPy is not installed. Please install cupy-cuda11x/12x first.")
        _BACKEND = cp
        return
    raise ValueError("backend must be 'cpu' or 'gpu'")


def get_backend_name():
    return "gpu" if (_BACKEND is cp and cp is not None) else "cpu"


def to_numpy(x):
    # Convert tensor/array to numpy for saving/plotting.
    if isinstance(x, Tensor):
        x = x.data
    if cp is not None and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return np.asarray(x)


def _to_backend_array(x):
    if isinstance(x, Tensor):
        x = x.data
    return _BACKEND.asarray(x, dtype=_BACKEND.float32)


def _unbroadcast(grad, shape):
    # Sum gradients over broadcasted dimensions.
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    for i, (gdim, sdim) in enumerate(zip(grad.shape, shape)):
        if sdim == 1 and gdim != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    # Minimal autograd tensor for MLP training.
    def __init__(self, data, requires_grad=False, _children=(), _op=""):
        self.data = _to_backend_array(data)
        self.requires_grad = requires_grad
        self.grad = _BACKEND.zeros_like(self.data, dtype=_BACKEND.float32) if self.requires_grad else None
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def zero_grad(self):
        if self.requires_grad:
            xp = self.data.__class__.__module__.split(".")[0]
            if xp == "cupy" and cp is not None:
                self.grad = cp.zeros_like(self.data, dtype=cp.float32)
            else:
                self.grad = np.zeros_like(self.data, dtype=np.float32)

    def backward(self):
        if self.data.size != 1:
            raise ValueError("backward() can only be called on scalar tensors.")
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        if cp is not None and isinstance(self.data, cp.ndarray):
            self.grad = cp.ones_like(self.data, dtype=cp.float32)
        else:
            self.grad = np.ones_like(self.data, dtype=np.float32)
        for node in reversed(topo):
            node._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad, (self, other), "+")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        out = Tensor(-self.data, self.requires_grad, (self,), "neg")

        def _backward():
            if self.requires_grad:
                self.grad -= out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad, (self, other), "*")

        def _backward():
            if self.requires_grad:
                self.grad += _unbroadcast(other.data * out.grad, self.data.shape)
            if other.requires_grad:
                other.grad += _unbroadcast(self.data * out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * other.pow(-1.0)

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad, (self, other), "matmul")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad @ other.data.T
            if other.requires_grad:
                other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def T(self):
        out = Tensor(self.data.T, self.requires_grad, (self,), "transpose")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.T

        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), self.requires_grad, (self,), "reshape")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad, (self,), "sum")

        def _backward():
            if self.requires_grad:
                grad = out.grad
                if axis is not None and not keepdims:
                    if isinstance(axis, tuple):
                        for ax in sorted(axis):
                            if cp is not None and isinstance(grad, cp.ndarray):
                                grad = cp.expand_dims(grad, ax)
                            else:
                                grad = np.expand_dims(grad, ax)
                    else:
                        if cp is not None and isinstance(grad, cp.ndarray):
                            grad = cp.expand_dims(grad, axis)
                        else:
                            grad = np.expand_dims(grad, axis)
                if cp is not None and isinstance(self.data, cp.ndarray):
                    self.grad += cp.broadcast_to(grad, self.data.shape)
                else:
                    self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        denom = self.data.size if axis is None else np.prod(np.array(self.data.shape)[axis])
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / float(denom))

    def exp(self):
        if cp is not None and isinstance(self.data, cp.ndarray):
            out_data = cp.exp(self.data)
        else:
            out_data = np.exp(self.data)
        out = Tensor(out_data, self.requires_grad, (self,), "exp")

        def _backward():
            if self.requires_grad:
                self.grad += out_data * out.grad

        out._backward = _backward
        return out

    def log(self):
        if cp is not None and isinstance(self.data, cp.ndarray):
            out = Tensor(cp.log(self.data + 1e-12), self.requires_grad, (self,), "log")
        else:
            out = Tensor(np.log(self.data + 1e-12), self.requires_grad, (self,), "log")

        def _backward():
            if self.requires_grad:
                self.grad += out.grad / (self.data + 1e-12)

        out._backward = _backward
        return out

    def pow(self, power):
        if cp is not None and isinstance(self.data, cp.ndarray):
            out = Tensor(cp.power(self.data, power), self.requires_grad, (self,), "pow")
        else:
            out = Tensor(np.power(self.data, power), self.requires_grad, (self,), "pow")

        def _backward():
            if self.requires_grad:
                if cp is not None and isinstance(self.data, cp.ndarray):
                    self.grad += (power * cp.power(self.data, power - 1.0)) * out.grad
                else:
                    self.grad += (power * np.power(self.data, power - 1.0)) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        if cp is not None and isinstance(self.data, cp.ndarray):
            out = Tensor(cp.maximum(self.data, 0.0), self.requires_grad, (self,), "relu")
        else:
            out = Tensor(np.maximum(self.data, 0.0), self.requires_grad, (self,), "relu")

        def _backward():
            if self.requires_grad:
                if cp is not None and isinstance(self.data, cp.ndarray):
                    self.grad += (self.data > 0).astype(cp.float32) * out.grad
                else:
                    self.grad += (self.data > 0).astype(np.float32) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self):
        if cp is not None and isinstance(self.data, cp.ndarray):
            sig = 1.0 / (1.0 + cp.exp(-self.data))
        else:
            sig = 1.0 / (1.0 + np.exp(-self.data))
        out = Tensor(sig, self.requires_grad, (self,), "sigmoid")

        def _backward():
            if self.requires_grad:
                self.grad += sig * (1.0 - sig) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        if cp is not None and isinstance(self.data, cp.ndarray):
            t = cp.tanh(self.data)
        else:
            t = np.tanh(self.data)
        out = Tensor(t, self.requires_grad, (self,), "tanh")

        def _backward():
            if self.requires_grad:
                self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out
