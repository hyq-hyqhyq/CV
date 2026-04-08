import numpy as np

from src.autograd import Tensor
from src.layers import Linear


class MLP3:
    # 3-layer MLP (2 hidden layers + 1 output layer).
    def __init__(self, input_dim, hidden_dim, num_classes, activation="relu", seed=42):
        rng = np.random.default_rng(seed)
        if isinstance(hidden_dim, int):
            h1, h2 = hidden_dim, hidden_dim
        else:
            h1, h2 = hidden_dim
        self.fc1 = Linear(input_dim, h1, rng)
        self.fc2 = Linear(h1, h2, rng)
        self.fc3 = Linear(h2, num_classes, rng)
        self.activation = activation.lower()

    def _act(self, x):
        if self.activation == "relu":
            return x.relu()
        if self.activation == "sigmoid":
            return x.sigmoid()
        if self.activation == "tanh":
            return x.tanh()
        raise ValueError(f"Unsupported activation: {self.activation}")

    def __call__(self, x):
        x = self._act(self.fc1(x))
        x = self._act(self.fc2(x))
        return self.fc3(x)

    def parameters(self):
        return self.fc1.parameters() + self.fc2.parameters() + self.fc3.parameters()

    def state_dict(self):
        return {
            "fc1_W": self.fc1.W.data.copy(),
            "fc1_b": self.fc1.b.data.copy(),
            "fc2_W": self.fc2.W.data.copy(),
            "fc2_b": self.fc2.b.data.copy(),
            "fc3_W": self.fc3.W.data.copy(),
            "fc3_b": self.fc3.b.data.copy(),
            "activation": self.activation,
        }

    def load_state_dict(self, state):
        self.fc1.W.data[...] = state["fc1_W"]
        self.fc1.b.data[...] = state["fc1_b"]
        self.fc2.W.data[...] = state["fc2_W"]
        self.fc2.b.data[...] = state["fc2_b"]
        self.fc3.W.data[...] = state["fc3_W"]
        self.fc3.b.data[...] = state["fc3_b"]
        self.activation = state.get("activation", self.activation)
