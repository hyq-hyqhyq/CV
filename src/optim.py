class SGD:
    # SGD optimizer with optional weight decay.
    def __init__(self, params, lr=0.1, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * p.data
            p.data -= self.lr * grad

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


class ExponentialLRScheduler:
    # Multiply LR by gamma each epoch.
    def __init__(self, optimizer, gamma=0.95):
        self.optimizer = optimizer
        self.gamma = gamma

    def step(self):
        self.optimizer.lr *= self.gamma
