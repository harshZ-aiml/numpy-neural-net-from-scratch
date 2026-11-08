"""Optimizers on the flat parameter vector (pure NumPy)."""
import numpy as np


class SGD:
    def __init__(self, lr=0.5):
        self.lr = lr

    def step(self, w, g):
        return w - self.lr * g


class Momentum:
    def __init__(self, lr=0.1, beta=0.9):
        self.lr, self.beta, self.v = lr, beta, None

    def step(self, w, g):
        self.v = np.zeros_like(w) if self.v is None else self.v
        self.v = self.beta * self.v + g
        return w - self.lr * self.v


class Adam:
    def __init__(self, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, w, g):
        if self.m is None:
            self.m, self.v = np.zeros_like(w), np.zeros_like(w)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g ** 2
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return w - self.lr * mh / (np.sqrt(vh) + self.eps)


OPTIMIZERS = {
    "SGD": lambda: SGD(0.5),
    "Momentum": lambda: Momentum(0.1, 0.9),
    "Adam": lambda: Adam(0.05),
}
