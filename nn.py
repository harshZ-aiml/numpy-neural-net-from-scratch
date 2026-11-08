"""2-layer neural network from scratch in NumPy (tanh hidden layer, sigmoid output)."""
import numpy as np

HIDDEN = 8


def make_data(n=200, seed=0):
    """XOR-like data: label is 1 when x1 and x2 share the same sign."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    y = (X[:, 0] * X[:, 1] > 0).astype(float).reshape(-1, 1)
    return X, y


def init_params(hidden=HIDDEN, seed=0):
    rng = np.random.default_rng(seed + 1)
    return {
        "W1": rng.normal(size=(2, hidden)) * 0.5,
        "b1": np.zeros(hidden),
        "W2": rng.normal(size=(hidden, 1)) * 0.5,
        "b2": np.zeros(1),
    }


def pack(p):
    """Flatten parameters into one vector (needed for landscape + gradient check)."""
    return np.concatenate([p["W1"].ravel(), p["b1"], p["W2"].ravel(), p["b2"]])


def unpack(v, hidden=HIDDEN):
    i = 2 * hidden
    return {
        "W1": v[:i].reshape(2, hidden),
        "b1": v[i:i + hidden],
        "W2": v[i + hidden:i + 2 * hidden].reshape(hidden, 1),
        "b2": v[i + 2 * hidden:],
    }


def forward(p, X):
    h = np.tanh(X @ p["W1"] + p["b1"])
    prob = 1 / (1 + np.exp(-(h @ p["W2"] + p["b2"])))
    return h, prob


def loss_fn(p, X, y):
    _, prob = forward(p, X)
    eps = 1e-9
    return -np.mean(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps))


def gradients(p, X, y):
    """Manual backprop."""
    n = len(X)
    h, prob = forward(p, X)
    dz2 = (prob - y) / n
    dW2, db2 = h.T @ dz2, dz2.sum(0)
    dh = (dz2 @ p["W2"].T) * (1 - h ** 2)
    dW1, db1 = X.T @ dh, dh.sum(0)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def train(X, y, lr=0.5, epochs=2000, seed=0, verbose=True):
    """Full-batch gradient descent. Returns params, loss history, weight trajectory."""
    p = init_params(seed=seed)
    losses, path = [], [pack(p)]
    for i in range(epochs):
        g = gradients(p, X, y)
        for k in p:
            p[k] = p[k] - lr * g[k]
        losses.append(loss_fn(p, X, y))
        path.append(pack(p))
        if verbose and i % 200 == 0:
            print(f"epoch {i:5d}  loss {losses[-1]:.4f}")
    return p, np.array(losses), np.array(path)


def accuracy(p, X, y):
    _, prob = forward(p, X)
    return float(np.mean((prob > 0.5) == y))
