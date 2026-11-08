"""Verify hand-written backprop against numerical (finite-difference) gradients."""
import numpy as np
from nn import make_data, init_params, pack, unpack, loss_fn, gradients


def numerical_gradient(p, X, y, eps=1e-5):
    v = pack(p)
    g = np.zeros_like(v)
    for i in range(len(v)):
        vp, vm = v.copy(), v.copy()
        vp[i] += eps
        vm[i] -= eps
        g[i] = (loss_fn(unpack(vp), X, y) - loss_fn(unpack(vm), X, y)) / (2 * eps)
    return g


def run_check():
    X, y = make_data()
    p = init_params()
    analytic = pack(gradients(p, X, y))
    numeric = numerical_gradient(p, X, y)
    rel_err = np.linalg.norm(analytic - numeric) / (np.linalg.norm(analytic) + np.linalg.norm(numeric))
    print(f"Gradient check relative error: {rel_err:.2e}  ->", "PASS" if rel_err < 1e-6 else "FAIL")
    return rel_err


if __name__ == "__main__":
    run_check()
