"""Loss curve and decision boundary."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nn import forward


def plot_loss(losses, out_dir="results"):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(losses)
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Training loss")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{out_dir}/loss_curve.png", dpi=150); plt.close(fig)


def plot_boundary(p, X, y, out_dir="results"):
    xx, yy = np.meshgrid(np.linspace(-3.5, 3.5, 300), np.linspace(-3.5, 3.5, 300))
    _, prob = forward(p, np.c_[xx.ravel(), yy.ravel()])
    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(xx, yy, prob.reshape(xx.shape), levels=20, cmap="RdBu", alpha=0.7)
    fig.colorbar(cs, label="P(class 1)")
    ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap="RdBu", edgecolors="k", s=25)
    ax.set_title("Learned decision boundary")
    fig.tight_layout(); fig.savefig(f"{out_dir}/decision_boundary.png", dpi=150); plt.close(fig)
