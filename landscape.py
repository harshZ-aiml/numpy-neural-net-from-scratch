"""Loss landscape + gradient descent path.

The 2D plane is chosen with PCA of the training trajectory, so the path
is well represented on it. (Random directions also work but the path
usually looks like a tiny squiggle.)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from nn import unpack, loss_fn


def pca_plane(path):
    center = path[-1]
    _, _, vt = np.linalg.svd(path - center, full_matrices=False)
    return center, vt[0], vt[1]


def compute_landscape(path, X, y, grid=60, margin=0.3):
    center, d1, d2 = pca_plane(path)
    coords = (path - center) @ np.stack([d1, d2]).T  # path in plane coordinates
    lo = coords.min(0) - margin * np.ptp(coords, axis=0)
    hi = coords.max(0) + margin * np.ptp(coords, axis=0)
    a = np.linspace(lo[0], hi[0], grid)
    b = np.linspace(lo[1], hi[1], grid)
    Z = np.zeros((grid, grid))
    for i, bi in enumerate(b):
        for j, aj in enumerate(a):
            Z[i, j] = loss_fn(unpack(center + aj * d1 + bi * d2), X, y)
    path_loss = np.array([loss_fn(unpack(w), X, y) for w in path])
    return a, b, Z, coords, path_loss


def plot_landscape(path, X, y, out_dir="results"):
    a, b, Z, coords, path_loss = compute_landscape(path, X, y)
    A, B = np.meshgrid(a, b)
    step = max(1, len(coords) // 150)

    # 2D contour with path
    fig, ax = plt.subplots(figsize=(7, 5.5))
    cs = ax.contourf(A, B, Z, levels=30, cmap="viridis")
    fig.colorbar(cs, label="loss")
    ax.plot(coords[::step, 0], coords[::step, 1], "r.-", lw=1.5, ms=3, label="gradient descent path")
    ax.scatter(*coords[0], c="white", edgecolors="k", s=80, zorder=5, label="start")
    ax.scatter(*coords[-1], c="red", edgecolors="k", s=80, marker="*", zorder=5, label="end")
    ax.set_xlabel("PCA direction 1"); ax.set_ylabel("PCA direction 2")
    ax.set_title("Loss landscape and gradient descent path")
    ax.legend()
    fig.tight_layout(); fig.savefig(f"{out_dir}/landscape_contour.png", dpi=150); plt.close(fig)

    # 3D surface with path
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(A, B, Z, cmap="viridis", alpha=0.75, linewidth=0)
    ax.plot(coords[::step, 0], coords[::step, 1], path_loss[::step], "r-", lw=2.5)
    ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2"); ax.set_zlabel("loss")
    ax.set_title("Loss surface (3D)")
    fig.tight_layout(); fig.savefig(f"{out_dir}/landscape_3d.png", dpi=150); plt.close(fig)

    return a, b, Z, coords


def make_gif(a, b, Z, coords, out_dir="results", frames=60):
    A, B = np.meshgrid(a, b)
    idx = np.unique(np.geomspace(1, len(coords), frames).astype(int)) - 1
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(A, B, Z, levels=30, cmap="viridis")
    line, = ax.plot([], [], "r-", lw=1.5)
    dot, = ax.plot([], [], "ro", ms=8, mec="k")
    ax.set_xlabel("PCA direction 1"); ax.set_ylabel("PCA direction 2")
    ax.set_title("Gradient descent on the loss landscape")

    def update(k):
        j = idx[k]
        line.set_data(coords[:j + 1, 0], coords[:j + 1, 1])
        dot.set_data([coords[j, 0]], [coords[j, 1]])
        return line, dot

    anim = FuncAnimation(fig, update, frames=len(idx), blit=True)
    anim.save(f"{out_dir}/descent.gif", writer=PillowWriter(fps=12))
    plt.close(fig)
