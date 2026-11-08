"""Optimizer comparison: SGD vs Momentum vs Adam.
- same init, same data, same loss landscape (shared PCA plane)
- held-out test set, multi-seed statistics
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from nn import make_data, init_params, pack, unpack, gradients, loss_fn, accuracy
from optim import OPTIMIZERS

COLORS = {"SGD": "tab:red", "Momentum": "tab:orange", "Adam": "white"}


def train_opt(X, y, opt, epochs=2000, seed=0):
    w = pack(init_params(seed=seed))
    losses, path = [], [w.copy()]
    for _ in range(epochs):
        g = pack(gradients(unpack(w), X, y))
        w = opt.step(w, g)
        losses.append(loss_fn(unpack(w), X, y))
        path.append(w.copy())
    return w, np.array(losses), np.array(path)


def splits(seed=0):
    Xtr, ytr = make_data(200, seed=seed)
    Xte, yte = make_data(1000, seed=seed + 1000)
    return Xtr, ytr, Xte, yte


def multi_seed(n_seeds=10, epochs=2000, thresh=0.1):
    rows = {k: [] for k in OPTIMIZERS}
    for s in range(n_seeds):
        Xtr, ytr, Xte, yte = splits(s)
        for name, mk in OPTIMIZERS.items():
            w, losses, _ = train_opt(Xtr, ytr, mk(), epochs, seed=s)
            p = unpack(w)
            hit = np.argmax(losses < thresh) if (losses < thresh).any() else epochs
            rows[name].append((losses[-1], loss_fn(p, Xte, yte), accuracy(p, Xtr, ytr), accuracy(p, Xte, yte), hit))
    return {k: np.array(v) for k, v in rows.items()}


def report(stats, n_seeds):
    lines = [f"| Optimizer | Train loss | Test loss | Train acc | Test acc | Epochs to loss<0.1 |",
             "|---|---|---|---|---|---|"]
    for k, a in stats.items():
        m, sd = a.mean(0), a.std(0)
        lines.append(f"| {k} | {m[0]:.3f}±{sd[0]:.3f} | {m[1]:.3f}±{sd[1]:.3f} | "
                     f"{m[2]:.1%}±{sd[2]:.1%} | {m[3]:.1%}±{sd[3]:.1%} | {m[4]:.0f}±{sd[4]:.0f} |")
    table = "\n".join(lines)
    print(f"\nMean±std over {n_seeds} seeds (test set = 1000 unseen points)\n" + table)
    return table


def shared_plane(paths):
    allp = np.vstack(list(paths.values()))
    center = allp.mean(0)
    _, _, vt = np.linalg.svd(allp - center, full_matrices=False)
    return center, vt[0], vt[1]


def landscape_grid(center, d1, d2, coords_all, X, y, grid=60, margin=0.25):
    lo = coords_all.min(0) - margin * np.ptp(coords_all, axis=0)
    hi = coords_all.max(0) + margin * np.ptp(coords_all, axis=0)
    a, b = np.linspace(lo[0], hi[0], grid), np.linspace(lo[1], hi[1], grid)
    Z = np.array([[loss_fn(unpack(center + aj * d1 + bi * d2), X, y) for aj in a] for bi in b])
    return a, b, Z


def run(out="results", epochs=2000, n_seeds=10):
    X, y, Xte, yte = splits(0)
    res = {n: train_opt(X, y, mk(), epochs, seed=0) for n, mk in OPTIMIZERS.items()}
    paths = {n: r[2] for n, r in res.items()}

    # loss curves
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for n, (_, l, _) in res.items():
        ax.plot(l, label=n)
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("train loss (log)")
    ax.set_title("Optimizer comparison: training loss"); ax.grid(alpha=0.3); ax.legend()
    fig.tight_layout(); fig.savefig(f"{out}/compare_loss.png", dpi=150); plt.close(fig)

    # shared landscape
    center, d1, d2 = shared_plane(paths)
    M = np.stack([d1, d2]).T
    coords = {n: (p - center) @ M for n, p in paths.items()}
    a, b, Z = landscape_grid(center, d1, d2, np.vstack(list(coords.values())), X, y)
    A, B = np.meshgrid(a, b)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    cs = ax.contourf(A, B, Z, levels=30, cmap="viridis")
    fig.colorbar(cs, label="train loss")
    for n, c in coords.items():
        st = max(1, len(c) // 200)
        ax.plot(c[::st, 0], c[::st, 1], "-", color=COLORS[n], lw=2, label=n)
        ax.scatter(*c[-1], color=COLORS[n], edgecolors="k", marker="*", s=140, zorder=5)
    ax.scatter(*coords["SGD"][0], c="cyan", edgecolors="k", s=90, zorder=6, label="shared start")
    ax.set_xlabel("PCA 1 (all trajectories)"); ax.set_ylabel("PCA 2")
    ax.set_title("SGD vs Momentum vs Adam on the same loss landscape (2D slice of 33-D)")
    ax.title.set_fontsize(10); ax.legend()
    fig.tight_layout(); fig.savefig(f"{out}/compare_landscape.png", dpi=150); plt.close(fig)

    # GIF
    n_fr = 60
    idx = np.unique(np.geomspace(1, epochs + 1, n_fr).astype(int)) - 1
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.contourf(A, B, Z, levels=30, cmap="viridis")
    lines = {n: ax.plot([], [], "-", color=COLORS[n], lw=2, label=n)[0] for n in coords}
    dots = {n: ax.plot([], [], "o", color=COLORS[n], mec="k", ms=8)[0] for n in coords}
    ax.legend(loc="upper right"); ax.set_title("Optimizers racing down the landscape")
    ax.set_xlabel("PCA 1"); ax.set_ylabel("PCA 2")

    def upd(k):
        j = idx[k]
        for n, c in coords.items():
            lines[n].set_data(c[:j + 1, 0], c[:j + 1, 1])
            dots[n].set_data([c[j, 0]], [c[j, 1]])
        return list(lines.values()) + list(dots.values())

    FuncAnimation(fig, upd, frames=len(idx), blit=True).save(
        f"{out}/compare_descent.gif", writer=PillowWriter(fps=12))
    plt.close(fig)

    stats = multi_seed(n_seeds, epochs)
    table = report(stats, n_seeds)
    open(f"{out}/compare_table.md", "w").write(table + "\n")
    return stats


if __name__ == "__main__":
    run()
